from __future__ import annotations

import json

import torch

from controlled_run.maxrl import MaxRLRewardBatchRecorder, make_practical_maxrl_trainer
from controlled_run.signal_ledger import make_signal_ledger_trainer


class FakeAccelerator:
    process_index = 0

    def gather(self, tensor):
        return tensor


class FakeRankOneAccelerator:
    process_index = 1

    def __init__(self, global_rewards):
        self.global_rewards = global_rewards

    def gather(self, tensor):
        del tensor
        return self.global_rewards.clone()


class FakeModel:
    training = True


class FakeState:
    global_step = 7


class FakeBaseTrainer:
    def __init__(self, output_advantages, *, accelerator=None):
        self._output_advantages = output_advantages
        self.accelerator = accelerator if accelerator is not None else FakeAccelerator()
        self.model = FakeModel()
        self.state = FakeState()
        self.num_generations = 4
        self.num_generations_eval = 4

    def _generate_and_score_completions(self, inputs):
        del inputs
        return {"advantages": self._output_advantages.clone()}


class FakeLoggingBaseTrainer(FakeBaseTrainer):
    def __init__(self, output_advantages, *, global_logged_advantages, accelerator=None):
        super().__init__(output_advantages, accelerator=accelerator)
        self._global_logged_advantages = list(global_logged_advantages)
        self._logs = {"advantages": [999.0]}

    def _generate_and_score_completions(self, inputs):
        output = super()._generate_and_score_completions(inputs)
        self._logs["advantages"].extend(self._global_logged_advantages)
        return output


def _capture_group(recorder: MaxRLRewardBatchRecorder, rewards, *, dataset_index=10) -> None:
    recorder.capture(
        dataset_indices=[dataset_index] * len(rewards),
        correctness=[bool(reward) for reward in rewards],
        terminated=[True] * len(rewards),
        rewards=rewards,
        completion_lengths=[8] * len(rewards),
    )


def _ledger_advantages(tmp_path, *, rank=0):
    path = tmp_path / f"signal_ledger_test_rank{rank}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [row["advantage"] for row in rows]


def _wrapped_trainer(recorder, tmp_path):
    maxrl_trainer = make_practical_maxrl_trainer(FakeBaseTrainer, recorder=recorder)
    return make_signal_ledger_trainer(
        maxrl_trainer,
        recorder=recorder,
        ledger_dir=tmp_path,
        importance_sampling_clip_max=3.0,
        importance_sampling_mode="token_truncate",
        launch_timestamp="test",
    )


def test_maxrl_wrapper_replaces_advantages_before_existing_ledger_records_them(tmp_path):
    recorder = MaxRLRewardBatchRecorder()
    _capture_group(recorder, [1.0, 0.0, 0.0, 0.0])
    trainer_cls = _wrapped_trainer(recorder, tmp_path)
    trainer = trainer_cls(torch.tensor([10.0, 11.0, 12.0, 13.0]))

    output = trainer._generate_and_score_completions([])

    expected = torch.tensor([3.0, -1.0, -1.0, -1.0])
    assert torch.equal(output["advantages"], expected)
    assert _ledger_advantages(tmp_path) == [3.0, -1.0, -1.0, -1.0]


def test_maxrl_wrapper_uses_global_groups_then_returns_only_rank_slice(tmp_path):
    recorder = MaxRLRewardBatchRecorder()
    _capture_group(recorder, [1.0, 1.0, 0.0, 0.0], dataset_index=20)
    global_rewards = torch.tensor(
        [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        dtype=torch.float32,
    )
    trainer_cls = _wrapped_trainer(recorder, tmp_path)
    trainer = trainer_cls(
        torch.tensor([10.0, 11.0, 12.0, 13.0]),
        accelerator=FakeRankOneAccelerator(global_rewards),
    )

    output = trainer._generate_and_score_completions([])

    expected = torch.tensor([1.0, 1.0, -1.0, -1.0])
    assert torch.equal(output["advantages"], expected)
    assert _ledger_advantages(tmp_path, rank=1) == [1.0, 1.0, -1.0, -1.0]


def test_maxrl_wrapper_replaces_latest_trl_global_advantage_log_batch():
    recorder = MaxRLRewardBatchRecorder()
    _capture_group(recorder, [1.0, 1.0, 0.0, 0.0], dataset_index=20)
    global_rewards = torch.tensor(
        [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        dtype=torch.float32,
    )
    trainer_cls = make_practical_maxrl_trainer(FakeLoggingBaseTrainer, recorder=recorder)
    trainer = trainer_cls(
        torch.tensor([10.0, 11.0, 12.0, 13.0]),
        global_logged_advantages=[10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0],
        accelerator=FakeRankOneAccelerator(global_rewards),
    )

    trainer._generate_and_score_completions([])

    assert trainer._logs["advantages"] == [
        999.0,
        3.0,
        -1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
    ]


def test_maxrl_reward_peek_does_not_consume_batch():
    recorder = MaxRLRewardBatchRecorder()
    _capture_group(recorder, [1.0, 0.0, 0.0, 0.0])

    first = recorder.peek()
    second = recorder.peek()
    consumed = recorder.pop()

    assert first == second == consumed
