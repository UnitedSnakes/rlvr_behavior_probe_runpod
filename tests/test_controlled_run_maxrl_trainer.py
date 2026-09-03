from __future__ import annotations

import json

import pytest
import torch

from controlled_run.signal_ledger import RewardBatchRecorder, make_signal_ledger_trainer


class FakeAccelerator:
    process_index = 0

    def gather(self, tensor):
        return tensor


class FakeModel:
    training = True


class FakeState:
    global_step = 7


class FakeBaseTrainer:
    def __init__(self, output_advantages):
        self._output_advantages = output_advantages
        self.accelerator = FakeAccelerator()
        self.model = FakeModel()
        self.state = FakeState()
        self.num_generations = 4
        self.num_generations_eval = 4

    def _generate_and_score_completions(self, inputs):
        del inputs
        return {"advantages": self._output_advantages.clone()}


def _capture_k1_group(recorder: RewardBatchRecorder) -> None:
    recorder.capture(
        dataset_indices=[10, 10, 10, 10],
        correctness=[True, False, False, False],
        terminated=[True, True, True, True],
        rewards=[1.0, 0.0, 0.0, 0.0],
        completion_lengths=[8, 8, 8, 8],
    )


def _ledger_advantages(tmp_path):
    path = tmp_path / "signal_ledger_test_rank0.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [row["advantage"] for row in rows]


def test_signal_ledger_default_grpo_path_preserves_base_advantages(tmp_path):
    recorder = RewardBatchRecorder()
    _capture_k1_group(recorder)
    trainer_cls = make_signal_ledger_trainer(
        FakeBaseTrainer,
        recorder=recorder,
        ledger_dir=tmp_path,
        importance_sampling_clip_max=3.0,
        importance_sampling_mode="token_truncate",
        launch_timestamp="test",
    )
    base = torch.tensor([10.0, 11.0, 12.0, 13.0])
    trainer = trainer_cls(base)

    output = trainer._generate_and_score_completions([])

    assert torch.equal(output["advantages"], base)
    assert _ledger_advantages(tmp_path) == [10.0, 11.0, 12.0, 13.0]


def test_signal_ledger_practical_maxrl_replaces_advantages_before_loss_and_ledger(tmp_path):
    recorder = RewardBatchRecorder()
    _capture_k1_group(recorder)
    trainer_cls = make_signal_ledger_trainer(
        FakeBaseTrainer,
        recorder=recorder,
        ledger_dir=tmp_path,
        importance_sampling_clip_max=3.0,
        importance_sampling_mode="token_truncate",
        launch_timestamp="test",
        advantage_estimator="practical_maxrl",
    )
    trainer = trainer_cls(torch.tensor([10.0, 11.0, 12.0, 13.0]))

    output = trainer._generate_and_score_completions([])

    expected = torch.tensor([3.0, -1.0, -1.0, -1.0])
    assert torch.equal(output["advantages"], expected)
    assert _ledger_advantages(tmp_path) == [3.0, -1.0, -1.0, -1.0]


def test_signal_ledger_rejects_unknown_advantage_estimator(tmp_path):
    with pytest.raises(ValueError, match="advantage estimator"):
        make_signal_ledger_trainer(
            FakeBaseTrainer,
            recorder=RewardBatchRecorder(),
            ledger_dir=tmp_path,
            importance_sampling_clip_max=3.0,
            importance_sampling_mode="token_truncate",
            launch_timestamp="test",
            advantage_estimator="mystery",
        )
