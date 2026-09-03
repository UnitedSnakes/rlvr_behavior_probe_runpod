from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import controlled_run.train_grpo as train_grpo
from controlled_run.config import load_config


ROOT = Path(__file__).resolve().parents[1]


class FakeGRPOConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeTokenizer:
    eos_token_id = 1
    pad_token_id = 0

    def save_pretrained(self, output_dir):
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer.json").write_text("{}", encoding="utf-8")


class FakeModel:
    def save_pretrained(self, output_dir):
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "model.safetensors").write_bytes(b"weights")


class FakeTrainingTrainer:
    init_kwargs = None
    trained = False

    def __init__(self, **kwargs):
        type(self).init_kwargs = kwargs

    def train(self):
        type(self).trained = True


def test_progress_step_map_uses_exact_percentages_for_large_run():
    mapping = train_grpo.progress_step_map(1000)
    assert mapping[5] == 50
    assert mapping[25] == 250
    assert mapping[100] == 1000
    assert sorted(mapping) == list(range(5, 101, 5))


def test_progress_step_map_collapses_duplicate_steps_to_later_percentage():
    mapping = train_grpo.progress_step_map(7)
    assert list(mapping.values()) == sorted(set(mapping.values()))
    assert mapping[100] == 7
    assert max(mapping.values()) == 7


def test_build_grpo_arguments_maps_behavior_study_recipe(monkeypatch, tmp_path):
    monkeypatch.setattr(train_grpo, "_load_grpo_classes", lambda: (FakeGRPOConfig, object, object))
    config = load_config(ROOT / "controlled_run/configs/grpo_qwen3_0_6b.yaml")
    args = train_grpo.build_grpo_arguments(config, tmp_path / "trainer")
    kwargs = args.kwargs
    assert kwargs["num_generations"] == 16
    assert kwargs["generation_batch_size"] == 32
    assert kwargs["per_device_train_batch_size"] == 4
    assert kwargs["gradient_accumulation_steps"] == 4
    assert kwargs["max_completion_length"] == 2048
    assert kwargs["mask_truncated_completions"] is False
    assert kwargs["vllm_max_model_length"] == 2560
    assert kwargs["vllm_gpu_memory_utilization"] == 0.30
    assert kwargs["learning_rate"] == 1e-6
    assert kwargs["warmup_steps"] == 0.10
    assert kwargs["beta"] == 0.0
    assert kwargs["epsilon"] == 0.2
    assert kwargs["loss_type"] == "dapo"
    assert kwargs["scale_rewards"] == "group"
    assert kwargs["vllm_importance_sampling_clip_max"] == 3.0
    assert "canonical_world_size" not in kwargs
    assert "global_optimizer_batch_size" not in kwargs


def test_resolve_runtime_world_size(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert train_grpo.resolve_runtime_world_size() == 1
    monkeypatch.setenv("WORLD_SIZE", "2")
    assert train_grpo.resolve_runtime_world_size() == 2


def test_validate_pilot_steps_enforces_small_diagnostic_window():
    assert train_grpo.validate_pilot_steps("pilot", 20) == 20
    assert train_grpo.validate_pilot_steps("pilot", 500) == 500
    assert train_grpo.validate_pilot_steps("canonical", None) is None
    with pytest.raises(ValueError, match="20.*500"):
        train_grpo.validate_pilot_steps("pilot", 19)
    with pytest.raises(ValueError, match="20.*500"):
        train_grpo.validate_pilot_steps("pilot", 501)
    with pytest.raises(ValueError, match="canonical.*override"):
        train_grpo.validate_pilot_steps("canonical", 20)


def test_policy_snapshot_callback_saves_each_target_once(tmp_path):
    callback = train_grpo.PolicySnapshotCallback(tmp_path, FakeTokenizer(), "pi0-lineage")

    class State:
        max_steps = 100
        global_step = 5
        is_world_process_zero = True

    state = State()
    model = FakeModel()
    callback.on_step_end(None, state, object(), model=model)
    assert (tmp_path / "pi_005" / "model.safetensors").exists()


def test_controlled_grpo_core_applies_objective_hooks_before_existing_ledger(monkeypatch, tmp_path):
    config = load_config(ROOT / "controlled_run/configs/grpo_qwen3_0_6b.yaml")
    tokenizer = FakeTokenizer()
    model = FakeModel()
    recorder = object()
    captured = {}

    monkeypatch.setattr(
        train_grpo,
        "verify_pi0_for_grpo",
        lambda _path: {"manifest": {"kind": "pi0"}, "lineage_id": "pi0-lineage"},
    )
    monkeypatch.setattr(train_grpo, "load_config", lambda _path: dict(config))
    monkeypatch.setattr(train_grpo, "validate_grpo_config", lambda _config: None)
    monkeypatch.setattr(
        train_grpo,
        "validate_grpo_runtime_batch",
        lambda _config, world_size: {"world_size": world_size, "effective_batch": 32},
    )
    monkeypatch.setattr(train_grpo, "resolve_runtime_world_size", lambda: 2)
    monkeypatch.setattr(
        train_grpo,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda _path: tokenizer),
    )
    monkeypatch.setattr(
        train_grpo,
        "AutoModelForCausalLM",
        SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: model),
    )
    monkeypatch.setattr(train_grpo, "resolve_hf_revision", lambda *_args, **_kwargs: "gsm8k-sha")
    monkeypatch.setattr(train_grpo, "load_dataset", lambda *_args, **_kwargs: ["raw"])
    monkeypatch.setattr(
        train_grpo,
        "build_gsm8k_rl_rows",
        lambda _raw: [{"prompt": "q", "answer": "a", "dataset_index": 0}],
    )
    monkeypatch.setattr(
        train_grpo,
        "assert_prompt_token_limit",
        lambda *_args, **_kwargs: {"max_prompt_tokens": 12},
    )
    monkeypatch.setattr(train_grpo, "Dataset", SimpleNamespace(from_list=lambda rows: rows))
    monkeypatch.setattr(train_grpo, "build_grpo_arguments", lambda *_args, **_kwargs: "args")
    monkeypatch.setattr(
        train_grpo,
        "_load_grpo_classes",
        lambda: (FakeGRPOConfig, FakeTrainingTrainer, object),
    )
    monkeypatch.setattr(train_grpo, "resolve_terminal_token_ids", lambda _tokenizer: [1])

    def fake_reward(_terminal_ids, *, ledger_recorder):
        captured["reward_recorder"] = ledger_recorder
        return "reward"

    monkeypatch.setattr(train_grpo, "make_gsm8k_terminated_binary_reward", fake_reward)
    monkeypatch.setattr(train_grpo, "utc_launch_timestamp", lambda: "stamp")

    class ObjectiveTrainer(FakeTrainingTrainer):
        pass

    def trainer_transform(base_trainer, *, recorder):
        captured["transform_base"] = base_trainer
        captured["transform_recorder"] = recorder
        return ObjectiveTrainer

    def fake_ledger(base_trainer, **kwargs):
        captured["ledger_base"] = base_trainer
        captured["ledger_recorder"] = kwargs["recorder"]
        return base_trainer

    monkeypatch.setattr(train_grpo, "make_signal_ledger_trainer", fake_ledger)

    result = train_grpo._run_controlled_grpo(
        config_path=ROOT / "controlled_run/configs/grpo_qwen3_0_6b.yaml",
        pi0_dir=tmp_path / "pi0",
        output_dir=tmp_path / "run",
        mode="pilot",
        pilot_steps=20,
        recorder_factory=lambda: recorder,
        trainer_transform=trainer_transform,
        manifest_filename="maxrl_run_manifest.json",
        manifest_extra={"objective": {"advantage_estimator": "practical_maxrl"}},
        result_extra={"advantage_estimator": "practical_maxrl"},
    )

    assert captured["transform_base"] is FakeTrainingTrainer
    assert captured["ledger_base"] is ObjectiveTrainer
    assert captured["transform_recorder"] is recorder
    assert captured["ledger_recorder"] is recorder
    assert captured["reward_recorder"] is recorder
    assert ObjectiveTrainer.trained is True

    manifest_path = tmp_path / "run" / "maxrl_run_manifest.json"
    assert manifest_path.exists()
    assert not (tmp_path / "run" / "grpo_run_manifest.json").exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["objective"]["advantage_estimator"] == "practical_maxrl"
    assert manifest["config"]["num_generations"] == 16
    assert result["advantage_estimator"] == "practical_maxrl"
    assert result["pilot_steps"] == 20
