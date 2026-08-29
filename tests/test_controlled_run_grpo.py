from __future__ import annotations

import json
from pathlib import Path

import pytest

import controlled_run.train_grpo as train_grpo
from controlled_run.config import load_config


ROOT = Path(__file__).resolve().parents[1]


class FakeGRPOConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeTokenizer:
    def save_pretrained(self, output_dir):
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer.json").write_text("{}", encoding="utf-8")


class FakeModel:
    def save_pretrained(self, output_dir):
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "model.safetensors").write_bytes(b"weights")


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
    monkeypatch.setattr(
        train_grpo,
        "_load_grpo_classes",
        lambda: (FakeGRPOConfig, object, object),
    )
    config = load_config(ROOT / "controlled_run/configs/grpo_qwen3_0_6b.yaml")

    args = train_grpo.build_grpo_arguments(config, tmp_path / "trainer")

    kwargs = args.kwargs
    assert kwargs["num_generations"] == 16
    assert kwargs["generation_batch_size"] == 32
    assert kwargs["per_device_train_batch_size"] == 8
    assert kwargs["gradient_accumulation_steps"] == 4
    assert kwargs["max_completion_length"] == 1024
    assert kwargs["mask_truncated_completions"] is True
    assert "max_prompt_length" not in kwargs
    assert kwargs["vllm_max_model_length"] == 1536
    assert kwargs["use_vllm"] is True
    assert kwargs["vllm_mode"] == "colocate"
    assert kwargs["vllm_gpu_memory_utilization"] == 0.30
    assert kwargs["learning_rate"] == 1e-6
    assert kwargs["warmup_steps"] == 0.10
    assert "warmup_ratio" not in kwargs
    assert kwargs["beta"] == 0.0
    assert kwargs["epsilon"] == 0.2
    assert kwargs["num_iterations"] == 1
    assert kwargs["loss_type"] == "dapo"
    assert kwargs["scale_rewards"] == "group"
    assert kwargs["vllm_importance_sampling_correction"] is True
    assert kwargs["vllm_importance_sampling_mode"] == "sequence_mask"
    assert kwargs["vllm_importance_sampling_clip_max"] == 3.0
    assert "vllm_importance_sampling_cap" not in kwargs
    assert kwargs["report_to"] == "none"


def test_build_grpo_arguments_allows_scale_rewards_control(monkeypatch, tmp_path):
    monkeypatch.setattr(
        train_grpo,
        "_load_grpo_classes",
        lambda: (FakeGRPOConfig, object, object),
    )
    config = load_config(ROOT / "controlled_run/configs/grpo_qwen3_0_6b.yaml")
    config["scale_rewards"] = "none"

    args = train_grpo.build_grpo_arguments(config, tmp_path / "trainer")

    assert args.kwargs["scale_rewards"] == "none"


def test_policy_snapshot_callback_saves_each_target_once(tmp_path):
    callback = train_grpo.PolicySnapshotCallback(
        output_dir=tmp_path,
        tokenizer=FakeTokenizer(),
        pi0_lineage_id="pi0-lineage",
    )

    class State:
        max_steps = 100
        global_step = 0
        is_world_process_zero = True

    state = State()
    model = FakeModel()
    control = object()

    state.global_step = 5
    callback.on_step_end(None, state, control, model=model)
    state.global_step = 10
    callback.on_step_end(None, state, control, model=model)
    callback.on_step_end(None, state, control, model=model)

    assert (tmp_path / "pi_005" / "model.safetensors").exists()
    assert (tmp_path / "pi_010" / "model.safetensors").exists()
    metadata = json.loads(
        (tmp_path / "pi_010" / "policy_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata == {
        "actual_step": 10,
        "pi0_lineage_id": "pi0-lineage",
        "target_percentage": 10,
    }


def test_policy_snapshot_callback_does_not_write_on_nonzero_process(tmp_path):
    callback = train_grpo.PolicySnapshotCallback(
        output_dir=tmp_path,
        tokenizer=FakeTokenizer(),
        pi0_lineage_id="pi0-lineage",
    )

    class State:
        max_steps = 100
        global_step = 5
        is_world_process_zero = False

    callback.on_step_end(None, State(), object(), model=FakeModel())

    assert not list(tmp_path.glob("pi_*"))


def test_validate_pilot_steps_enforces_small_diagnostic_window():
    assert train_grpo.validate_pilot_steps("pilot", 20) == 20
    assert train_grpo.validate_pilot_steps("pilot", 50) == 50
    assert train_grpo.validate_pilot_steps("canonical", None) is None

    with pytest.raises(ValueError, match="20.*50"):
        train_grpo.validate_pilot_steps("pilot", 19)
    with pytest.raises(ValueError, match="20.*50"):
        train_grpo.validate_pilot_steps("pilot", 51)
    with pytest.raises(ValueError, match="canonical.*override"):
        train_grpo.validate_pilot_steps("canonical", 20)
