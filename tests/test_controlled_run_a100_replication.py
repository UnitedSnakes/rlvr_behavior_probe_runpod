from pathlib import Path

import pytest

from controlled_run.config import (
    build_grpo_1xa100_replication_config,
    load_config,
    validate_grpo_1xa100_replication_runtime_batch,
)
from controlled_run.runtime_acceptance import validate_a100_replication_runtime
import controlled_run.train_grpo_replication as grpo_rep
import controlled_run.train_maxrl_replication as maxrl_rep


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "controlled_run/configs/grpo_qwen3_0_6b.yaml"


def test_single_a100_replication_preserves_batch_geometry():
    canonical = load_config(CONFIG)
    replication = build_grpo_1xa100_replication_config(canonical, seed=43)

    assert canonical["canonical_world_size"] == 2
    assert canonical["per_device_train_batch_size"] == 4
    assert canonical["seed"] == 42

    assert replication["canonical_world_size"] == 1
    assert replication["per_device_train_batch_size"] == 8
    assert replication["gradient_accumulation_steps"] == 4
    assert replication["gradient_checkpointing"] is True
    assert replication["vllm_gpu_memory_utilization"] == 0.30
    assert replication["seed"] == 43

    geometry = validate_grpo_1xa100_replication_runtime_batch(
        replication,
        world_size=1,
    )
    assert geometry == {
        "world_size": 1,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 4,
        "global_optimizer_batch_size": 32,
        "generation_batch_size": 32,
        "steps_per_generation": 4,
        "num_generations": 16,
        "unique_prompts_per_generation_batch": 2,
    }


def test_single_a100_replication_rejects_world_size_two():
    canonical = load_config(CONFIG)
    replication = build_grpo_1xa100_replication_config(canonical, seed=44)
    with pytest.raises(ValueError, match="WORLD_SIZE=1"):
        validate_grpo_1xa100_replication_runtime_batch(
            replication,
            world_size=2,
        )


def test_single_a100_replication_rejects_unfrozen_seed():
    canonical = load_config(CONFIG)
    with pytest.raises(ValueError, match="42, 43, 44"):
        build_grpo_1xa100_replication_config(canonical, seed=45)


def _runtime_status(**overrides):
    status = {
        "python_version": "3.12.13",
        "cuda_available": True,
        "torch_cuda": "13.0",
        "gpu_name": "NVIDIA A100-SXM4-80GB",
        "visible_cuda_devices": 1,
        "gpu_total_memory_gib": 79.1,
        "bf16_supported": True,
        "flash_attention_2_available": True,
        "packages": {
            "torch": "2.13.0+cu130",
            "transformers": "5.15.0",
            "datasets": "5.0.1",
            "accelerate": "1.14.0",
            "trl": "1.12.0",
            "vllm": "0.27.1",
            "flash-attn": "2.8.3.post1",
        },
    }
    status.update(overrides)
    return status


def test_single_a100_runtime_gate_accepts_one_80gb_a100():
    accepted = validate_a100_replication_runtime(
        _runtime_status(),
        required_attention_backend="flash_attention_2",
    )
    assert "A100" in accepted["gpu_name"]


def test_single_a100_runtime_gate_rejects_multiple_visible_devices():
    with pytest.raises(RuntimeError, match="exactly one visible CUDA device"):
        validate_a100_replication_runtime(
            _runtime_status(visible_cuda_devices=2),
            required_attention_backend="flash_attention_2",
        )


def test_grpo_replication_delegates_frozen_metadata(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(
        grpo_rep,
        "_run_controlled_grpo",
        lambda **kwargs: captured.update(kwargs) or {"ok": True},
    )
    result = grpo_rep.run_grpo_replication(
        config_path=CONFIG,
        pi0_dir=tmp_path / "pi0",
        output_dir=tmp_path / "grpo43",
        seed=43,
        pilot_steps=20,
    )
    assert result == {"ok": True}
    assert captured["mode"] == "pilot"
    assert captured["pilot_steps"] == 20
    metadata = captured["manifest_extra"]["replication"]
    assert metadata["training_seed"] == 43
    assert metadata["hardware_contract"].startswith("1x NVIDIA A100 80GB")
    assert metadata["execution_topology"] == "single_process_single_gpu"
    assert "gradient_checkpointing" not in metadata["changed_from_seed42_a40_canonical"]


def test_maxrl_replication_delegates_frozen_objective(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(
        maxrl_rep,
        "_run_controlled_grpo",
        lambda **kwargs: captured.update(kwargs) or {"ok": True},
    )
    result = maxrl_rep.run_maxrl_replication(
        config_path=CONFIG,
        pi0_dir=tmp_path / "pi0",
        output_dir=tmp_path / "maxrl44",
        seed=44,
        pilot_steps=20,
    )
    assert result == {"ok": True}
    assert captured["mode"] == "pilot"
    assert captured["manifest_extra"]["objective"]["objective_family"] == "MaxRL"
    assert captured["manifest_extra"]["replication"]["training_seed"] == 44
