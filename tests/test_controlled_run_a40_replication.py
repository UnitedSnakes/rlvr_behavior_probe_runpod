from pathlib import Path

import pytest

from controlled_run.config import (
    build_grpo_a40_replication_config,
    load_config,
    validate_grpo_a40_replication_runtime_batch,
)
import controlled_run.train_grpo_replication as grpo_rep
import controlled_run.train_maxrl_replication as maxrl_rep


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "controlled_run/configs/grpo_qwen3_0_6b.yaml"


def test_a40_replication_changes_only_seed():
    canonical = load_config(CONFIG)
    replication = build_grpo_a40_replication_config(canonical, seed=43)

    changed = {
        key for key in canonical
        if canonical[key] != replication[key]
    }
    assert changed == {"seed"}
    assert replication["seed"] == 43


def test_a40_replication_preserves_canonical_batch_geometry():
    canonical = load_config(CONFIG)
    replication = build_grpo_a40_replication_config(canonical, seed=44)

    geometry = validate_grpo_a40_replication_runtime_batch(
        replication,
        world_size=2,
    )
    assert geometry == {
        "world_size": 2,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "global_optimizer_batch_size": 32,
        "generation_batch_size": 32,
        "steps_per_generation": 4,
        "num_generations": 16,
        "unique_prompts_per_generation_batch": 2,
    }


def test_a40_replication_rejects_wrong_world_size():
    canonical = load_config(CONFIG)
    replication = build_grpo_a40_replication_config(canonical, seed=43)
    with pytest.raises(ValueError, match="WORLD_SIZE=2"):
        validate_grpo_a40_replication_runtime_batch(
            replication,
            world_size=1,
        )


def test_a40_replication_rejects_unfrozen_seed():
    canonical = load_config(CONFIG)
    with pytest.raises(ValueError, match="43, 44"):
        build_grpo_a40_replication_config(canonical, seed=45)


def test_grpo_replication_metadata_is_a40_matched(monkeypatch, tmp_path):
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
    metadata = captured["manifest_extra"]["replication"]
    assert metadata["hardware_contract"].startswith("2x NVIDIA A40")
    assert metadata["changed_from_seed42_a40_canonical"] == ["training_seed"]


def test_maxrl_replication_metadata_is_a40_matched(monkeypatch, tmp_path):
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
    assert captured["manifest_extra"]["objective"]["objective_family"] == "MaxRL"
    assert captured["manifest_extra"]["replication"]["training_seed"] == 44
