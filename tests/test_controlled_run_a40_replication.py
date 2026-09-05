from pathlib import Path

import pytest

from controlled_run.config import (
    build_grpo_a40_replication_config,
    load_config,
    validate_grpo_a40_replication_runtime_batch,
)


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


