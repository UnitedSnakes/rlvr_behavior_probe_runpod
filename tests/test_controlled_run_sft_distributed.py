from __future__ import annotations

import pytest

from controlled_run.config import validate_sft_runtime_batch


def _config():
    return {
        "global_batch_size": 64,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 32,
    }


def test_canonical_two_gpu_sft_preserves_global_batch_64():
    info = validate_sft_runtime_batch(_config(), world_size=2, canonical=True)
    assert info == {
        "world_size": 2,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 32,
        "global_batch_size": 64,
    }


def test_canonical_sft_rejects_wrong_world_size():
    with pytest.raises(ValueError, match="exactly 2 GPUs"):
        validate_sft_runtime_batch(_config(), world_size=1, canonical=True)


def test_canonical_sft_rejects_changed_effective_batch():
    config = _config()
    config["gradient_accumulation_steps"] = 16
    with pytest.raises(ValueError, match="global batch size 64"):
        validate_sft_runtime_batch(config, world_size=2, canonical=True)


def test_single_gpu_smoke_allows_noncanonical_effective_batch():
    info = validate_sft_runtime_batch(_config(), world_size=1, canonical=False)
    assert info["world_size"] == 1
    assert info["global_batch_size"] == 32
