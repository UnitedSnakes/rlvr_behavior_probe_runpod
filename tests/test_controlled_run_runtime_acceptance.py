from __future__ import annotations

import pytest

from controlled_run.runtime_acceptance import validate_a40_runtime


def _status(**overrides):
    status = {
        "python_version": "3.12.13",
        "cuda_available": True,
        "torch_cuda": "13.0",
        "gpu_name": "NVIDIA A40",
        "flash_attention_2_available": True,
        "packages": {
            "torch": "2.13.0+cu130",
            "transformers": "5.15.0",
            "datasets": "5.0.1",
            "accelerate": "1.14.0",
            "trl": "0.28.0",
            "vllm": "0.27.1",
        },
    }
    status.update(overrides)
    return status


def test_validate_a40_runtime_accepts_flash_attention_2_runtime():
    status = validate_a40_runtime(_status(), required_attention_backend="flash_attention_2")

    assert status["gpu_name"] == "NVIDIA A40"
    assert status["flash_attention_2_available"] is True


def test_validate_a40_runtime_rejects_missing_flash_attention_2():
    with pytest.raises(RuntimeError, match="FlashAttention2.*unavailable"):
        validate_a40_runtime(
            _status(flash_attention_2_available=False),
            required_attention_backend="flash_attention_2",
        )


def test_validate_a40_runtime_rejects_non_cuda_host():
    with pytest.raises(RuntimeError, match="CUDA"):
        validate_a40_runtime(
            _status(cuda_available=False, gpu_name="unavailable"),
            required_attention_backend="flash_attention_2",
        )


def test_validate_a40_runtime_rejects_wrong_gpu_family():
    with pytest.raises(RuntimeError, match="A40"):
        validate_a40_runtime(
            _status(gpu_name="NVIDIA RTX 4090"),
            required_attention_backend="flash_attention_2",
        )
