from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from controlled_run.runtime_acceptance import (
    collect_runtime_status,
    validate_a40_runtime,
    validate_model_probe_result,
)


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
            "trl": "1.12.0",
            "vllm": "0.27.1",
            "flash-attn": "2.8.3.post1",
        },
    }
    status.update(overrides)
    return status


def test_validate_a40_runtime_accepts_flash_attention_2_runtime():
    status = validate_a40_runtime(_status(), required_attention_backend="flash_attention_2")

    assert status["gpu_name"] == "NVIDIA A40"
    assert status["flash_attention_2_available"] is True


def test_validate_a40_runtime_rejects_wrong_trl_version():
    status = _status()
    status["packages"] = dict(status["packages"], trl="0.28.0")

    with pytest.raises(RuntimeError, match="trl.*1.12.0.*0.28.0"):
        validate_a40_runtime(status, required_attention_backend="flash_attention_2")


def test_validate_a40_runtime_rejects_missing_trl():
    status = _status()
    status["packages"] = dict(status["packages"], trl=None)

    with pytest.raises(RuntimeError, match="missing required package trl"):
        validate_a40_runtime(status, required_attention_backend="flash_attention_2")


def test_validate_a40_runtime_rejects_missing_compiled_flash_attn():
    status = _status()
    status["packages"] = dict(status["packages"], **{"flash-attn": None})

    with pytest.raises(RuntimeError, match="flash-attn"):
        validate_a40_runtime(status, required_attention_backend="flash_attention_2")


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


def test_collect_runtime_status_reports_missing_optional_training_packages(monkeypatch):
    real_import_module = importlib.import_module

    def fake_import_module(name: str):
        if name in {"trl", "vllm"}:
            raise ModuleNotFoundError(name)
        return real_import_module(name)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    status = collect_runtime_status()

    assert status["packages"]["trl"] is None
    assert status["packages"]["vllm"] is None


def test_a40_dockerfile_builds_flash_attention_for_sm80_with_bounded_parallelism():
    dockerfile = (
        Path(__file__).resolve().parents[1] / "docker" / "Dockerfile"
    ).read_text(encoding="utf-8")

    assert "FLASH_ATTN_CUDA_ARCHS=80" in dockerfile
    assert "FLASH_ATTENTION_FORCE_BUILD=TRUE" in dockerfile
    assert "MAX_JOBS=1" in dockerfile
    assert "NVCC_THREADS=1" in dockerfile
    assert "MAX_JOBS=4" not in dockerfile


def test_validate_model_probe_result_requires_real_fa2_forward_backward():
    result = validate_model_probe_result(
        {
            "model_name": "Qwen/Qwen3-0.6B-Base",
            "model_revision": "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
            "requested_attention_backend": "flash_attention_2",
            "resolved_attention_backend": "flash_attention_2",
            "forward_ok": True,
            "backward_ok": True,
            "loss": 1.25,
        }
    )

    assert result["backward_ok"] is True


@pytest.mark.parametrize(
    ("field", "value", "pattern"),
    [
        ("resolved_attention_backend", "sdpa", "resolved.*flash_attention_2"),
        ("forward_ok", False, "forward"),
        ("backward_ok", False, "backward"),
        ("loss", float("nan"), "finite"),
    ],
)
def test_validate_model_probe_result_rejects_failed_probe(field, value, pattern):
    probe = {
        "model_name": "Qwen/Qwen3-0.6B-Base",
        "model_revision": "da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
        "requested_attention_backend": "flash_attention_2",
        "resolved_attention_backend": "flash_attention_2",
        "forward_ok": True,
        "backward_ok": True,
        "loss": 1.25,
    }
    probe[field] = value

    with pytest.raises(RuntimeError, match=pattern):
        validate_model_probe_result(probe)
