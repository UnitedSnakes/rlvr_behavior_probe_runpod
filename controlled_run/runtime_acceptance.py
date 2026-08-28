from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import math
import platform
from typing import Any

from controlled_run.constants import BASE_MODEL


CANONICAL_BASE_MODEL_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
EXPECTED_PACKAGES = {
    "torch": "2.13.0+cu130",
    "transformers": "5.15.0",
    "datasets": "5.0.1",
    "accelerate": "1.14.0",
    "trl": "1.12.0",
    "vllm": "0.27.1",
    "flash-attn": "2.8.3.post1",
}


def collect_runtime_status() -> dict[str, Any]:
    import accelerate
    import datasets
    import torch
    import transformers
    from transformers.utils import is_flash_attn_2_available

    return {
        "python_version": platform.python_version(),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable",
        "flash_attention_2_available": bool(is_flash_attn_2_available()),
        "packages": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "datasets": datasets.__version__,
            "accelerate": accelerate.__version__,
            "trl": _optional_import_version("trl"),
            "vllm": _optional_import_version("vllm"),
            "flash-attn": _optional_distribution_version("flash-attn"),
        },
    }


def _optional_import_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except (ImportError, ModuleNotFoundError):
        return None
    version = getattr(module, "__version__", None)
    if version:
        return str(version)
    return _optional_distribution_version(name)


def _optional_distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def validate_a40_runtime(
    status: dict[str, Any],
    *,
    required_attention_backend: str,
) -> dict[str, Any]:
    if not status.get("cuda_available") or not status.get("torch_cuda"):
        raise RuntimeError("A40 runtime acceptance requires a working CUDA runtime")

    if str(status.get("torch_cuda")) != "13.0":
        raise RuntimeError(
            "A40 runtime acceptance requires CUDA 13.0; "
            f"found {status.get('torch_cuda')!r}"
        )

    gpu_name = str(status.get("gpu_name", ""))
    if "A40" not in gpu_name:
        raise RuntimeError(
            f"A40 runtime acceptance requires an NVIDIA A40 GPU; found {gpu_name!r}"
        )

    packages = status.get("packages")
    if not isinstance(packages, dict):
        raise RuntimeError("A40 runtime acceptance is missing package-version metadata")

    for name, expected in EXPECTED_PACKAGES.items():
        actual = packages.get(name)
        if not actual:
            raise RuntimeError(f"A40 runtime acceptance is missing required package {name}")
        if str(actual) != expected:
            raise RuntimeError(
                f"A40 runtime acceptance requires {name} {expected}; found {actual}"
            )

    if required_attention_backend == "flash_attention_2":
        if status.get("flash_attention_2_available") is not True:
            raise RuntimeError(
                "Canonical SFT requires FlashAttention2, but the current A40 runtime "
                "reports it unavailable. Do not silently fall back to another attention backend."
            )
    elif required_attention_backend:
        raise RuntimeError(
            f"Unsupported controlled attention backend for A40 acceptance: "
            f"{required_attention_backend!r}"
        )

    return status


def run_model_probe(
    *,
    model_name: str = BASE_MODEL,
    model_revision: str = CANONICAL_BASE_MODEL_REVISION,
    attention_backend: str = "flash_attention_2",
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    result: dict[str, Any] = {
        "model_name": model_name,
        "model_revision": model_revision,
        "requested_attention_backend": attention_backend,
        "resolved_attention_backend": None,
        "forward_ok": False,
        "backward_ok": False,
        "loss": None,
    }
    model = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=model_revision,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=model_revision,
            dtype=torch.bfloat16,
            attn_implementation=attention_backend,
        )
        model.to("cuda")
        model.train()
        resolved = getattr(model.config, "_attn_implementation", None)
        if resolved is None:
            resolved = getattr(model.config, "_attn_implementation_internal", None)
        result["resolved_attention_backend"] = resolved

        inputs = tokenizer(
            "Compute 1 + 1 and return the answer.",
            return_tensors="pt",
        )
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
        outputs = model(**inputs, labels=inputs["input_ids"])
        result["forward_ok"] = True
        loss = outputs.loss
        result["loss"] = float(loss.detach().float().cpu())
        loss.backward()
        torch.cuda.synchronize()
        result["backward_ok"] = True
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return result


def validate_model_probe_result(result: dict[str, Any]) -> dict[str, Any]:
    requested = result.get("requested_attention_backend")
    resolved = result.get("resolved_attention_backend")
    if requested != "flash_attention_2":
        raise RuntimeError(
            f"FA2 model probe must request flash_attention_2; found {requested!r}"
        )
    if resolved != "flash_attention_2":
        raise RuntimeError(
            "FA2 model probe resolved attention backend must remain flash_attention_2; "
            f"found {resolved!r}"
        )
    if result.get("forward_ok") is not True:
        raise RuntimeError(
            f"FA2 model probe forward pass failed: {result.get('error', 'unknown error')}"
        )
    if result.get("backward_ok") is not True:
        raise RuntimeError(
            f"FA2 model probe backward pass failed: {result.get('error', 'unknown error')}"
        )
    loss = result.get("loss")
    try:
        finite = math.isfinite(float(loss))
    except (TypeError, ValueError):
        finite = False
    if not finite:
        raise RuntimeError(f"FA2 model probe requires a finite loss; found {loss!r}")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Validate the CUDA/A40 runtime required by controlled training."
    )
    parser.add_argument(
        "--attention-backend",
        default="flash_attention_2",
        help="Canonical attention backend that must be available.",
    )
    parser.add_argument(
        "--probe-model",
        action="store_true",
        help=(
            "After static acceptance, load the pinned Qwen3 base model and require a "
            "real BF16 FlashAttention2 forward/backward pass."
        ),
    )
    args = parser.parse_args(argv)

    status = collect_runtime_status()
    print(json.dumps(status, indent=2, sort_keys=True))
    validate_a40_runtime(status, required_attention_backend=args.attention_backend)
    print("A40 runtime static acceptance: PASS")

    if args.probe_model:
        probe = run_model_probe(attention_backend=args.attention_backend)
        print(json.dumps({"model_probe": probe}, indent=2, sort_keys=True))
        validate_model_probe_result(probe)
        print("A40 FlashAttention2 model probe: PASS")


if __name__ == "__main__":
    main()
