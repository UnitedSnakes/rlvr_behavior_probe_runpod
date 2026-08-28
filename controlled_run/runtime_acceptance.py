from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from typing import Any


def collect_runtime_status() -> dict[str, Any]:
    import accelerate
    import datasets
    import torch
    import transformers
    import trl
    import vllm
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
            "trl": trl.__version__,
            "vllm": vllm.__version__,
            "flash-attn": _optional_distribution_version("flash-attn"),
        },
    }


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

    gpu_name = str(status.get("gpu_name", ""))
    if "A40" not in gpu_name:
        raise RuntimeError(
            f"A40 runtime acceptance requires an NVIDIA A40 GPU; found {gpu_name!r}"
        )

    packages = status.get("packages")
    if not isinstance(packages, dict):
        raise RuntimeError("A40 runtime acceptance is missing package-version metadata")
    for name in ("torch", "transformers", "datasets", "accelerate", "trl", "vllm"):
        if not packages.get(name):
            raise RuntimeError(f"A40 runtime acceptance is missing required package {name}")

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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Validate the CUDA/A40 runtime required by controlled training."
    )
    parser.add_argument(
        "--attention-backend",
        default="flash_attention_2",
        help="Canonical attention backend that must be available.",
    )
    args = parser.parse_args(argv)

    status = collect_runtime_status()
    print(json.dumps(status, indent=2, sort_keys=True))
    validate_a40_runtime(status, required_attention_backend=args.attention_backend)
    print("A40 runtime acceptance: PASS")


if __name__ == "__main__":
    main()
