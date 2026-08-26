from __future__ import annotations

import importlib.metadata
import platform as stdlib_platform

from probe.prompts import TOKENIZER_NAME, TOKENIZER_REVISION


def platform_from_device(device: str) -> str:
    device_name = str(device)
    if device_name.startswith("cuda"):
        return "cuda"
    if device_name == "mps":
        return "metal"
    return "cpu"


def runtime_implementation(engine: str, device: str) -> str:
    platform_name = platform_from_device(device)

    if engine == "hf":
        return "transformers"
    if engine == "vllm" and platform_name == "cuda":
        return "vllm-cuda"
    if engine == "vllm" and platform_name == "metal":
        return "vllm-metal"

    raise ValueError(
        f"vLLM is unsupported on resolved device {device!r}; "
        "use CUDA or Apple Silicon Metal."
    )


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_runtime_metadata(engine: str, device: str) -> dict:
    return {
        "platform": platform_from_device(device),
        "implementation": runtime_implementation(engine, device),
        "machine": stdlib_platform.machine(),
        "python_version": stdlib_platform.python_version(),
        "packages": {
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "vllm": package_version("vllm"),
            "vllm-metal": package_version("vllm-metal"),
        },
        "tokenizer": {
            "name": TOKENIZER_NAME,
            "revision": TOKENIZER_REVISION,
        },
    }
