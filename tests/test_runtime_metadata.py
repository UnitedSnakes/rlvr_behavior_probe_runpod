import importlib.metadata

import pytest

import probe.runtime as runtime


def test_platform_from_device_classifies_cuda_metal_and_cpu():
    assert runtime.platform_from_device("cuda") == "cuda"
    assert runtime.platform_from_device("cuda:0") == "cuda"
    assert runtime.platform_from_device("mps") == "metal"
    assert runtime.platform_from_device("cpu") == "cpu"


def test_runtime_implementation_matches_engine_and_device():
    assert runtime.runtime_implementation("hf", "cpu") == "transformers"
    assert runtime.runtime_implementation("vllm", "cuda") == "vllm-cuda"
    assert runtime.runtime_implementation("vllm", "mps") == "vllm-metal"

    with pytest.raises(ValueError, match="unsupported"):
        runtime.runtime_implementation("vllm", "cpu")


def test_collect_runtime_metadata_is_self_describing(monkeypatch):
    versions = {
        "torch": "2.test",
        "transformers": "5.test",
        "vllm": "0.27.1",
    }

    def fake_version(name):
        if name == "vllm-metal":
            raise importlib.metadata.PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(runtime.importlib.metadata, "version", fake_version)
    monkeypatch.setattr(runtime.stdlib_platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(runtime.stdlib_platform, "python_version", lambda: "3.12.test")

    metadata = runtime.collect_runtime_metadata("vllm", "cuda:0")

    assert metadata == {
        "platform": "cuda",
        "implementation": "vllm-cuda",
        "machine": "x86_64",
        "python_version": "3.12.test",
        "packages": {
            "torch": "2.test",
            "transformers": "5.test",
            "vllm": "0.27.1",
            "vllm-metal": None,
        },
        "tokenizer": {
            "name": "Qwen/Qwen2.5-1.5B-Instruct",
            "revision": "main",
        },
    }
