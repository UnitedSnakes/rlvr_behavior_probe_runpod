from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_macos_vllm_extras_do_not_override_runtime_owned_packages():
    requirements = (
        REPO_ROOT / "requirements-macos-vllm.txt"
    ).read_text(encoding="utf-8").lower()

    forbidden_prefixes = ("torch", "transformers", "vllm", "vllm-metal")
    lines = [line.strip() for line in requirements.splitlines() if line.strip()]
    for line in lines:
        assert not line.startswith(forbidden_prefixes)


def test_readme_documents_metal_runtime_boundary_and_exact_checkpoint():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    required = [
        "macOS 15+",
        "arm64",
        "Python 3.12",
        "~/.venv-vllm-metal",
        "vllm-project/vllm-metal/main/install.sh",
        "requirements-macos-vllm.txt",
        "checkpoint-8-of-10",
        "Metal results are for development, smoke tests, and small exploratory runs",
        "CUDA vLLM remains the canonical measurement backend",
    ]
    for text in required:
        assert text in readme
