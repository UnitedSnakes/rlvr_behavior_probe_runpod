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
