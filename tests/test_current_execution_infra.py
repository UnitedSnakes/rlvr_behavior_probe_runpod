from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_current_branch_builds_runpod_image():
    workflow = (
        REPO_ROOT
        / ".github"
        / "workflows"
        / "build-runpod-image.yml"
    ).read_text(encoding="utf-8")

    assert "- codex/signal-ledger" in workflow


def test_bootstrap_supports_exact_commit_gate():
    bootstrap = (
        REPO_ROOT / "docker" / "rlvr-bootstrap.sh"
    ).read_text(encoding="utf-8")

    assert "RLVR_EXPECT_COMMIT" in bootstrap
    assert "rev-parse HEAD" in bootstrap
    assert "does not match RLVR_EXPECT_COMMIT" in bootstrap


def test_bootstrap_supports_fail_closed_2xa40_preflight():
    bootstrap = (
        REPO_ROOT / "docker" / "rlvr-bootstrap.sh"
    ).read_text(encoding="utf-8")

    required = [
        "RLVR_RUN_2XA40_PREFLIGHT",
        "timeout 90s",
        "torchrun --nproc_per_node=2",
        "-m controlled_run.distributed_preflight",
        "/workspace/rlvr-2xa40-preflight.json",
    ]

    for text in required:
        assert text in bootstrap

    assert "NCCL_P2P_DISABLE" not in bootstrap
