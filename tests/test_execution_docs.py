from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_readme_documents_active_m5_and_2xa40_contract():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    required = [
        "M5 Pro development lane",
        "Active controlled 2×A40 RunPod workflow",
        "RLVR_BRANCH=codex/signal-ledger",
        "RLVR_EXPECT_COMMIT",
        "RLVR_RUN_2XA40_PREFLIGHT=1",
        "controlled_run.distributed_preflight",
        "controlled_run_outputs/",
        "NCCL_P2P_DISABLE=1",
        "controlled_run_outputs/sft/pi_0/pi_0",
    ]

    for text in required:
        assert text in readme


def test_claude_documents_current_maxrl_and_execution_contract():
    claude = (REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8")

    required = [
        "Practical MaxRL-15 is implemented",
        "2×A40 distributed NCCL preflight",
        "RLVR_EXPECT_COMMIT",
        "RLVR_RUN_2XA40_PREFLIGHT=1",
        "/workspace/rlvr-2xa40-preflight.json",
        "controlled_run_outputs/sft/pi_0/pi_0",
        "2026-09-03-m5-a40-execution-infra-amendment.md",
    ]

    for text in required:
        assert text in claude


def test_infra_amendment_preserves_scientific_boundary():
    amendment = (
        REPO_ROOT
        / "docs"
        / "superpowers"
        / "specs"
        / "2026-09-03-m5-a40-execution-infra-amendment.md"
    ).read_text(encoding="utf-8")

    assert "Operational infrastructure amendment only." in amendment
    assert "does not change the frozen GRPO or practical-MaxRL scientific" in amendment
    assert "A pod that fails the ordinary/default NCCL path is rejected" in amendment
