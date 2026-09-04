import base64
import os
import stat
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = REPO_ROOT / "docker" / "rlvr-bootstrap.sh"


def write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def install_fake_commands(fake_bin: Path) -> None:
    write_executable(
        fake_bin / "ssh-keygen",
        """#!/usr/bin/env bash
set -eu
if [ "${1:-}" = "-F" ]; then
    exit 1
fi
if [ "${1:-}" = "-y" ]; then
    exit 0
fi
exit 0
""",
    )
    write_executable(
        fake_bin / "ssh-keyscan",
        """#!/usr/bin/env bash
printf 'github.com ssh-ed25519 TEST-HOST-KEY\n'
""",
    )
    write_executable(
        fake_bin / "python",
        """#!/usr/bin/env bash
cat >/dev/null
printf 'python: fake-python\n'
printf 'torch: fake-torch\n'
printf 'cuda: fake-cuda\n'
printf 'vllm: 0.27.1\n'
printf 'gpu: fake-gpu\n'
printf 'HF_TOKEN set: %s\n' "${HF_TOKEN:+True}"
""",
    )


def install_fake_git(fake_bin: Path) -> None:
    write_executable(
        fake_bin / "git",
        r'''#!/usr/bin/env python3
import os
import sys
from pathlib import Path

args = sys.argv[1:]
log_path = Path(os.environ["FAKE_GIT_LOG"])
with log_path.open("a", encoding="utf-8") as log:
    log.write(" ".join(args) + "\n")

if args and args[0] == "clone":
    destination = Path(args[-1])
    destination.mkdir(parents=True, exist_ok=True)
    (destination / ".git").mkdir(exist_ok=True)
    raise SystemExit(0)

if len(args) >= 3 and args[0] == "-C":
    command = args[2:]
    if command[:2] == ["remote", "get-url"]:
        print(os.environ.get("FAKE_GIT_REMOTE", os.environ["RLVR_REPO"]))
        raise SystemExit(0)
    if command[:2] == ["status", "--porcelain"]:
        if os.environ.get("FAKE_GIT_DIRTY") == "1":
            print(" M tracked.txt")
        raise SystemExit(0)
    if command and command[0] == "fetch":
        raise SystemExit(0)
    if command[:2] == ["show-ref", "--verify"]:
        raise SystemExit(0)
    if command and command[0] in {"checkout", "branch", "merge"}:
        raise SystemExit(0)
    if command[:3] == ["rev-parse", "--short", "HEAD"]:
        print("deadbee")
        raise SystemExit(0)
    if command[:2] == ["rev-parse", "--abbrev-ref"]:
        print(os.environ["RLVR_BRANCH"])
        raise SystemExit(0)

raise SystemExit(0)
''',
    )


def bootstrap_env(tmp_path: Path) -> dict[str, str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    install_fake_commands(fake_bin)
    install_fake_git(fake_bin)

    secret = base64.b64encode(b"fake-private-key").decode("ascii")
    env = os.environ.copy()

    # Bootstrap tests must not inherit optional RunPod execution gates from
    # the host environment. Individual tests opt into these explicitly.
    env.pop("RLVR_EXPECT_COMMIT", None)
    env.pop("RLVR_RUN_2XA40_PREFLIGHT", None)

    env.update(
        {
            "HOME": str(tmp_path / "home"),
            "PATH": f"{fake_bin}:{env['PATH']}",
            "GITHUB_DEPLOY_KEY_B64": secret,
            "RLVR_REPO": "git@github.com:UnitedSnakes/rlvr_behavior_probe_runpod.git",
            "RLVR_BRANCH": "difficulty-bin-analysis",
            "RLVR_REPO_DIR": str(tmp_path / "repo"),
            "FAKE_GIT_LOG": str(tmp_path / "git.log"),
        }
    )
    return env


def run_bootstrap(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(BOOTSTRAP)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_missing_deploy_key_fails_without_echoing_secrets(tmp_path):
    env = bootstrap_env(tmp_path)
    env.pop("GITHUB_DEPLOY_KEY_B64")

    result = run_bootstrap(env)

    assert result.returncode != 0
    assert "GITHUB_DEPLOY_KEY_B64" in result.stderr


def test_invalid_base64_fails_before_git(tmp_path):
    env = bootstrap_env(tmp_path)
    env["GITHUB_DEPLOY_KEY_B64"] = "definitely%not-base64"

    result = run_bootstrap(env)

    assert result.returncode != 0
    assert not Path(env["FAKE_GIT_LOG"]).exists()
    assert "definitely%not-base64" not in result.stdout + result.stderr


def test_initial_clone_installs_key_and_checks_out_requested_branch(tmp_path):
    env = bootstrap_env(tmp_path)
    secret = env["GITHUB_DEPLOY_KEY_B64"]

    result = run_bootstrap(env)

    assert result.returncode == 0, result.stderr
    key_path = Path(env["HOME"]) / ".ssh" / "id_ed25519"
    assert key_path.read_bytes() == b"fake-private-key"
    assert (Path(env["RLVR_REPO_DIR"]) / ".git").is_dir()
    git_log = Path(env["FAKE_GIT_LOG"]).read_text(encoding="utf-8")
    assert "clone --branch difficulty-bin-analysis --single-branch" in git_log
    assert secret not in result.stdout + result.stderr


def test_existing_non_git_destination_is_not_overwritten(tmp_path):
    env = bootstrap_env(tmp_path)
    repo_dir = Path(env["RLVR_REPO_DIR"])
    repo_dir.mkdir(parents=True)
    sentinel = repo_dir / "keep-me.txt"
    sentinel.write_text("important", encoding="utf-8")

    result = run_bootstrap(env)

    assert result.returncode != 0
    assert sentinel.read_text(encoding="utf-8") == "important"
    assert "not a Git repository" in result.stderr


def test_dirty_existing_checkout_fails_before_checkout_or_merge(tmp_path):
    env = bootstrap_env(tmp_path)
    repo_dir = Path(env["RLVR_REPO_DIR"])
    (repo_dir / ".git").mkdir(parents=True)
    env["FAKE_GIT_DIRTY"] = "1"

    result = run_bootstrap(env)

    assert result.returncode != 0
    git_log = Path(env["FAKE_GIT_LOG"]).read_text(encoding="utf-8")
    assert "status --porcelain" in git_log
    assert " checkout " not in f" {git_log} "
    assert " merge " not in f" {git_log} "
    assert "reset --hard" not in git_log
    assert "clean" not in git_log


def test_existing_checkout_fetches_and_fast_forwards_safely(tmp_path):
    env = bootstrap_env(tmp_path)
    repo_dir = Path(env["RLVR_REPO_DIR"])
    (repo_dir / ".git").mkdir(parents=True)

    result = run_bootstrap(env)

    assert result.returncode == 0, result.stderr
    git_log = Path(env["FAKE_GIT_LOG"]).read_text(encoding="utf-8")
    assert "fetch origin +refs/heads/difficulty-bin-analysis:refs/remotes/origin/difficulty-bin-analysis" in git_log
    assert "checkout difficulty-bin-analysis" in git_log
    assert "merge --ff-only origin/difficulty-bin-analysis" in git_log


def test_bootstrap_is_idempotent_on_second_run(tmp_path):
    env = bootstrap_env(tmp_path)

    first = run_bootstrap(env)
    second = run_bootstrap(env)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    git_log = Path(env["FAKE_GIT_LOG"]).read_text(encoding="utf-8")
    assert git_log.count("clone --branch") == 1
    assert "fetch origin" in git_log


def test_remote_mismatch_fails_without_rewriting_origin(tmp_path):
    env = bootstrap_env(tmp_path)
    repo_dir = Path(env["RLVR_REPO_DIR"])
    (repo_dir / ".git").mkdir(parents=True)
    env["FAKE_GIT_REMOTE"] = "git@github.com:someone/other-repo.git"

    result = run_bootstrap(env)

    assert result.returncode != 0
    assert "origin does not match RLVR_REPO" in result.stderr
    git_log = Path(env["FAKE_GIT_LOG"]).read_text(encoding="utf-8")
    assert "remote set-url" not in git_log
