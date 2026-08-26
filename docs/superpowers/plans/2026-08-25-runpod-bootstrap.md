# RunPod Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a fresh RunPod pod configure GitHub SSH access and prepare the requested research branch automatically while preserving RunPod SSH/Jupyter access when bootstrap fails.

**Architecture:** Add one idempotent shell executable to the GHCR image and invoke it from the RunPod template after starting `/start.sh` in the background. The bootstrap owns deploy-key installation and safe repository synchronization; the template startup wrapper deliberately absorbs bootstrap failure and leaves `/start.sh` as the long-lived service process. Push builds publish only a commit-SHA image tag; the stable `0.27.1` tag is promoted manually only after a real fresh-pod smoke test.

**Tech Stack:** Bash, Git/SSH, Docker, GitHub Actions, pytest, RunPod templates, GHCR

**Spec:** `docs/superpowers/specs/2026-08-25-runpod-bootstrap-design.md`

## Global Constraints

- Install the runtime executable at `/usr/local/bin/rlvr-bootstrap`.
- Keep the repository checkout outside the image; default checkout path is `/workspace/rlvr_behavior_probe_runpod`.
- Never bake or log `GITHUB_DEPLOY_KEY_B64`, the decoded private key, or `HF_TOKEN`.
- Required bootstrap variables are `GITHUB_DEPLOY_KEY_B64`, `RLVR_REPO`, and `RLVR_BRANCH`; `RLVR_REPO_DIR` defaults to `/workspace/rlvr_behavior_probe_runpod`.
- `HF_TOKEN` is not a bootstrap dependency; only report whether it is set.
- Never use `git reset --hard`, `git clean`, or another destructive working-tree operation.
- A standalone `rlvr-bootstrap` failure must return nonzero.
- A bootstrap failure during RunPod startup must not terminate `/start.sh` or make SSH/Jupyter unavailable.
- Preserve `/start.sh`; do not replace the RunPod base-image `ENTRYPOINT`.
- Keep `run_probe.py --upload-repo` explicit; do not add a default Hugging Face results repository to experiment code.
- Use `HKReporter/rlvr-behavior-probe-results` in the README upload example.
- Do not move the stable `0.27.1` image tag until a commit-SHA image has passed a fresh-pod startup test.

---

## File Structure

- Create `docker/rlvr-bootstrap.sh`: runtime bootstrap executable source; owns validation, SSH setup, safe Git synchronization, and non-secret runtime summary.
- Create `tests/test_runpod_bootstrap.py`: subprocess-level tests for bootstrap behavior using fake `git`, `ssh-keygen`, `ssh-keyscan`, and `python` commands; no live GitHub dependency.
- Create `tests/test_runpod_image_config.py`: static contract tests for Docker installation, SHA-first image publishing, stable-tag promotion gating, and README template documentation.
- Modify `docker/Dockerfile`: copy the bootstrap executable into the image and verify it is executable.
- Modify `.github/workflows/build-runpod-image.yml`: publish SHA tags on normal pushes and gate `0.27.1` behind an explicit manual `publish_stable` input.
- Modify `README.md`: document secrets, bootstrap variables, exact Container start command, log/recovery flow, SHA-first validation, and the correct Hugging Face Dataset namespace.

---

### Task 1: Build the idempotent bootstrap executable

**Files:**
- Create: `docker/rlvr-bootstrap.sh`
- Create: `tests/test_runpod_bootstrap.py`

**Interfaces:**
- Consumes environment variables:
  - `GITHUB_DEPLOY_KEY_B64: str` required
  - `RLVR_REPO: str` required
  - `RLVR_BRANCH: str` required
  - `RLVR_REPO_DIR: str` optional, default `/workspace/rlvr_behavior_probe_runpod`
  - `HF_TOKEN: str` optional; presence only
- Produces:
  - `/root/.ssh/id_ed25519` (or `$HOME/.ssh/id_ed25519` when `HOME` is overridden for tests)
  - `$HOME/.ssh/known_hosts` containing a GitHub host entry
  - a checkout at `RLVR_REPO_DIR`
  - exit code `0` on success, nonzero on bootstrap failure
  - safe stdout/stderr suitable for redirection to `/workspace/rlvr-bootstrap.log`

- [ ] **Step 1: Write subprocess test helpers and the first failing configuration tests**

Create `tests/test_runpod_bootstrap.py` with helpers that isolate `$HOME`, prepend a fake-command directory to `PATH`, and never call the live network:

```python
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
printf 'github.com ssh-ed25519 TEST-HOST-KEY\\n'
""",
    )
    write_executable(
        fake_bin / "python",
        """#!/usr/bin/env bash
cat >/dev/null
printf 'python: fake-python\\n'
printf 'torch: fake-torch\\n'
printf 'cuda: fake-cuda\\n'
printf 'vllm: 0.27.1\\n'
printf 'gpu: fake-gpu\\n'
printf 'HF_TOKEN set: %s\\n' "${HF_TOKEN:+True}"
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
```

- [ ] **Step 2: Run the focused tests and verify they fail because the bootstrap script does not exist yet**

Run:

```bash
python -m pytest -q tests/test_runpod_bootstrap.py
```

Expected: FAIL because `docker/rlvr-bootstrap.sh` is missing.

- [ ] **Step 3: Add behavioral tests for clone, occupied destination, dirty checkout, repeated execution, and secret redaction**

Append these tests to `tests/test_runpod_bootstrap.py`:

```python
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
```

- [ ] **Step 4: Implement the minimal bootstrap script**

Create `docker/rlvr-bootstrap.sh`:

```bash
#!/usr/bin/env bash
set -Eeuo pipefail

DEFAULT_REPO_DIR="/workspace/rlvr_behavior_probe_runpod"

log() {
    printf '[rlvr-bootstrap] %s\n' "$*"
}

fail() {
    printf '[rlvr-bootstrap] ERROR: %s\n' "$*" >&2
    exit 1
}

require_env() {
    local name="$1"
    if [ -z "${!name:-}" ]; then
        fail "$name is required"
    fi
}

install_deploy_key() {
    local ssh_dir="${HOME:-/root}/.ssh"
    local key_path="$ssh_dir/id_ed25519"
    local temp_key

    log "installing deploy key"
    mkdir -p "$ssh_dir"
    chmod 700 "$ssh_dir"
    temp_key="$(mktemp "$ssh_dir/id_ed25519.tmp.XXXXXX")"
    chmod 600 "$temp_key"

    if ! printf '%s' "$GITHUB_DEPLOY_KEY_B64" | base64 --decode > "$temp_key"; then
        rm -f "$temp_key"
        fail "GITHUB_DEPLOY_KEY_B64 is not valid base64"
    fi

    if ! ssh-keygen -y -f "$temp_key" >/dev/null 2>&1; then
        rm -f "$temp_key"
        fail "decoded GitHub deploy key is not a valid SSH private key"
    fi

    mv "$temp_key" "$key_path"
    chmod 600 "$key_path"
}

configure_known_hosts() {
    local ssh_dir="${HOME:-/root}/.ssh"
    local known_hosts="$ssh_dir/known_hosts"

    log "configuring GitHub host key"
    touch "$known_hosts"
    chmod 600 "$known_hosts"

    if ! ssh-keygen -F github.com -f "$known_hosts" >/dev/null 2>&1; then
        if ! ssh-keyscan github.com >> "$known_hosts" 2>/dev/null; then
            fail "could not obtain github.com SSH host key"
        fi
    fi
}

sync_repository() {
    local repo_dir="$1"
    local current_remote

    if [ ! -e "$repo_dir" ]; then
        log "cloning repository"
        mkdir -p "$(dirname "$repo_dir")"
        git clone \
            --branch "$RLVR_BRANCH" \
            --single-branch \
            "$RLVR_REPO" \
            "$repo_dir"
        return
    fi

    if [ ! -d "$repo_dir/.git" ]; then
        fail "$repo_dir exists but is not a Git repository"
    fi

    current_remote="$(git -C "$repo_dir" remote get-url origin)"
    if [ "$current_remote" != "$RLVR_REPO" ]; then
        fail "origin does not match RLVR_REPO; refusing to rewrite the remote"
    fi

    if [ -n "$(git -C "$repo_dir" status --porcelain)" ]; then
        fail "local changes are present; refusing to update the checkout"
    fi

    log "fetching requested branch"
    git -C "$repo_dir" fetch origin \
        "+refs/heads/$RLVR_BRANCH:refs/remotes/origin/$RLVR_BRANCH"

    if git -C "$repo_dir" show-ref --verify --quiet "refs/heads/$RLVR_BRANCH"; then
        git -C "$repo_dir" checkout "$RLVR_BRANCH"
    else
        git -C "$repo_dir" checkout \
            --track \
            -b "$RLVR_BRANCH" \
            "origin/$RLVR_BRANCH"
    fi

    git -C "$repo_dir" branch \
        --set-upstream-to="origin/$RLVR_BRANCH" \
        "$RLVR_BRANCH"
    git -C "$repo_dir" merge --ff-only "origin/$RLVR_BRANCH"
}

print_runtime_summary() {
    local repo_dir="$1"

    log "runtime summary"
    printf 'repository: %s\n' "$repo_dir"
    printf 'branch: %s\n' "$(git -C "$repo_dir" rev-parse --abbrev-ref HEAD)"
    printf 'commit: %s\n' "$(git -C "$repo_dir" rev-parse --short HEAD)"

    python - <<'PY'
import os
import sys

import torch
import vllm

print("python:", sys.executable)
print("python version:", sys.version.split()[0])
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("vllm:", vllm.__version__)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable")
print("HF_TOKEN set:", bool(os.environ.get("HF_TOKEN")))
PY
}

main() {
    local repo_dir

    log "checking configuration"
    require_env GITHUB_DEPLOY_KEY_B64
    require_env RLVR_REPO
    require_env RLVR_BRANCH
    repo_dir="${RLVR_REPO_DIR:-$DEFAULT_REPO_DIR}"

    install_deploy_key
    configure_known_hosts
    sync_repository "$repo_dir"
    print_runtime_summary "$repo_dir"
    log "bootstrap complete"
}

main "$@"
```

Implementation note: decode to a temporary file first, validate it, then atomically move it over `id_ed25519`. An invalid newly supplied Secret must not leave a partially decoded private-key file behind.

- [ ] **Step 5: Run the bootstrap tests and make only the minimal corrections required for them to pass**

Run:

```bash
python -m pytest -q tests/test_runpod_bootstrap.py
```

Expected: all tests in `tests/test_runpod_bootstrap.py` PASS.

- [ ] **Step 6: Run shell syntax validation**

Run:

```bash
bash -n docker/rlvr-bootstrap.sh
```

Expected: exit code `0` and no output.

- [ ] **Step 7: Commit Task 1**

```bash
git add docker/rlvr-bootstrap.sh tests/test_runpod_bootstrap.py
git commit -m "Add safe RunPod repository bootstrap"
```

---

### Task 2: Install bootstrap in the image and make image promotion SHA-first

**Files:**
- Create: `tests/test_runpod_image_config.py`
- Modify: `docker/Dockerfile`
- Modify: `.github/workflows/build-runpod-image.yml`

**Interfaces:**
- Consumes: `docker/rlvr-bootstrap.sh` from Task 1.
- Produces: executable `/usr/local/bin/rlvr-bootstrap` in the built image.
- Produces GHCR tags:
  - every push/manual build: `sha-<short-commit>`
  - manual dispatch with `publish_stable=true`: additionally `0.27.1`

- [ ] **Step 1: Write failing static contract tests for Docker installation and tag promotion**

Create `tests/test_runpod_image_config.py`:

```python
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_installs_executable_bootstrap():
    dockerfile = (REPO_ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")

    assert (
        "COPY --chmod=0755 docker/rlvr-bootstrap.sh "
        "/usr/local/bin/rlvr-bootstrap"
    ) in dockerfile
    assert "RUN test -x /usr/local/bin/rlvr-bootstrap" in dockerfile


def test_normal_push_does_not_automatically_replace_stable_image_tag():
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "build-runpod-image.yml"
    ).read_text(encoding="utf-8")

    assert "publish_stable:" in workflow
    assert "type=sha,prefix=sha-" in workflow
    assert (
        "type=raw,value=0.27.1,"
        "enable=${{ github.event_name == 'workflow_dispatch' && inputs.publish_stable }}"
    ) in workflow
```

- [ ] **Step 2: Run the static tests and verify they fail against the current Dockerfile/workflow**

Run:

```bash
python -m pytest -q tests/test_runpod_image_config.py
```

Expected: FAIL because the bootstrap copy and `publish_stable` gate are absent.

- [ ] **Step 3: Install the bootstrap script in the Docker image**

Add this after the package installation block and before the existing Python import/version verification in `docker/Dockerfile`:

```dockerfile
COPY --chmod=0755 docker/rlvr-bootstrap.sh /usr/local/bin/rlvr-bootstrap

RUN test -x /usr/local/bin/rlvr-bootstrap
```

Do not add a custom `ENTRYPOINT` or `CMD`.

- [ ] **Step 4: Gate the stable image tag behind a manual workflow input**

Change the workflow trigger header in `.github/workflows/build-runpod-image.yml` to:

```yaml
on:
  workflow_dispatch:
    inputs:
      publish_stable:
        description: Publish the tested 0.27.1 stable tag
        required: false
        default: false
        type: boolean
  push:
    branches:
      - difficulty-bin-analysis
    paths:
      - docker/**
      - .github/workflows/build-runpod-image.yml
```

Change only the tag block to:

```yaml
tags: |
  type=sha,prefix=sha-
  type=raw,value=0.27.1,enable=${{ github.event_name == 'workflow_dispatch' && inputs.publish_stable }}
```

This ensures the first image artifact for a code change is the traceable SHA tag. Stable promotion becomes a deliberate action after the RunPod test in Task 4.

- [ ] **Step 5: Run the static tests and bootstrap tests**

Run:

```bash
python -m pytest -q \
  tests/test_runpod_bootstrap.py \
  tests/test_runpod_image_config.py
```

Expected: PASS.

- [ ] **Step 6: Run the existing focused repository tests to catch integration regressions**

Run:

```bash
python -m pytest -q \
  tests/test_results_upload.py \
  tests/test_run_probe_modes.py \
  tests/test_vllm_sampler.py
```

Expected: the same 18 focused tests that passed on the fresh A40 pod remain green.

- [ ] **Step 7: Commit Task 2**

```bash
git add \
  docker/Dockerfile \
  .github/workflows/build-runpod-image.yml \
  tests/test_runpod_image_config.py
git commit -m "Install bootstrap and gate stable image promotion"
```

---

### Task 3: Document the exact RunPod template and recovery flow

**Files:**
- Modify: `README.md`
- Modify: `tests/test_runpod_image_config.py`

**Interfaces:**
- Consumes image executable `/usr/local/bin/rlvr-bootstrap` from Task 2.
- Produces a copy-pastable RunPod template contract with these exact values:
  - `HF_TOKEN={{ RUNPOD_SECRET_huggingface_token }}`
  - `GITHUB_DEPLOY_KEY_B64={{ RUNPOD_SECRET_github_rlvr_deploy_key_b64 }}`
  - `RLVR_REPO=git@github.com:UnitedSnakes/rlvr_behavior_probe_runpod.git`
  - `RLVR_BRANCH=difficulty-bin-analysis`
  - `RLVR_REPO_DIR=/workspace/rlvr_behavior_probe_runpod`
- Produces bootstrap log path `/workspace/rlvr-bootstrap.log` and recovery command `rlvr-bootstrap`.

- [ ] **Step 1: Add failing README contract assertions**

Append to `tests/test_runpod_image_config.py`:

```python
def test_readme_documents_bootstrap_template_and_correct_hf_repo():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    required = [
        "GITHUB_DEPLOY_KEY_B64={{ RUNPOD_SECRET_github_rlvr_deploy_key_b64 }}",
        "HF_TOKEN={{ RUNPOD_SECRET_huggingface_token }}",
        "RLVR_REPO=git@github.com:UnitedSnakes/rlvr_behavior_probe_runpod.git",
        "RLVR_BRANCH=difficulty-bin-analysis",
        "RLVR_REPO_DIR=/workspace/rlvr_behavior_probe_runpod",
        "/workspace/rlvr-bootstrap.log",
        "rlvr-bootstrap",
        "HKReporter/rlvr-behavior-probe-results",
    ]
    for text in required:
        assert text in readme

    assert "UnitedSnakes/rlvr-behavior-probe-results" not in readme
```

- [ ] **Step 2: Run the README contract test and verify it fails**

Run:

```bash
python -m pytest -q \
  tests/test_runpod_image_config.py::test_readme_documents_bootstrap_template_and_correct_hf_repo
```

Expected: FAIL because the current README still has the old manual startup flow and wrong HF namespace.

- [ ] **Step 3: Replace the current RunPod setup subsection with the bootstrap-aware flow**

Update `README.md` so the RunPod section contains the existing image/port settings plus this environment block:

```text
HF_TOKEN={{ RUNPOD_SECRET_huggingface_token }}
GITHUB_DEPLOY_KEY_B64={{ RUNPOD_SECRET_github_rlvr_deploy_key_b64 }}
RLVR_REPO=git@github.com:UnitedSnakes/rlvr_behavior_probe_runpod.git
RLVR_BRANCH=difficulty-bin-analysis
RLVR_REPO_DIR=/workspace/rlvr_behavior_probe_runpod
```

Document this exact Container start command:

```bash
/bin/bash -lc '/start.sh & start_pid=$!; rlvr-bootstrap > /workspace/rlvr-bootstrap.log 2>&1; bootstrap_status=$?; if [ "$bootstrap_status" -ne 0 ]; then printf "[rlvr-bootstrap] startup bootstrap failed with exit %s; pod remains available; rerun rlvr-bootstrap manually\n" "$bootstrap_status" >> /workspace/rlvr-bootstrap.log; fi; wait "$start_pid"'
```

Explain the behavior in plain language:

```text
/start.sh starts first so RunPod SSH/Jupyter remain available. The bootstrap then
configures the deploy key and prepares the repository. A bootstrap error is written
to /workspace/rlvr-bootstrap.log but is not allowed to kill the pod. After correcting
the cause, rerun `rlvr-bootstrap` manually.
```

Document the SHA-first rollout rule:

```text
Normal Docker-related pushes publish only a sha-* image tag. Test that image on a
fresh pod first. After the fresh-pod bootstrap and one-question vLLM smoke test pass,
manually dispatch the image workflow with publish_stable=true to promote the tested
build to 0.27.1.
```

Finally, change the upload example to:

```bash
python run_probe.py \
  --engine vllm \
  --only-rl \
  --rollouts 256 \
  --result-dir results_rl256_vllm \
  --upload-repo HKReporter/rlvr-behavior-probe-results
```

Keep the existing warning that secrets never belong in the Dockerfile or repository.

- [ ] **Step 4: Run the README/static contract tests**

Run:

```bash
python -m pytest -q tests/test_runpod_image_config.py
```

Expected: PASS.

- [ ] **Step 5: Run all bootstrap-related and existing focused tests together**

Run:

```bash
python -m pytest -q \
  tests/test_runpod_bootstrap.py \
  tests/test_runpod_image_config.py \
  tests/test_results_upload.py \
  tests/test_run_probe_modes.py \
  tests/test_vllm_sampler.py
```

Expected: PASS.

- [ ] **Step 6: Commit Task 3**

```bash
git add README.md tests/test_runpod_image_config.py
git commit -m "Document automatic RunPod bootstrap"
```

---

### Task 4: Verify the SHA image on a genuinely fresh RunPod before stable promotion

**Files:**
- No source changes expected.
- Evidence to inspect: GitHub Actions image build, RunPod `/workspace/rlvr-bootstrap.log`, fresh-pod checkout, focused pytest output, HF Dataset upload.

**Interfaces:**
- Consumes: the `sha-<short-commit>` image emitted by Task 2's workflow.
- Produces: evidence that automatic startup works on a real GPU pod without manual SSH-key or clone setup.
- Only after all checks pass: publishes `ghcr.io/unitedsnakes/rlvr-vllm:0.27.1` via manual stable promotion.

- [ ] **Step 1: Push the implementation branch and verify the image workflow publishes the SHA tag successfully**

Inspect the `Build RunPod vLLM image` workflow for the implementation commit. The build/push job must succeed. Record the emitted image tag:

```text
ghcr.io/unitedsnakes/rlvr-vllm:sha-<short-commit>
```

Do not manually dispatch `publish_stable=true` yet.

- [ ] **Step 2: Point a temporary RunPod template at the SHA image and configure startup**

Use:

```text
Container image: ghcr.io/unitedsnakes/rlvr-vllm:sha-<short-commit>
Container disk: 30 GB
Network volume: none
HTTP port: 8888
TCP port: 22
```

Set the exact environment variables documented in Task 3 and the exact Container start command from Task 3.

- [ ] **Step 3: Deploy a completely fresh pod and verify automatic bootstrap before running any manual setup command**

After SSH becomes available, run only inspection commands first:

```bash
cat /workspace/rlvr-bootstrap.log

test -d /workspace/rlvr_behavior_probe_runpod/.git \
  && echo "repo present"

git -C /workspace/rlvr_behavior_probe_runpod status --short --branch
git -C /workspace/rlvr_behavior_probe_runpod log -1 --oneline

which python
python - <<'PY'
import os
import sys
import torch
import vllm

print("python:", sys.executable)
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("vllm:", vllm.__version__)
print("gpu:", torch.cuda.get_device_name(0))
print("spawn:", os.environ.get("VLLM_WORKER_MULTIPROC_METHOD"))
print("HF_TOKEN set:", bool(os.environ.get("HF_TOKEN")))
PY
```

Expected:

- bootstrap log ends with `bootstrap complete`;
- repo is already present without manual `ssh-keyscan`, `base64 -d`, or `git clone`;
- branch is `difficulty-bin-analysis`;
- Python resolves to `/opt/vllm-env/bin/python`;
- vLLM is `0.27.1`;
- CUDA sees the selected GPU;
- `HF_TOKEN set: True`.

- [ ] **Step 4: Verify idempotency on the real pod**

Run:

```bash
rlvr-bootstrap
```

Expected: exit code `0`; it fetches/fast-forwards safely and does not reclone or delete the repository.

Then confirm:

```bash
git -C /workspace/rlvr_behavior_probe_runpod status --short --branch
```

Expected: requested branch remains checked out and no bootstrap-created working-tree changes appear.

- [ ] **Step 5: Run the focused tests on the fresh pod**

```bash
cd /workspace/rlvr_behavior_probe_runpod

python -m pytest -q \
  tests/test_runpod_bootstrap.py \
  tests/test_runpod_image_config.py \
  tests/test_results_upload.py \
  tests/test_run_probe_modes.py \
  tests/test_vllm_sampler.py
```

Expected: PASS.

- [ ] **Step 6: Run a one-question vLLM + HF upload end-to-end smoke test**

Use one line from the fixed 30-question file rather than asking `prepare_questions()` to resample it:

```bash
cd /workspace/rlvr_behavior_probe_runpod
head -n 1 data/gsm8k_subset.jsonl > /tmp/gsm8k_smoke.jsonl
rm -rf results_freshpod_bootstrap_smoke

python run_probe.py \
  --engine vllm \
  --only-rl \
  --questions 1 \
  --question-file /tmp/gsm8k_smoke.jsonl \
  --rollouts 1 \
  --max-new-tokens 384 \
  --result-dir results_freshpod_bootstrap_smoke \
  --upload-repo HKReporter/rlvr-behavior-probe-results

cat results_freshpod_bootstrap_smoke/run_config.json
wc -l results_freshpod_bootstrap_smoke/rl_raw.jsonl
```

Expected:

- generation exits successfully;
- HF upload commits the two result files;
- `run_config.json` contains `upload_repo`, `run_started_at`, and `upload_path`;
- `wc -l` reports exactly `1` rollout;
- the timestamped path exists in the private Dataset.

- [ ] **Step 7: Exercise the non-fatal startup failure path once**

Create a temporary RunPod template copy whose `RLVR_BRANCH` is deliberately invalid, for example:

```text
RLVR_BRANCH=bootstrap-intentional-missing-branch
```

Deploy a fresh pod with the same SHA image and start command.

Expected:

- SSH/Jupyter still become reachable;
- `/workspace/rlvr-bootstrap.log` contains a Git/bootstrap failure and the recovery message;
- the pod remains alive;
- manually correcting `RLVR_BRANCH` in the shell and running `rlvr-bootstrap` succeeds.

Delete this failure-test pod after verification.

- [ ] **Step 8: Promote the tested image to the stable tag**

Only after Steps 3–7 pass, manually dispatch `Build RunPod vLLM image` on `difficulty-bin-analysis` with:

```text
publish_stable=true
```

Verify that the workflow succeeds and publishes:

```text
ghcr.io/unitedsnakes/rlvr-vllm:0.27.1
```

Update the normal RunPod template back from the temporary SHA tag to `0.27.1` while keeping the new environment variables and Container start command.

- [ ] **Step 9: Final verification of the promoted stable template**

Deploy one final fresh pod from the normal template and inspect:

```bash
cat /workspace/rlvr-bootstrap.log
git -C /workspace/rlvr_behavior_probe_runpod log -1 --oneline
```

Expected: automatic bootstrap succeeds from the stable tag with no manual GitHub setup.

No additional source commit is required for Task 4 unless verification exposes a defect. If a defect is found, stop, return to the relevant earlier task, add a failing regression test, fix it, and repeat the SHA-image verification before stable promotion.
