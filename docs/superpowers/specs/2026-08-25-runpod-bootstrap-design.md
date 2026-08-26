# RunPod Bootstrap Design

Date: 2026-08-25

## Goal

Make a fresh RunPod pod created from the existing RLVR vLLM image immediately usable for experiments without manually reconstructing the GitHub SSH setup and cloning the repository each time.

The bootstrap must preserve the normal RunPod pod experience, including SSH and Jupyter access. A bootstrap failure must not terminate the pod or prevent interactive recovery.

## Scope

This change adds an automatic repository bootstrap path around the existing RunPod image and template. It does not change model sampling semantics, experiment configuration, result upload semantics, or Hugging Face authentication.

The implementation will add a small bootstrap script to the image, update the Dockerfile to install it, and document the RunPod template environment variables and startup command required to invoke it.

The repository itself remains outside the image and is cloned into `/workspace` at runtime.

## Current state

The image already provides:

- `/opt/vllm-env/bin/python`
- vLLM 0.27.1 and compatible PyTorch/CUDA dependencies
- `git` and `openssh-client`
- `VLLM_WORKER_MULTIPROC_METHOD=spawn`
- Hugging Face caches under `/workspace/.cache`

The RunPod template already injects `HF_TOKEN` and `GITHUB_DEPLOY_KEY_B64` from RunPod Secrets.

Today, a new pod still requires manual commands to decode the deploy key, populate `known_hosts`, clone the repository, and select `difficulty-bin-analysis`.

## Architecture

Keep the existing separation of responsibilities:

1. **GHCR image** supplies software and the reusable bootstrap executable.
2. **RunPod template** supplies project-specific runtime configuration and invokes bootstrap at startup.
3. **RunPod Secrets** supply `HF_TOKEN` and the base64-encoded GitHub deploy key.
4. **GitHub** supplies the current research code checkout.
5. **Hugging Face** stores model assets and experiment-result backups.
6. **`/workspace`** remains disposable pod-local storage.

The image must not bake in the repository checkout, deploy key, Hugging Face token, or experiment results.

## Bootstrap executable

Install an executable named:

```text
/usr/local/bin/rlvr-bootstrap
```

The script must be safe to run more than once on the same pod.

It reads these environment variables:

```text
GITHUB_DEPLOY_KEY_B64     required for GitHub SSH setup
RLVR_REPO                 required repository SSH URL
RLVR_BRANCH               required branch to check out
RLVR_REPO_DIR             optional destination, default /workspace/rlvr_behavior_probe_runpod
```

`HF_TOKEN` is not required by bootstrap itself. The script may report whether it is set, but must never print its value. Hugging Face upload behavior remains owned by `run_probe.py`.

The recommended RunPod template values are:

```text
RLVR_REPO=git@github.com:UnitedSnakes/rlvr_behavior_probe_runpod.git
RLVR_BRANCH=difficulty-bin-analysis
RLVR_REPO_DIR=/workspace/rlvr_behavior_probe_runpod
```

The results Dataset remains a `run_probe.py --upload-repo` argument. This change does not add a default results repo to experiment code.

## SSH setup

Bootstrap creates `/root/.ssh` with restrictive permissions, decodes `GITHUB_DEPLOY_KEY_B64` to `/root/.ssh/id_ed25519`, and validates the resulting key before attempting Git operations.

It must fail its own bootstrap run if:

- `GITHUB_DEPLOY_KEY_B64` is missing,
- base64 decoding fails,
- the decoded file is not a valid SSH private key, or
- required repository configuration is missing.

The script must not print the encoded secret, decoded key, or token values.

For GitHub host verification, bootstrap should create or update `known_hosts` using `ssh-keyscan github.com` before Git operations. Re-running bootstrap must not cause harmful duplicate state.

## Repository synchronization

Bootstrap handles two repository states.

### Repository absent

If `RLVR_REPO_DIR` does not exist, clone `RLVR_REPO` into that path and check out `RLVR_BRANCH`.

### Repository already present

If `RLVR_REPO_DIR` is an existing Git repository, do not delete or reclone it. Fetch from `origin` and ensure the requested branch exists locally and tracks the corresponding remote branch when needed.

Bootstrap must not run `git reset --hard`, `git clean`, or otherwise destroy local experiment work. If the checkout contains local changes that prevent switching branches or updating safely, bootstrap should report the problem and fail its own run while leaving the checkout untouched.

If `RLVR_REPO_DIR` exists but is not a Git repository, bootstrap should report an error rather than overwrite the directory.

## Startup integration

Do not replace the RunPod base image `ENTRYPOINT` or remove its `/start.sh` service path.

RunPod documents that a template Container start command overrides the image `CMD`, and its custom-template guidance uses `/start.sh` to preserve Jupyter/SSH services. The template will therefore use a startup wrapper that keeps `/start.sh` running and launches `rlvr-bootstrap` as an additional startup task.

Bootstrap failure is deliberately non-fatal to the pod. SSH/Jupyter services must remain available so the user can inspect logs, correct a secret or network problem, and rerun `rlvr-bootstrap` manually.

The exact shell wrapper should preserve `/start.sh` as the long-lived RunPod service process and route bootstrap stdout/stderr to a persistent pod-local log such as:

```text
/workspace/rlvr-bootstrap.log
```

The template startup command must make bootstrap failure visible in that log without propagating the bootstrap exit code as the pod's fatal startup status.

## Logging and status

Bootstrap logs should be concise and safe to share. They should report stages such as:

```text
checking configuration
installing deploy key
configuring GitHub host key
cloning repository
checking out branch
runtime summary
bootstrap complete
```

On success, print a short runtime summary containing non-secret information:

- repository path
- current Git commit
- selected branch
- Python executable
- Python version
- torch version
- CUDA version
- vLLM version
- detected GPU when CUDA is available
- boolean presence of `HF_TOKEN`

On failure, print the failed stage and the underlying command/error while avoiding secret values.

## Failure behavior

There are two separate failure domains.

### Bootstrap process

`rlvr-bootstrap` should return nonzero when its own work fails. This makes manual invocation and testing meaningful.

### Pod startup

The RunPod startup wrapper must treat bootstrap failure as non-fatal. `/start.sh` remains active, so the pod stays reachable over SSH/Jupyter.

Examples of recoverable bootstrap failures include:

- invalid or stale deploy-key Secret,
- GitHub temporarily unreachable,
- wrong repository URL or branch,
- pre-existing non-Git directory at the destination,
- local repository changes blocking checkout.

After correcting the cause, the user can run:

```bash
rlvr-bootstrap
```

without recreating the pod.

## Docker image changes

Store the source bootstrap script in the repository under `docker/` and copy it into `/usr/local/bin/rlvr-bootstrap` during the image build with executable permissions.

Do not add project secrets or the repository itself to the Docker build context beyond the bootstrap source already tracked in Git.

Changes under `docker/` already trigger the existing GHCR image workflow, so the normal image publishing path can build the new bootstrap-enabled image.

The existing image tag should only be updated after build verification and a real fresh-pod startup test. A commit-SHA image tag remains the preferred artifact for first deployment verification.

## README and template documentation

Update the RunPod section of `README.md` to document:

- the required Secrets `HF_TOKEN` and `GITHUB_DEPLOY_KEY_B64`,
- the non-secret bootstrap variables `RLVR_REPO`, `RLVR_BRANCH`, and optional `RLVR_REPO_DIR`,
- the template Container start command,
- the bootstrap log location,
- the manual recovery command `rlvr-bootstrap`,
- the correct Hugging Face results repository example `HKReporter/rlvr-behavior-probe-results`.

The README must continue to warn against committing secrets.

## Testing

Testing has four levels.

### Script-level tests

Add automated tests that exercise bootstrap behavior with temporary directories and stubbed external commands where practical. Cover at minimum:

- missing deploy-key environment variable,
- invalid base64 input,
- repository destination already occupied by a non-Git directory,
- initial clone path,
- existing-repository path,
- refusal to destructively overwrite local changes,
- idempotent repeated execution,
- logs do not echo the secret value.

Tests should avoid requiring a live GitHub connection.

### Docker build verification

The image build must verify that:

```text
/usr/local/bin/rlvr-bootstrap
```

exists and is executable.

Existing critical Python import/version checks remain in place.

### Manual bootstrap smoke test

Inside a fresh pod, verify:

- both required Secrets are present,
- bootstrap produces a valid deploy key without base64 warnings,
- GitHub authentication succeeds,
- the repository appears at `RLVR_REPO_DIR`,
- the requested branch and expected commit are checked out,
- running `rlvr-bootstrap` a second time succeeds without destructive changes.

### Fresh-pod end-to-end test

Deploy a fresh pod from the updated template and confirm:

1. SSH/Jupyter become reachable regardless of bootstrap outcome.
2. `rlvr-bootstrap.log` shows successful repository setup in the normal case.
3. no manual clone or SSH-key setup is needed.
4. the existing focused pytest suite passes.
5. a one-question vLLM `run_probe.py` job can complete and upload results to the private Hugging Face Dataset.

The feature is operational only after this fresh-pod test succeeds.

## Non-goals

This design does not:

- make bootstrap failure terminate the pod,
- automatically run an experiment at pod startup,
- automatically choose experiment arguments,
- change `run_probe.py` upload defaults,
- create Hugging Face repositories,
- persist `/workspace` across deleted pods,
- auto-merge, auto-push, reset, clean, or discard Git changes,
- put GitHub or Hugging Face credentials in the image.

## Success criteria

After deploying a pod from the updated template, the user should be able to SSH into it and find the requested research branch already available under `/workspace/rlvr_behavior_probe_runpod` without manually configuring GitHub SSH access.

A GitHub/bootstrap failure must leave the pod accessible and provide enough information in `/workspace/rlvr-bootstrap.log` to diagnose and rerun bootstrap manually.
