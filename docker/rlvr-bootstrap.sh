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
    printf 'gpustat: %s\n' "$(command -v gpustat || printf 'unavailable')"

    python - <<'PY'
import importlib.metadata
import os
import sys

import accelerate
import datasets
import torch
import transformers
import trl
import vllm

print("python:", sys.executable)
print("python version:", sys.version.split()[0])
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("accelerate:", accelerate.__version__)
print("trl:", trl.__version__)
print("vllm:", vllm.__version__)
try:
    print("gpustat package:", importlib.metadata.version("gpustat"))
except importlib.metadata.PackageNotFoundError:
    print("gpustat package: unavailable")
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
