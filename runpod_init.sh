#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/workspace
VENV="$WORKSPACE/.venv"
CACHE_ROOT="$WORKSPACE/.cache"
REPO="$WORKSPACE/rlvr_behavior_probe_runpod"

export HF_HOME="$CACHE_ROOT/huggingface"
export HF_DATASETS_CACHE="$CACHE_ROOT/huggingface/datasets"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT"
export TMPDIR="$WORKSPACE/tmp"

echo "======================================"
echo " RunPod research environment setup"
echo "======================================"

# =========================
# 1. System packages
# =========================
apt update

apt install -y --no-install-recommends \
    git \
    curl \
    wget \
    unzip \
    tmux \
    htop \
    nano \
    openssh-server \
    build-essential

apt clean
rm -rf /var/lib/apt/lists/*

# =========================
# 2. Workspace and caches
# =========================
mkdir -p \
    "$HF_HOME" \
    "$HF_DATASETS_CACHE" \
    "$PIP_CACHE_DIR" \
    "$TORCH_HOME" \
    "$TMPDIR"

# Persist environment variables without duplicating
# this block every time the setup script is rerun.
if ! grep -q "# RunPod research environment" ~/.bashrc 2>/dev/null; then
    cat >> ~/.bashrc <<'BASHRC'

# RunPod research environment
export HF_HOME=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
export PIP_CACHE_DIR=/workspace/.cache/pip
export TORCH_HOME=/workspace/.cache/torch
export XDG_CACHE_HOME=/workspace/.cache
export TMPDIR=/workspace/tmp

if [ -f /workspace/.venv/bin/activate ]; then
    source /workspace/.venv/bin/activate
fi
BASHRC
fi

# =========================
# 3. Git identity
# =========================
git config --global user.name "run pod"
git config --global user.email "clouddragonlee888@gmail.com"
git config --global core.editor "nano"

# =========================
# 4. Clone research repo
# =========================
if [ ! -d "$REPO/.git" ]; then
    git clone \
        https://github.com/UnitedSnakes/rlvr_behavior_probe_runpod.git \
        "$REPO"
else
    echo "Repo already exists, skipping clone."
fi

# =========================
# 5. Python environment
# =========================

# Keep the environment on /workspace so the small
# root overlay is not filled by PyTorch and other packages.
if [ ! -d "$VENV" ]; then
    python -m venv "$VENV"
fi

source "$VENV/bin/activate"

python -m pip install \
    --upgrade \
    --no-cache-dir \
    pip

# Pin a CUDA build known to work with the current
# RunPod host driver.
#
# Do NOT replace this with an unpinned:
#   pip install torch
python -m pip install \
    --no-cache-dir \
    "torch==2.11.0+cu128" \
    --index-url https://download.pytorch.org/whl/cu128

cd "$REPO"

python -m pip install \
    --no-cache-dir \
    -r requirements.txt

# Small development/runtime extras actually useful
# for this project.
python -m pip install \
    --no-cache-dir \
    gpustat \
    pytest \
    sentencepiece \
    protobuf \
    scipy \
    scikit-learn \
    tqdm

# =========================
# 6. GPU sanity check
# =========================
echo
echo "======================================"
echo " GPU info"
echo "======================================"

nvidia-smi

echo
echo "======================================"
echo " PyTorch CUDA check"
echo "======================================"

python - <<'PY'
import torch

print("PyTorch version:", torch.__version__)
print("PyTorch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in PyTorch")

print("GPU:", torch.cuda.get_device_name(0))
print("BF16 supported:", torch.cuda.is_bf16_supported())

x = torch.randn(16, 16, device="cuda")
print("CUDA tensor check:", x.device, x.mean().item())
PY

# =========================
# 7. Disk sanity check
# =========================
echo
echo "======================================"
echo " Disk usage"
echo "======================================"

df -h / "$WORKSPACE"

# =========================
# 8. Final instructions
# =========================
echo
echo "======================================"
echo " Setup complete"
echo "======================================"
echo

echo "Project:"
echo "  $REPO"
echo

echo "Python environment:"
echo "  source $VENV/bin/activate"
echo

echo "Hugging Face login:"
echo "  hf auth login"
echo

echo "Start persistent experiment session:"
echo "  tmux new -s rlvr"
echo

echo "Reconnect later:"
echo "  tmux attach -t rlvr"
echo

echo "GPU monitor:"
echo "  gpustat -i 1"
echo