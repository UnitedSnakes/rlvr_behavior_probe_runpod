#!/usr/bin/env bash
set -e

echo "======================================"
echo " RunPod research environment setup"
echo "======================================"

# =========================
# 1. System packages
# =========================
apt update

apt install -y \
    git \
    curl \
    wget \
    unzip \
    tmux \
    htop \
    openssh-server \
    build-essential

# =========================
# 2. Workspace
# =========================
mkdir -p /workspace
mkdir -p /workspace/.cache/huggingface

cd /workspace

# =========================
# 3. Hugging Face cache
# =========================
cat >> ~/.bashrc <<'EOF'

# RunPod research environment
export HF_HOME=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets
EOF

export HF_HOME=/workspace/.cache/huggingface
export HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets

# =========================
# 4. Git identity
# =========================
git config --global user.name "run pod"
git config --global user.email "clouddragonlee888@gmail.com"

# =========================
# 5. Python environment
# =========================
python -m pip install --upgrade pip

pip install \
    torch \
    transformers \
    accelerate \
    peft \
    bitsandbytes \
    datasets \
    huggingface_hub \
    sentencepiece \
    protobuf \
    scikit-learn \
    scipy \
    pandas \
    numpy \
    tqdm \
    matplotlib \
    gpustat \
    unsloth \
    tf-keras \
    pytest

# =========================
# 6. Clone research repo
# =========================
cd /workspace

if [ ! -d "/workspace/rlvr_behavior_probe_runpod" ]; then
    git clone https://github.com/UnitedSnakes/rlvr_behavior_probe_runpod.git
else
    echo "Repo already exists, skipping clone."
fi

cd /workspace/rlvr_behavior_probe_runpod

# =========================
# 7. GPU sanity check
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
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)
    print("BF16 supported:", torch.cuda.is_bf16_supported())
PY

# =========================
# 8. Final instructions
# =========================
echo
echo "======================================"
echo " Setup complete"
echo "======================================"
echo
echo "Project:"
echo "  /workspace/rlvr_behavior_probe_runpod"
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