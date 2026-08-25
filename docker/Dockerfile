FROM runpod/pytorch:1.0.3-cu1300-torch290-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    HF_HOME=/workspace/.cache/huggingface \
    XDG_CACHE_HOME=/workspace/.cache \
    PATH=/opt/vllm-env/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
        ca-certificates \
        curl \
        git \
        jq \
        openssh-client \
        tmux && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir uv==0.8.15

RUN uv venv /opt/vllm-env \
        --python 3.12 \
        --seed \
        --managed-python

COPY docker/requirements-vllm.txt /tmp/requirements-vllm.txt

RUN uv pip install \
        --python /opt/vllm-env/bin/python \
        "vllm==0.27.1" \
        --torch-backend=cu130 && \
    uv pip install \
        --python /opt/vllm-env/bin/python \
        -r /tmp/requirements-vllm.txt && \
    rm /tmp/requirements-vllm.txt

RUN /opt/vllm-env/bin/python - <<'PY'
import sys
import torch
import transformers
import vllm

assert sys.version_info[:2] == (3, 12), sys.version
assert vllm.__version__ == "0.27.1", vllm.__version__
print("python:", sys.version)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("transformers:", transformers.__version__)
print("vllm:", vllm.__version__)
PY

WORKDIR /workspace
