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
        ninja-build \
        openssh-client \
        tmux && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir uv==0.11.13

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

COPY --chmod=0755 docker/rlvr-bootstrap.sh /usr/local/bin/rlvr-bootstrap

RUN test -x /usr/local/bin/rlvr-bootstrap && \
    command -v gpustat >/dev/null

RUN /opt/vllm-env/bin/python - <<'PY'
import importlib.metadata
import sys

import accelerate
import datasets
import torch
import transformers
import trl
import vllm

assert sys.version_info[:2] == (3, 12), sys.version
assert vllm.__version__ == "0.27.1", vllm.__version__
assert transformers.__version__ == "5.15.0", transformers.__version__
assert datasets.__version__ == "5.0.1", datasets.__version__
assert accelerate.__version__ == "1.14.0", accelerate.__version__
assert trl.__version__ == "0.28.0", trl.__version__

print("python:", sys.version)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("accelerate:", accelerate.__version__)
print("trl:", trl.__version__)
print("vllm:", vllm.__version__)
print("gpustat:", importlib.metadata.version("gpustat"))
PY

WORKDIR /workspace
