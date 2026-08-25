# RunPod vLLM Image Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and publish a reusable RunPod-compatible vLLM 0.27.1 image so fresh GPU pods can run this repository without reinstalling the Python/CUDA inference stack.

**Architecture:** Extend RunPod's CUDA/PyTorch base image so its default `/start.sh` still provides SSH/Jupyter services. Create an isolated Python 3.12 environment at `/opt/vllm-env`, install the vLLM CUDA 13.0 stack there, publish the image to GHCR from GitHub Actions, and keep code/model weights/results outside the image.

**Tech Stack:** Docker, RunPod `runpod/pytorch`, Python 3.12, uv, vLLM 0.27.1, CUDA 13.0, GitHub Actions, GHCR.

**Spec:** `docs/superpowers/specs/2026-08-25-runpod-vllm-image-design.md`

## Global Constraints

- Preserve RunPod base-image Jupyter/SSH startup behavior; do not override the base image entrypoint or `/start.sh`.
- Use Python 3.12 in `/opt/vllm-env` and place `/opt/vllm-env/bin` first on `PATH`.
- Pin vLLM to `0.27.1` and select the vLLM-supported CUDA 13.0 PyTorch backend rather than reusing the base image's PyTorch environment.
- Set `VLLM_WORKER_MULTIPROC_METHOD=spawn`, `HF_HOME=/workspace/.cache/huggingface`, and `XDG_CACHE_HOME=/workspace/.cache` in the image.
- Do not bake model weights, repository source, experiment outputs, GitHub credentials, Hugging Face credentials, SSH private keys, or other secrets into the image.
- Publish `linux/amd64` only.
- Publish to `ghcr.io/unitedsnakes/rlvr-vllm` using the repository-provided `GITHUB_TOKEN` with `contents: read` and `packages: write` permissions.
- Build on Docker/runtime changes and manual dispatch, not on ordinary research-code commits.
- GPU inference verification remains a RunPod smoke test; GitHub Actions only verifies image build and CPU-safe imports/version metadata.

---

### Task 1: Define the isolated vLLM runtime image

**Files:**
- Create: `docker/requirements-vllm.txt`
- Create: `docker/Dockerfile`

**Interfaces:**
- Consumes: RunPod base image `runpod/pytorch:1.0.3-cu1300-torch290-ubuntu2404`.
- Produces: an image whose default `python` is `/opt/vllm-env/bin/python`, with `vllm==0.27.1` and project analysis/test dependencies available.

Because these files are deployment configuration, verification is build-based rather than unit-test red/green. The Docker build itself is the executable contract.

- [ ] **Step 1: Create the vLLM-specific dependency file**

Create `docker/requirements-vllm.txt` with:

```text
transformers==5.15.0
datasets>=2.20
accelerate>=0.33
pandas>=2.2
numpy>=1.26
huggingface_hub>=0.24
matplotlib>=3.8.3,<3.10
pytest>=8.3
```

Do not list `torch` here. vLLM's CUDA wheel must own the compatible PyTorch dependency set.

- [ ] **Step 2: Create the Dockerfile**

Create `docker/Dockerfile` with:

```dockerfile
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
```

Do not add `CMD` or `ENTRYPOINT`; inherit RunPod's default `/start.sh` behavior.

- [ ] **Step 3: Build the image locally or in a disposable Docker builder to validate syntax and dependency resolution**

Run:

```bash
docker build \
  --platform linux/amd64 \
  -f docker/Dockerfile \
  -t rlvr-vllm:test \
  .
```

Expected: Docker exits `0`; the build-time version check prints Python 3.12 and vLLM 0.27.1.

- [ ] **Step 4: Validate the image's default Python and runtime variables**

Run:

```bash
docker run --rm --platform linux/amd64 rlvr-vllm:test \
  bash -lc 'which python && python - <<"PY"
import os
import sys
import torch
import vllm

print(sys.executable)
print(torch.__version__)
print(torch.version.cuda)
print(vllm.__version__)
print(os.environ["VLLM_WORKER_MULTIPROC_METHOD"])
print(os.environ["HF_HOME"])
print(os.environ["XDG_CACHE_HOME"])
PY'
```

Expected values:

```text
/opt/vllm-env/bin/python
...
0.27.1
spawn
/workspace/.cache/huggingface
/workspace/.cache
```

The exact torch patch/build string is not asserted here because it is selected by the vLLM-compatible CUDA 13.0 dependency resolution.

- [ ] **Step 5: Commit the runtime image files**

```bash
git add docker/Dockerfile docker/requirements-vllm.txt
git commit -m "Add reusable RunPod vLLM image"
```

---

### Task 2: Publish the image to GHCR with GitHub Actions

**Files:**
- Create: `.github/workflows/build-runpod-image.yml`

**Interfaces:**
- Consumes: `docker/Dockerfile` and `docker/requirements-vllm.txt` from Task 1.
- Produces: `ghcr.io/unitedsnakes/rlvr-vllm:0.27.1` and a traceable `sha-<shortsha>` tag.

- [ ] **Step 1: Create the GHCR workflow**

Create `.github/workflows/build-runpod-image.yml` with:

```yaml
name: Build RunPod vLLM image

on:
  workflow_dispatch:
  push:
    branches:
      - difficulty-bin-analysis
    paths:
      - docker/**
      - .github/workflows/build-runpod-image.yml

permissions:
  contents: read
  packages: write

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - name: Check out repository
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Generate image metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ghcr.io/unitedsnakes/rlvr-vllm
          tags: |
            type=raw,value=0.27.1
            type=sha,prefix=sha-

      - name: Build and push image
        uses: docker/build-push-action@v6
        with:
          context: .
          file: docker/Dockerfile
          platforms: linux/amd64
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

No repository secret is needed for GHCR publishing; `GITHUB_TOKEN` is supplied by Actions.

- [ ] **Step 2: Validate workflow structure before pushing**

If `actionlint` is available, run:

```bash
actionlint .github/workflows/build-runpod-image.yml
```

Otherwise validate YAML syntax with Python:

```bash
python - <<'PY'
from pathlib import Path
import yaml

path = Path('.github/workflows/build-runpod-image.yml')
with path.open() as handle:
    yaml.safe_load(handle)
print('workflow yaml parses')
PY
```

Expected: exit `0`.

- [ ] **Step 3: Commit the workflow**

```bash
git add .github/workflows/build-runpod-image.yml
git commit -m "Build RunPod image with GitHub Actions"
```

- [ ] **Step 4: Push and inspect the GitHub Actions run**

```bash
git push origin difficulty-bin-analysis
```

Open the `Build RunPod vLLM image` workflow run and verify the `Build and push image` step exits successfully. Do not call the image operational based only on YAML parsing or a pushed commit.

- [ ] **Step 5: Make the GHCR package public after the first successful build**

In GitHub, open the package settings for `rlvr-vllm` and change package visibility to Public. Then verify the image can be pulled without registry credentials:

```bash
docker logout ghcr.io || true
docker pull ghcr.io/unitedsnakes/rlvr-vllm:0.27.1
```

Expected: pull succeeds anonymously.

---

### Task 3: Document and verify the RunPod template path

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: the public image from Task 2.
- Produces: exact template settings and a fresh-pod verification checklist.

- [ ] **Step 1: Add a `RunPod vLLM image` section to the README**

Document these exact template settings:

```text
Container image: ghcr.io/unitedsnakes/rlvr-vllm:0.27.1
Container disk: 30 GB
Network volume: none
HTTP port: 8888
TCP port: 22
```

Document that secrets such as `HF_TOKEN` and the GitHub deploy key are runtime/template concerns and must never be added to the Dockerfile or committed to the repository.

Document the new-pod sanity command:

```bash
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
PY
```

- [ ] **Step 2: Run the existing repository tests on the fresh RunPod image**

After cloning `difficulty-bin-analysis` on a pod created from the custom template, run:

```bash
python -m pytest -q \
  tests/test_vllm_sampler.py \
  tests/test_run_probe_modes.py \
  tests/test_reachability_depth.py
```

Expected: all selected tests pass.

- [ ] **Step 3: Run a GPU smoke test without activating a venv or manually exporting spawn**

Run:

```bash
python run_probe.py \
  --engine vllm \
  --questions 2 \
  --rollouts 2 \
  --max-new-tokens 2048 \
  --temperature 1.0 \
  --top-p 0.95 \
  --device cuda \
  --dtype bfloat16 \
  --question-file data/gsm8k_subset.jsonl \
  --rl-revision main \
  --result-dir results_template_smoke \
  --only-rl
```

Expected: vLLM initializes, generates both questions, and `run_probe.py` exits normally without `ModuleNotFoundError`, CUDA-fork errors, or manual environment activation.

- [ ] **Step 4: Commit documentation after successful image/GPU verification**

```bash
git add README.md
git commit -m "Document RunPod vLLM template"
git push origin difficulty-bin-analysis
```

---

## Follow-up subprojects

These are intentionally separate because each can fail or evolve independently of the image build:

1. **RunPod bootstrap/init script:** consume the existing RunPod Secrets for the GitHub deploy key and Hugging Face token, materialize the deploy key with correct permissions, populate `known_hosts`, and clone/pull the repository automatically.
2. **Hugging Face result backup:** add an explicit upload command/flag that uploads completed `result_dir` contents to a dedicated HF Dataset repository so raw rollouts survive Pod deletion.

Each follow-up gets its own small design/plan and tests after the base image is buildable.
