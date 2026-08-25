# RunPod vLLM Image Design

Date: 2026-08-25

## Goal

Create a reusable RunPod environment for this repository so a new GPU pod can run the vLLM sampling workflow without rebuilding Python, PyTorch, CUDA-side Python packages, FlashInfer, and vLLM from scratch.

The image should reproduce the environment that successfully ran the current smoke test and RL sampling workflow while keeping research code, model weights, caches, and experiment outputs outside the image.

## Scope

This change adds the build and distribution path for the runtime environment. It does not change sampling semantics, model checkpoints, experiment analysis, or the existing HF runtime.

The implementation will add:

- `docker/Dockerfile`
- `docker/requirements-vllm.txt`
- `.github/workflows/build-runpod-image.yml`
- brief usage documentation for building/pulling the image and creating a RunPod template

The existing `requirements.txt` remains the HF/general environment definition and will not be repurposed as the vLLM lock file.

## Architecture

The runtime is split into four layers:

1. **GHCR image**: reusable software environment.
2. **GitHub repository**: research code and configuration.
3. **Hugging Face Hub**: SFT/RL model checkpoints and tokenizer assets.
4. **RunPod local `/workspace`**: ephemeral repository checkout, model cache, vLLM compile cache, and experiment outputs.

No RunPod Network Volume is required for the baseline workflow. Important experiment outputs must be copied out of a disposable pod before it is deleted.

## Base image and Python environment

Use a RunPod-compatible CUDA/PyTorch base image that preserves the normal RunPod pod experience, including shell access and standard runtime tooling.

Inside that image, create a dedicated Python 3.12 environment at:

```text
/opt/vllm-env
```

The image must place `/opt/vllm-env/bin` first on `PATH`, so an interactive shell starts with the intended Python environment without requiring `source activate`.

This separation avoids coupling the vLLM environment to whatever Python or PyTorch version happens to be installed in the base image.

## Version policy

The first image version targets the combination already exercised successfully on RunPod:

- Python 3.12
- vLLM 0.27.1
- PyTorch/CUDA backend selected by the vLLM-compatible installation path

The Docker build should pin vLLM to `0.27.1`. The PyTorch/CUDA dependency set should come from the compatible vLLM installation rather than independently pinning a conflicting torch build.

Additional project packages required for sampling, testing, and analysis should be version-constrained in `docker/requirements-vllm.txt` where reproducibility materially benefits from pinning.

## Runtime environment variables

Set these defaults in the image:

```text
VLLM_WORKER_MULTIPROC_METHOD=spawn
HF_HOME=/workspace/.cache/huggingface
XDG_CACHE_HOME=/workspace/.cache
```

`VLLM_WORKER_MULTIPROC_METHOD=spawn` avoids the CUDA re-initialization failure observed when vLLM forks after the parent process has interacted with CUDA.

Caches remain under `/workspace` so they are easy to inspect or preserve when desired. They are not part of the immutable image.

## Image contents

The image contains software only:

- Python 3.12 runtime
- vLLM 0.27.1 and its compatible PyTorch/CUDA Python dependency stack
- project analysis/test dependencies
- common command-line utilities needed for the RunPod workflow

The image must not contain:

- Hugging Face access tokens
- GitHub credentials
- model weights or checkpoint snapshots
- experiment results
- a baked-in checkout of this repository
- user-specific secrets or SSH private keys

Keeping code and checkpoints outside the image prevents a normal research-code edit or checkpoint change from forcing a large image rebuild.

## GHCR publishing

GitHub Actions builds a Linux `amd64` image and publishes it to GitHub Container Registry.

Primary package name:

```text
ghcr.io/unitedsnakes/rlvr-vllm
```

The workflow publishes immutable or traceable tags, including:

- `0.27.1` for the intended vLLM environment release
- a commit-SHA-derived tag for exact provenance

The workflow should not rely on manually stored GitHub package credentials. It should use the repository-provided `GITHUB_TOKEN` with the minimum package permissions required to push the image.

The GHCR package is intended to be public so RunPod can pull it without registry credentials. Package visibility is a GitHub-side setting and is not embedded as a secret in the workflow.

## Workflow triggers

The image build should not run on every research-code commit.

Trigger it when either:

- Docker/runtime dependency files under `docker/` change, or
- the workflow is manually dispatched.

This avoids rebuilding a large environment image for changes to analyses, prompts, or experimental code that do not alter the environment.

## RunPod template

The RunPod template points to:

```text
ghcr.io/unitedsnakes/rlvr-vllm:0.27.1
```

The baseline template does not require a Network Volume.

`/workspace` is treated as disposable local storage. On a new pod, the normal flow is:

```text
start pod
→ clone/pull repository into /workspace
→ download/cache model checkpoints from Hugging Face as needed
→ run tests/smoke test
→ run experiment
→ copy important result files off the pod before deletion
```

The same image should be usable across compatible NVIDIA GPUs such as RTX 3090, A40, and A6000 without rebuilding a GPU-specific image.

## Failure handling

The build should fail if the Python environment or package installation fails; it must not silently continue with a partially configured runtime.

The image should include a lightweight build-time or CI verification step that at minimum imports the critical Python packages and prints their versions. GPU execution cannot be validated inside a normal GitHub Actions image build, so CUDA/vLLM generation remains a RunPod smoke-test responsibility.

A new image version should not replace the previously known-good tag until the build has completed successfully.

## Testing and verification

Implementation verification has three levels:

1. **Static/build verification**
   - Dockerfile builds successfully.
   - Critical imports succeed inside the built image.
   - Python and vLLM versions match the intended environment.

2. **Repository tests on RunPod**
   - `tests/test_vllm_sampler.py`
   - `tests/test_run_probe_modes.py`
   - relevant analysis tests

3. **GPU smoke test on RunPod**
   - start a pod from the new template
   - confirm `python` resolves to `/opt/vllm-env/bin/python`
   - confirm CUDA sees the selected GPU
   - run a small SFT or RL vLLM sampling job without manually activating a virtual environment or setting the multiprocessing environment variable

The image is considered operational only after the GPU smoke test succeeds.

## Non-goals

This design does not attempt to:

- cache model checkpoints permanently inside the image
- make `/workspace` persistent without a Network Volume
- upload experiment outputs automatically
- replace the existing HF execution path
- create GPU-specific images
- optimize vLLM kernel compilation time beyond preserving normal cache behavior during the lifetime of a pod

## Success criteria

A fresh RunPod pod created from the custom template should require no Python package installation before running this repository's vLLM workflow.

After cloning the repository, the user should be able to run:

```bash
python -c "import torch, vllm; print(torch.__version__, vllm.__version__)"
```

and then run `run_probe.py --engine vllm ...` without activating another virtual environment or manually setting `VLLM_WORKER_MULTIPROC_METHOD`.
