# Cross-Platform vLLM Runtime Design

Date: 2026-08-26

## Goal

Make the probe runnable through the same `run_probe.py --engine vllm` interface on two execution environments:

- RunPod Linux + CUDA, using the existing vLLM 0.27.1 image for canonical experiments.
- Apple Silicon macOS, using a native arm64 Python 3.12 environment with vLLM-Metal for local development, smoke tests, and cheap exploratory rollouts.

This design targets interface compatibility and operational convenience. It does not claim that CUDA vLLM and Metal/vLLM-Metal produce scientifically interchangeable sampling distributions.

## Scope

This change covers four related pieces of infrastructure:

1. Explicitly define effective sampling parameters so HF and vLLM do not silently use different decoding policies.
2. Make the vLLM sampler select a CUDA or Apple Silicon runtime behind the same probe interface.
3. Close the known RunPod image gap for top-k sampling by providing `ninja` and testing the path that triggers FlashInfer JIT compilation.
4. Record enough runtime metadata in experiment outputs to distinguish CUDA and Metal runs and prevent accidental cross-backend comparisons.

Out of scope:

- Treating Metal rollouts as canonical measurements.
- Requiring numerical or distributional equivalence between CUDA vLLM and vLLM-Metal.
- Replacing the RunPod CUDA path with Mac execution for 30-question, 256-rollout production evaluations.
- Auto-converting checkpoints to a different MLX model if the exact Hugging Face checkpoint cannot be loaded on Apple Silicon.

## Current Problems

### Sampling parameters are not fully explicit

`run_probe.py` currently exposes temperature and top-p, but not top-k or repetition penalty. The HF sampler calls `model.generate()` without explicitly overriding top-k or repetition penalty, so those values can be inherited from a checkpoint `generation_config`. The vLLM sampler constructs `SamplingParams` with its own defaults instead.

This caused a large apparent HF-vLLM accuracy discrepancy. The SFT checkpoint generation config contained `top_k=20` and `repetition_penalty=1.1`, while the vLLM path effectively used `top_k=0` and `repetition_penalty=1.0`. A matched vLLM rerun with HF-like sampling reproduced the old HF aggregate accuracy almost exactly.

### The vLLM sampler is CUDA-only by construction

`VLLMSampler` currently rejects any device whose string does not begin with `cuda`. `resolve_device("auto")` already resolves Apple Silicon to `mps`, but the vLLM layer does not support that path.

### The RunPod image misses a runtime build dependency

Top-k sampling through FlashInfer can trigger a JIT compilation path that invokes `ninja`. The current image does not install `ninja-build`. Import-time smoke checks therefore pass even though `top_k > 0` can fail at runtime.

### Result files do not fully identify the execution backend

Current result rows record model and revision but not the exact execution platform, backend package/version, or full effective sampling configuration. That makes it too easy to compare runs that were produced under different decoding operators.

## Design

### 1. One CLI, explicit decoding policy

Add two sampling flags to `run_probe.py`:

```text
--top-k
--repetition-penalty
```

Canonical defaults:

```text
temperature=1.0
top_p=0.95
top_k=0
repetition_penalty=1.0
```

These parameters must be passed explicitly to both HF and vLLM samplers. The probe must not rely on checkpoint `generation_config` values for these experiment-defining settings.

The sampling arguments become part of the run configuration and result metadata.

### 2. Platform-aware vLLM runtime selection

Keep the public experiment interface unchanged:

```bash
python run_probe.py --engine vllm ...
```

`run_probe.py` should continue to choose a sampler through the existing sampler-construction boundary. The vLLM implementation should detect the resolved platform and create the appropriate runtime adapter:

- CUDA device: existing vLLM 0.27.1 path.
- Apple Silicon / `mps`: vLLM-Metal path.
- Other devices: clear unsupported-platform error.

The adapter exposed to the rest of the codebase keeps the same logical interface:

```text
sample(question, n, batch_rollouts, max_new_tokens,
       temperature, top_p, top_k, repetition_penalty, seed)
```

Prompt formatting, tokenizer selection, scoring, result serialization, and Hugging Face result upload remain shared.

No Apple-specific conditionals should leak into scoring or analysis code.

### 3. Apple Silicon environment

Mac execution uses a native arm64 Python 3.12 environment. It is not expected to reproduce the Linux CUDA Docker image package-for-package.

The local setup should be documented as a separate environment built with `uv`. The environment must contain:

- project analysis/runtime dependencies,
- vLLM-Metal and its compatible vLLM dependencies,
- Hugging Face/tokenizer dependencies needed by the probe.

The first compatibility test must use the exact SFT repository and exact checkpoint revision used on CUDA:

```text
ns-0/qwen-2.5-1.5b-instruct-reasoning-sft
checkpoint-8-of-10
```

Do not silently substitute an MLX-community conversion or another model revision. If vLLM-Metal cannot load that exact model, stop and report the incompatibility before deciding on any conversion workflow.

### 4. Runtime metadata

Each run configuration should record at least:

```text
engine: vllm or hf
platform: cuda, metal, or cpu
runtime implementation: vllm-cuda, vllm-metal, or transformers
runtime package version(s)
model name
model revision
tokenizer repository
tokenizer revision
temperature
top_p
top_k
repetition_penalty
max_new_tokens
seed
```

Per-question result rows may either repeat the immutable sampling/runtime metadata or reference the run-level configuration, depending on the existing serialization pattern. The key requirement is that a saved run is self-describing without relying on shell history.

Metal results must be visibly distinguishable from CUDA results.

### 5. Scientific boundary

CUDA vLLM remains the canonical measurement backend for the project.

Metal is approved for:

- local development,
- parser/scorer work,
- prompt checks,
- one-question smoke tests,
- small exploratory rollout counts,
- catching code-path failures before opening a RunPod.

Metal results must not be merged into canonical CUDA probability estimates unless a separate experiment explicitly studies backend sensitivity.

The README should state this boundary plainly.

## RunPod Image Fixes

Add `ninja-build` to the image's OS packages.

The image verification must do more than import vLLM. It should include a focused runtime smoke path that exercises non-default sampling, specifically `top_k > 0`, so the FlashInfer JIT path cannot remain untested.

The existing SHA-first image gating remains unchanged:

1. Push produces a SHA-tagged image.
2. Start a fresh RunPod from that SHA image.
3. Verify bootstrap, runtime, focused tests, and a one-question vLLM generation.
4. Verify a top-k smoke path on the fresh image.
5. Only then manually promote the tested image to the stable tag.

## Acceptance Criteria

### Shared behavior

- `run_probe.py` accepts `--top-k` and `--repetition-penalty`.
- Both HF and vLLM implementations receive explicit values for those parameters.
- Existing canonical behavior remains available through defaults `top_k=0` and `repetition_penalty=1.0`.
- Run metadata records the full effective sampling configuration and runtime identity.

### RunPod / CUDA

- Existing focused tests pass.
- The SHA image builds successfully.
- A fresh Pod starts with the expected Python/vLLM/CUDA environment.
- `rlvr-bootstrap` remains idempotent and non-fatal to Pod startup.
- One-question vLLM smoke generation succeeds.
- A `top_k=20` smoke generation succeeds without missing `ninja` or JIT dependency errors.
- Only after the above checks is the image promoted to the stable tag.

### M5 Pro / Apple Silicon

- Host Python is native arm64, not Rosetta/x86_64.
- A Python 3.12 `uv` environment installs the chosen vLLM-Metal stack.
- `python run_probe.py --engine vllm --device auto ...` resolves to the Metal path without requiring a separate user-facing engine name.
- The exact SFT checkpoint `checkpoint-8-of-10` loads successfully, or the compatibility test fails clearly without substituting another checkpoint.
- A one-question, two-rollout SFT smoke test completes and produces the normal result schema.
- The saved run is marked as Metal/vLLM-Metal so it cannot be mistaken for CUDA output.

## Testing Strategy

Implementation follows TDD.

Add or update unit tests around:

- CLI defaults and explicit sampling arguments,
- propagation of top-k and repetition penalty to HF and vLLM sampler calls,
- platform/runtime selection,
- unsupported-platform errors,
- runtime metadata serialization.

For external-runtime behavior that cannot be meaningfully unit-tested without hardware, use focused acceptance commands on the target machine:

- Apple Silicon M5 Pro for vLLM-Metal installation and one-question generation.
- Fresh RunPod CUDA instance for image/bootstrap/JIT/top-k validation.

Hardware acceptance output is evidence, not a substitute for unit tests on code paths that can be isolated.

## Failure Handling

- If Metal cannot load the exact checkpoint, stop before introducing a model conversion and treat conversion support as a separate design decision.
- If vLLM-Metal exposes materially different sampling argument semantics, preserve the shared CLI but fail loudly for unsupported options rather than silently approximating them.
- If a fresh RunPod SHA image fails, do not promote it to stable.
- Upload failures continue to preserve local result files and return a non-zero status as in the existing upload design.

## Expected Workflow After Completion

Local development:

```text
M5 Pro
  -> pull branch
  -> activate local arm64 Python 3.12 environment
  -> run tests
  -> run 1-16 rollout smoke/exploratory jobs with --engine vllm
  -> inspect results locally
```

Canonical experiment:

```text
push code
  -> SHA RunPod image
  -> fresh-Pod acceptance when image/runtime changed
  -> RunPod CUDA vLLM production job
  -> automatic/private HF result backup
  -> terminate Pod
  -> analyze results locally
```

This keeps the Mac useful for fast iteration without weakening the project's measurement discipline.