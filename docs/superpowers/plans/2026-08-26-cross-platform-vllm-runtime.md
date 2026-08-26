# Cross-Platform vLLM Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep one `run_probe.py --engine vllm` interface across RunPod CUDA and Apple Silicon, make the effective sampling policy explicit, record runtime identity in saved runs, and close the known RunPod top-k JIT dependency gap.

**Architecture:** Preserve `run_probe.py` as the orchestration layer and keep a single logical `VLLMSampler` interface. CUDA continues to use the existing vLLM 0.27.1 runtime; Apple Silicon uses the vLLM-Metal plugin in its own native arm64 Python 3.12 environment. The experiment-facing sampler API is shared, while runtime identity is recorded so Metal output cannot be mistaken for canonical CUDA output. Sampling parameters are explicit on every backend rather than inherited from model generation config.

**Tech Stack:** Python 3.12, argparse, Transformers, vLLM 0.27.1 on CUDA, vLLM-Metal/MLX on Apple Silicon, uv, pytest, Docker, GitHub Actions, RunPod, Hugging Face Hub.

**Spec:** `docs/superpowers/specs/2026-08-26-cross-platform-vllm-runtime-design.md`

## Global Constraints

- Keep the public experiment command `python run_probe.py --engine vllm ...` on both CUDA and Apple Silicon.
- Canonical decoding defaults are `temperature=1.0`, `top_p=0.95`, `top_k=0`, and `repetition_penalty=1.0`.
- Pass `top_k` and `repetition_penalty` explicitly to both Transformers and vLLM. Do not rely on checkpoint `generation_config` for experiment-defining sampling settings.
- Keep `Qwen/Qwen2.5-1.5B-Instruct` as the canonical tokenizer/chat-template source and use tokenizer revision `main` explicitly.
- CUDA vLLM remains the canonical measurement backend. Metal is for development, smoke tests, and small exploratory jobs unless a separate backend-sensitivity experiment is declared.
- The first Apple Silicon acceptance test must use `ns-0/qwen-2.5-1.5b-instruct-reasoning-sft` at exact revision `checkpoint-8-of-10`. Do not silently replace it with an MLX-community conversion or another checkpoint.
- Do not install the CUDA-specific `docker/requirements-vllm.txt` into the vLLM-Metal environment. The Metal installer owns its compatible vLLM/Transformers/MLX runtime dependency set.
- Do not change the existing Hugging Face result-upload semantics, RunPod bootstrap safety behavior, or SHA-first stable-image promotion gate.
- Implementation is test-first. Hardware acceptance is additional evidence, not a substitute for unit tests.

---

## Task 1: Make the sampling protocol explicit on every backend

**Files:**
- Modify: `run_probe.py`
- Modify: `probe/model.py`
- Modify: `probe/vllm_model.py`
- Modify: `probe/prompts.py`
- Modify: `tests/test_run_probe_modes.py`
- Modify: `tests/test_vllm_sampler.py`
- Create: `tests/test_hf_sampler.py`

**Target interface:**

```python
sample(
    question: str,
    n: int,
    batch_rollouts: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    seed: int,
)
```

- [ ] **Step 1: Write failing CLI and orchestration tests**

Extend `tests/test_run_probe_modes.py` so `parse_args()` is expected to expose:

```python
assert args.top_k == 0
assert args.repetition_penalty == 1.0
```

Add an explicit-override case for `--top-k 20 --repetition-penalty 1.1`.

Update the fake-sampler checkpoint test so it records `sample(**kwargs)` and asserts that `run_one_checkpoint()` forwards both values.

- [ ] **Step 2: Write a failing vLLM propagation test**

Update `tests/test_vllm_sampler.py` to call:

```python
sampler.sample(
    ...,
    top_k=20,
    repetition_penalty=1.1,
)
```

and require the fake `SamplingParams` kwargs to contain:

```python
"top_k": 20,
"repetition_penalty": 1.1,
```

Also update the fake tokenizer factory to accept `revision=` and assert the canonical tokenizer revision is `main`.

- [ ] **Step 3: Write a failing HF sampler test**

Create `tests/test_hf_sampler.py` using fake `AutoTokenizer` and `AutoModelForCausalLM` objects. Use real small CPU torch tensors for the fake tokenizer output so no model or network access occurs. Assert that `Sampler.sample()` calls `model.generate()` with explicit:

```python
"temperature": 1.0,
"top_p": 0.95,
"top_k": 0,
"repetition_penalty": 1.0,
```

Add a second case with `top_k=20` and `repetition_penalty=1.1` so the regression that caused the HF-vLLM mismatch is directly covered.

Assert the tokenizer is loaded from the canonical tokenizer repository at revision `main`.

- [ ] **Step 4: Run the focused tests and verify RED**

```bash
python -m pytest -q \
  tests/test_run_probe_modes.py \
  tests/test_vllm_sampler.py \
  tests/test_hf_sampler.py
```

Expected: new assertions fail because the CLI and sampler signatures do not yet expose/forward the new parameters.

- [ ] **Step 5: Implement the minimal sampling changes**

In `probe/prompts.py` add:

```python
TOKENIZER_REVISION = "main"
```

In `run_probe.py` add:

```python
parser.add_argument("--top-k", type=int, default=0)
parser.add_argument("--repetition-penalty", type=float, default=1.0)
```

Forward both fields in `run_one_checkpoint()` and include them in the printed sampling summary.

In `probe/model.py`, extend `Sampler.sample()` and pass both arguments explicitly into `self.model.generate(...)`. Load the tokenizer with `revision=TOKENIZER_REVISION`.

In `probe/vllm_model.py`, extend `VLLMSampler.sample()` and pass both arguments explicitly into `SamplingParams(...)`. Use `TOKENIZER_REVISION` for both the local tokenizer load and vLLM `tokenizer_revision`.

Do not introduce a checkpoint-generation-config fallback.

- [ ] **Step 6: Run focused tests and then the existing probe tests**

```bash
python -m pytest -q \
  tests/test_hf_sampler.py \
  tests/test_vllm_sampler.py \
  tests/test_run_probe_modes.py \
  tests/test_results_upload.py \
  tests/test_reachability_depth.py
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit Task 1**

```bash
git add \
  run_probe.py \
  probe/model.py \
  probe/vllm_model.py \
  probe/prompts.py \
  tests/test_run_probe_modes.py \
  tests/test_vllm_sampler.py \
  tests/test_hf_sampler.py
git commit -m "Explicitly define sampling protocol"
```

---

## Task 2: Add platform-aware vLLM support and self-describing runtime metadata

**Files:**
- Create: `probe/runtime.py`
- Create: `tests/test_runtime_metadata.py`
- Modify: `probe/vllm_model.py`
- Modify: `run_probe.py`
- Modify: `tests/test_vllm_sampler.py`
- Modify: `tests/test_run_probe_modes.py`

**Runtime metadata shape:**

```json
{
  "platform": "cuda",
  "implementation": "vllm-cuda",
  "machine": "x86_64",
  "python_version": "3.12.x",
  "packages": {
    "torch": "...",
    "transformers": "...",
    "vllm": "...",
    "vllm-metal": null
  },
  "tokenizer": {
    "name": "Qwen/Qwen2.5-1.5B-Instruct",
    "revision": "main"
  }
}
```

`vars(args)` already records the sampling parameters, model names, revisions, seed, and generation budget; do not duplicate those fields inside the runtime object.

- [ ] **Step 1: Write failing pure-function runtime tests**

Create `tests/test_runtime_metadata.py` around small helpers in the future `probe/runtime.py`:

```python
assert platform_from_device("cuda") == "cuda"
assert platform_from_device("cuda:0") == "cuda"
assert platform_from_device("mps") == "metal"
assert platform_from_device("cpu") == "cpu"

assert runtime_implementation("hf", "cpu") == "transformers"
assert runtime_implementation("vllm", "cuda") == "vllm-cuda"
assert runtime_implementation("vllm", "mps") == "vllm-metal"
```

Require `runtime_implementation("vllm", "cpu")` to raise a clear unsupported-platform `ValueError`.

Monkeypatch `importlib.metadata.version`, `platform.machine`, and `platform.python_version` to test metadata collection deterministically without importing or starting vLLM.

- [ ] **Step 2: Write failing sampler platform-selection tests**

In `tests/test_vllm_sampler.py`:

- preserve the existing CUDA constructor test,
- add `device="mps"` and require construction to succeed with the same fake vLLM API,
- add `device="cpu"` and require a clear error saying CUDA or Apple Silicon Metal is required.

Do not create a second user-facing engine name.

- [ ] **Step 3: Write a failing run-config metadata test**

In `tests/test_run_probe_modes.py`, monkeypatch a future `collect_runtime_metadata()` helper and require `run_config.json` to include its returned runtime object after a no-generation/fake-generation run.

Assert existing upload metadata and mode behavior remain unchanged.

- [ ] **Step 4: Run the new tests and verify RED**

```bash
python -m pytest -q \
  tests/test_runtime_metadata.py \
  tests/test_vllm_sampler.py \
  tests/test_run_probe_modes.py
```

- [ ] **Step 5: Implement runtime helpers and relax the vLLM device guard**

Create `probe/runtime.py` with focused helpers:

```python
def platform_from_device(device: str) -> str: ...
def runtime_implementation(engine: str, device: str) -> str: ...
def package_version(name: str) -> str | None: ...
def collect_runtime_metadata(engine: str, device: str) -> dict: ...
```

Use `importlib.metadata.version()` for package strings so metadata collection does not import heavy runtime modules or initialize GPU state.

Update `VLLMSampler.__init__()` to accept CUDA devices and `mps`; keep a clear error for other devices. Do not fork the scoring or prompt path. The same `from vllm import LLM, SamplingParams` API remains the adapter boundary; on macOS the installed vLLM-Metal plugin supplies the Metal platform implementation.

Keep `VLLM_WORKER_MULTIPROC_METHOD=spawn` behavior unchanged for now. If real Metal acceptance shows that this variable is incompatible, debug that evidence separately instead of guessing in advance.

- [ ] **Step 6: Record runtime metadata in `run_config.json`**

After resolving the device, collect runtime metadata and write it under:

```python
config["runtime"] = runtime_metadata
```

Keep `device_resolved` for backwards compatibility. Do not repeat the runtime object in every rollout row.

- [ ] **Step 7: Run focused and regression tests**

```bash
python -m pytest -q \
  tests/test_runtime_metadata.py \
  tests/test_hf_sampler.py \
  tests/test_vllm_sampler.py \
  tests/test_run_probe_modes.py \
  tests/test_results_upload.py \
  tests/test_reachability_depth.py
```

- [ ] **Step 8: Commit Task 2**

```bash
git add \
  probe/runtime.py \
  probe/vllm_model.py \
  run_probe.py \
  tests/test_runtime_metadata.py \
  tests/test_vllm_sampler.py \
  tests/test_run_probe_modes.py
git commit -m "Add platform-aware vLLM runtime metadata"
```

---

## Task 3: Add the Apple Silicon environment contract and M5 acceptance workflow

**Files:**
- Create: `requirements-macos-vllm.txt`
- Create: `tests/test_macos_runtime_docs.py`
- Modify: `README.md`

**Dependency rule:** the vLLM-Metal installer owns `vllm`, `vllm-metal`, Transformers, MLX, torch/vLLM core dependencies, and their compatible versions. Do not install `docker/requirements-vllm.txt` or the generic `requirements.txt` wholesale into that environment because they can override runtime-owned dependencies.

Use a small project-extras file containing only dependencies not owned by the Metal runtime, initially:

```text
datasets>=2.20
pandas>=2.2
numpy>=1.26
huggingface_hub>=0.24
matplotlib>=3.8.3,<3.10
pytest>=8.3
```

- [ ] **Step 1: Write failing static documentation/dependency tests**

Create `tests/test_macos_runtime_docs.py` to require:

- `requirements-macos-vllm.txt` exists,
- it does not contain `torch`, `transformers`, `vllm`, or `vllm-metal` pins,
- README documents native `arm64` and Python 3.12 preflight,
- README documents the official vLLM-Metal installer and `~/.venv-vllm-metal`,
- README states Metal is development/smoke/exploratory only and CUDA remains canonical,
- README uses exact SFT revision `checkpoint-8-of-10` in the first Metal smoke test.

- [ ] **Step 2: Run the static test and verify RED**

```bash
python -m pytest -q tests/test_macos_runtime_docs.py
```

- [ ] **Step 3: Add the Mac extras file and README instructions**

Document preflight:

```bash
sw_vers -productVersion
uname -m
python3 -c 'import platform; print(platform.machine())'
file "$(which python3)"
```

Expected: macOS 15+ and `arm64`, not Rosetta/x86_64.

Document the official installer:

```bash
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
```

Then add project extras without overriding plugin-owned runtime packages:

```bash
uv pip install \
  --python ~/.venv-vllm-metal/bin/python \
  -r requirements-macos-vllm.txt
```

Document a runtime diagnostic using `importlib.metadata.version()` for `vllm` and `vllm-metal`.

- [ ] **Step 4: Document the exact one-question Metal smoke test**

The existing `data/gsm8k_subset.jsonl` contains 30 rows, while `prepare_questions()` requires its row count to equal `--questions`. Therefore derive a one-row file from the canonical subset rather than passing `--questions 1` against the 30-row file:

```bash
head -n 1 data/gsm8k_subset.jsonl > /tmp/gsm8k_subset_q0.jsonl
```

Then run:

```bash
~/.venv-vllm-metal/bin/python run_probe.py \
  --engine vllm \
  --device auto \
  --only-sft \
  --questions 1 \
  --rollouts 2 \
  --question-file /tmp/gsm8k_subset_q0.jsonl \
  --max-new-tokens 256 \
  --temperature 1.0 \
  --top-p 0.95 \
  --top-k 0 \
  --repetition-penalty 1.0 \
  --dtype bfloat16 \
  --sft-revision checkpoint-8-of-10 \
  --result-dir results_m5_smoke
```

Acceptance requires a normal result file plus `run_config.json` with:

```text
device_resolved = mps
runtime.platform = metal
runtime.implementation = vllm-metal
```

If the exact checkpoint does not load, stop. Do not switch the smoke test to an MLX-community model as a workaround.

- [ ] **Step 5: Run the static docs test and commit Task 3**

```bash
python -m pytest -q tests/test_macos_runtime_docs.py

git add requirements-macos-vllm.txt README.md tests/test_macos_runtime_docs.py
git commit -m "Document Apple Silicon vLLM workflow"
```

---

## Task 4: Close the RunPod top-k JIT dependency gap

**Files:**
- Modify: `docker/Dockerfile`
- Modify: `tests/test_runpod_image_config.py`
- Modify: `README.md`

- [ ] **Step 1: Add a failing static image test**

Extend `tests/test_runpod_image_config.py`:

```python
def test_dockerfile_installs_ninja_for_flashinfer_topk_jit():
    dockerfile = (REPO_ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
    assert "ninja-build" in dockerfile
```

Also require the README's fresh-Pod acceptance section to contain a `--top-k 20` smoke command.

- [ ] **Step 2: Run the image-config test and verify RED**

```bash
python -m pytest -q tests/test_runpod_image_config.py
```

Expected: the new ninja assertion fails.

- [ ] **Step 3: Add `ninja-build` to the Docker image**

Add it to the existing `apt-get install --no-install-recommends` package list. Do not change the base image, vLLM pin, `/opt/vllm-env`, startup entrypoint behavior, or bootstrap placement.

- [ ] **Step 4: Add the real top-k fresh-Pod acceptance command to README**

As on Mac, first derive a one-row canonical question file:

```bash
head -n 1 data/gsm8k_subset.jsonl > /tmp/gsm8k_subset_q0.jsonl
```

Use a real generation command that forces the previously failing path:

```bash
python run_probe.py \
  --engine vllm \
  --device cuda \
  --only-sft \
  --questions 1 \
  --rollouts 2 \
  --question-file /tmp/gsm8k_subset_q0.jsonl \
  --max-new-tokens 256 \
  --temperature 1.0 \
  --top-p 0.95 \
  --top-k 20 \
  --repetition-penalty 1.1 \
  --dtype bfloat16 \
  --sft-revision checkpoint-8-of-10 \
  --result-dir results_topk_smoke
```

This is an infrastructure acceptance test, not the canonical science protocol.

- [ ] **Step 5: Run static tests and commit Task 4**

```bash
python -m pytest -q \
  tests/test_runpod_image_config.py \
  tests/test_runpod_bootstrap.py

git add docker/Dockerfile README.md tests/test_runpod_image_config.py
git commit -m "Add RunPod top-k JIT dependency"
```

The Docker path change will trigger a new SHA-tagged image build when pushed.

---

## Task 5: Run full software regression and both hardware acceptance gates

**Files:** no new implementation files unless a failure reveals a specific bug. Any failure must be debugged before changing code.

- [ ] **Step 1: Run the full repository test suite before pushing**

```bash
python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Push `difficulty-bin-analysis` and verify the SHA image build**

```bash
git push origin difficulty-bin-analysis
```

Inspect the `Build RunPod vLLM image` workflow. Require the build/push job to finish successfully. Do not publish stable yet.

- [ ] **Step 3: M5 Pro acceptance**

On the Apple Silicon machine, pull the implementation commit, perform the Task 3 preflight and install/update vLLM-Metal with the official installer. Then install `requirements-macos-vllm.txt` into `~/.venv-vllm-metal`.

Run focused software tests inside that environment:

```bash
~/.venv-vllm-metal/bin/python -m pytest -q \
  tests/test_hf_sampler.py \
  tests/test_vllm_sampler.py \
  tests/test_runtime_metadata.py \
  tests/test_run_probe_modes.py \
  tests/test_macos_runtime_docs.py
```

Then run the exact-checkpoint two-rollout Metal smoke from Task 3 and inspect:

```bash
cat results_m5_smoke/run_config.json
```

Acceptance: native arm64/Python 3.12, tests pass, exact SFT checkpoint loads, two rollouts complete, result schema is normal, and runtime metadata says Metal/vLLM-Metal.

If the run hangs or fails, capture the complete error/log and use systematic debugging. No model substitution is permitted in this task.

- [ ] **Step 4: Fresh RunPod SHA-image acceptance**

Create a disposable Pod from the new `sha-*` image with the existing template environment and startup command.

Verify:

```bash
cat /workspace/rlvr-bootstrap.log
cd /workspace/rlvr_behavior_probe_runpod
git rev-parse --short HEAD
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

Rerun `rlvr-bootstrap` once to confirm idempotency.

Run focused tests:

```bash
python -m pytest -q \
  tests/test_hf_sampler.py \
  tests/test_vllm_sampler.py \
  tests/test_runtime_metadata.py \
  tests/test_run_probe_modes.py \
  tests/test_runpod_image_config.py \
  tests/test_runpod_bootstrap.py
```

Run both one-question smokes:

1. canonical sampling: `top_k=0`, `repetition_penalty=1.0`;
2. JIT audit sampling: `top_k=20`, `repetition_penalty=1.1`.

Acceptance: both complete without manual package installation, missing-`ninja` errors, CUDA-fork failures, or result-metadata ambiguity.

- [ ] **Step 5: Promote stable only after fresh SHA acceptance**

Manually dispatch `Build RunPod vLLM image` with `publish_stable=true`. Confirm the workflow succeeds and moves `ghcr.io/unitedsnakes/rlvr-vllm:0.27.1` to the exact tested build.

Optionally start one final disposable Pod from the stable tag and rerun the short runtime sanity check. Do not claim the infra is complete until the fresh SHA acceptance evidence is recorded.

---

## Final Verification Checklist

- [ ] `python -m pytest -q` passes on the implementation checkout.
- [ ] HF and vLLM receive explicit `top_k` and `repetition_penalty` values.
- [ ] Canonical defaults remain `top_k=0`, `repetition_penalty=1.0`.
- [ ] `run_config.json` records platform/runtime identity and package versions.
- [ ] `--engine vllm --device auto` selects Metal on Apple Silicon without a new engine name.
- [ ] Exact SFT `checkpoint-8-of-10` completes a two-rollout M5 smoke or fails clearly without substitution.
- [ ] New RunPod image contains `ninja-build`.
- [ ] Fresh SHA image passes bootstrap, focused tests, canonical smoke, and `top_k=20` JIT smoke.
- [ ] Stable tag is promoted only after the tested SHA image passes.
- [ ] README clearly says Metal output is not interchangeable with canonical CUDA measurement output.
