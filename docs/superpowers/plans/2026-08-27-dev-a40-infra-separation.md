# M5 Development / A40 Production Infrastructure Separation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make M5 Pro the default development/data-preparation environment and reserve A40 for CUDA-specific training, large-scale sampling, and canonical evaluation.

**Architecture:** Separate platform-neutral data/test dependencies from optional vLLM-Metal and CUDA training runtimes. Add a hashed data-bundle manifest so deterministic SFT artifacts can move from M5 to A40 without rematerialization, and make the Docker image the authoritative A40 runtime with explicit import/version checks.

**Tech Stack:** Python 3.12, Hugging Face datasets/transformers, pytest, Docker, RunPod, vLLM, TRL

**Spec:** `docs/superpowers/specs/2026-08-27-dev-a40-infra-separation-design.md`

## Global Constraints

- Do not change canonical SFT or GRPO scientific hyperparameters.
- Do not change pinned model/dataset revision semantics or deterministic source-index selection.
- Keep vLLM-Metal isolated from the ordinary M5 development environment.
- Treat the Docker image as the authoritative A40 runtime.
- Do not silently change the canonical attention backend if FlashAttention compatibility fails.

---

### Task 1: Add platform-neutral M5 development lane

**Files:**
- Create: `controlled_run/requirements-dev.txt`
- Modify: `README.md`

**Interfaces:**
- Produces a Python 3.12 environment capable of `pytest`, `controlled_run.prepare_data`, provenance checks, and analysis without vLLM/TRL/FlashAttention.
- Keeps `~/.venv-vllm-metal` as a distinct optional environment.

- [ ] Add `controlled_run/requirements-dev.txt` containing torch, transformers, datasets, accelerate, huggingface_hub, numpy, pandas, pyyaml, matplotlib, and pytest with ranges compatible with existing controlled-run code.
- [ ] Update README with explicit M5 dev environment creation and the rule that CPU data preparation happens there by default.
- [ ] Verify generic requirements do not contain `vllm`, `trl`, `flash-attn`, or `vllm-metal`.

### Task 2: Add deterministic data-bundle handoff and verification

**Files:**
- Modify: `controlled_run/prepare_data.py`
- Create: `controlled_run/data_bundle.py`
- Create: `tests/test_controlled_run_data_bundle.py`

**Interfaces:**
- `write_data_bundle_manifest(manifests_dir: Path, generated_dir: Path) -> dict`
- `verify_data_bundle(manifests_dir: Path, generated_dir: Path) -> dict`
- Writes `data/controlled_run/manifests/data_bundle_manifest.json`.

- [ ] Write failing tests that construct six tiny canonical artifact files, write a bundle manifest, and require verification to return counts/source revisions after all SHA256 checks pass.
- [ ] Add failing tests for a changed artifact hash, missing file, and count mismatch.
- [ ] Implement `data_bundle.py` with schema version 1, fixed artifact names, SHA256 verification, JSONL line-count validation for 10k/512 manifest and record files, and source-revision/source-identity preservation.
- [ ] Call `write_data_bundle_manifest` at the end of successful `prepare_data` after all six artifacts exist.
- [ ] Run focused bundle tests and full pytest suite.

### Task 3: Make Docker image authoritative for A40 runtime

**Files:**
- Modify: `controlled_run/requirements-a40.in`
- Modify: `docker/requirements-vllm.txt`
- Modify: `docker/Dockerfile`
- Modify: `docker/rlvr-bootstrap.sh`
- Modify: `README.md`

**Interfaces:**
- Docker owns exact vLLM/Transformers runtime versions.
- A40 training extras add TRL without independently downgrading vLLM/datasets.
- Bootstrap prints versions/import status for Torch/CUDA, Transformers, datasets, Accelerate, vLLM, TRL, and gpustat availability.

- [ ] Remove conflicting `vllm==0.25.1` / `datasets<5` pins from `controlled_run/requirements-a40.in`; make it training-extra-only and document that Docker owns the base runtime.
- [ ] Install TRL in the Docker image through the image requirements path and retain `gpustat` as observability convenience.
- [ ] Extend Docker build-time Python smoke to import datasets, accelerate, TRL and print exact versions along with existing Torch/CUDA/vLLM/Transformers versions.
- [ ] Extend bootstrap runtime summary with the same versions and `gpustat` presence.
- [ ] Keep FlashAttention as an explicit runtime acceptance check rather than silently installing/changing the canonical backend in this task.
- [ ] Update README with the M5→bundle→A40 flow and note that A40 need not rerun `prepare_data` when a verified bundle exists.
- [ ] Run CPU pytest; build/runtime smoke remains required on a fresh A40 image before stable promotion.

### Task 4: Update controlled-run checkpoint documentation

**Files:**
- Modify: `docs/superpowers/2026-08-27-controlled-qwen3-rlvr-checkpoint.md`

**Interfaces:**
- Replaces “Use one A40 first” with M5-first development/data preparation, then A40 at the first CUDA-specific gate.

- [ ] Document platform lanes and bundle handoff.
- [ ] Preserve the current live-run exception: the already-running A40 may finish data preparation and continue through runtime smoke/training.
- [ ] Keep all scientific gates and hyperparameters unchanged.
