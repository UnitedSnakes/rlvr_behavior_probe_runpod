# Controlled p0 Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a canonical Qwen3 p0 sampler that uses exact pi_0 lineage, the GRPO prompt path, GRPO-derived sampling settings, and deterministic one-GPU shards.

**Architecture:** Add one focused module, `controlled_run/sample_p0.py`, with pure helpers for sampling settings, shard bounds, seeds, manifest construction, and one vLLM execution path. Reuse `controlled_run.data`, `controlled_run.rewards`, `controlled_run.provenance`, and `controlled_run.checkpointing`; do not import the historical `probe.prompts` path.

**Tech Stack:** Python 3.12, Hugging Face datasets/transformers, vLLM 0.27.1, PyTorch/CUDA, pytest.

**Spec:** `docs/superpowers/specs/2026-08-29-controlled-p0-sampling-design.md`

## Global Constraints

- Frozen policy lineage must be verified before sampling.
- Held-out dataset is `openai/gsm8k` test split.
- Sampling settings come from `controlled_run/configs/grpo_qwen3_0_6b.yaml`: G=16, T=0.8, top_p=0.95, top_k=0, repetition_penalty=1.0, max_completion_length=1024, seed=42.
- Prompt construction must reuse `build_gsm8k_rl_rows()` and the 512-token fail-closed preflight.
- Tokenizer must come from the evaluated policy directory.
- One GPU per shard; no tensor parallelism.
- Do not modify historical `run_probe.py` or merge PR #5.

---

### Task 1: Pure canonical p0 contract

**Files:**
- Create: `tests/test_controlled_run_p0.py`
- Create: `controlled_run/sample_p0.py`

**Interfaces:**
- Produces `canonical_sampling_settings(config) -> dict`
- Produces `slice_shard(rows, start_index, end_index) -> list[tuple[int, dict]]`
- Produces `question_seed(seed, dataset_index) -> int`

- [ ] Write tests asserting settings are copied from the GRPO config, shard bounds are deterministic/inclusive-exclusive, seeds use original dataset index, and the module does not import `probe.prompts`.
- [ ] Run the focused test and confirm RED because `controlled_run.sample_p0` does not yet exist.
- [ ] Implement only the pure helpers and imports required to satisfy the tests.
- [ ] Run the focused tests and confirm GREEN.
- [ ] Commit.

### Task 2: Lineage, prompt, scoring, and manifest contract

**Files:**
- Modify: `tests/test_controlled_run_p0.py`
- Modify: `controlled_run/sample_p0.py`

**Interfaces:**
- Produces `prepare_p0_rows(raw_dataset, tokenizer, max_prompt_tokens) -> tuple[list[dict], dict]`
- Produces `build_p0_manifest(...) -> dict`
- Reuses `verify_pi0_for_grpo()` semantics or equivalent manifest verification.

- [ ] Add tests proving prompt rows pass through `build_gsm8k_rl_rows()`, the 512-token audit is invoked, lineage is recorded, and the manifest contains dataset SHA/split, GRPO config SHA, sampling config, shard bounds, and pi0 lineage.
- [ ] Run focused tests and confirm RED on missing behavior.
- [ ] Implement the smallest lineage/prompt/manifest helpers using shared controlled-run utilities.
- [ ] Run focused tests and confirm GREEN.
- [ ] Commit.

### Task 3: vLLM execution and CLI

**Files:**
- Modify: `tests/test_controlled_run_p0.py`
- Modify: `controlled_run/sample_p0.py`

**Interfaces:**
- Produces `run_p0(...) -> dict`
- CLI: `python -m controlled_run.sample_p0 --policy-dir PATH --output-dir PATH [--start-index N] [--end-index N] [--gpu-memory-utilization F]`

- [ ] Add a mocked vLLM test showing policy-local tokenizer IDs are supplied through `TokensPrompt`, canonical `SamplingParams` are used, shared `gsm8k_binary_reward` scores all completions, and `p0_raw.jsonl` is written incrementally.
- [ ] Run focused test and confirm RED.
- [ ] Implement vLLM loading with `tensor_parallel_size=1`, policy-local tokenizer, deterministic per-question seeds, raw JSONL output, runtime metadata, and CLI.
- [ ] Run focused tests and confirm GREEN.
- [ ] Commit.

### Task 4: Verification

**Files:** none unless a regression is found.

- [ ] Run `python -m pytest tests/test_controlled_run_p0.py -q` and require all pass.
- [ ] Run `python -m pytest -q` and require the complete suite pass.
- [ ] Inspect GitHub Actions for the final commit and require success before calling the implementation complete.
- [ ] On the live A40, pull the final branch and run a tiny held-out shard before starting canonical p0.
