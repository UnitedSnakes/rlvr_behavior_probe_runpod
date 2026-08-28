# SFT Source Identity and Validation Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make controlled OpenR1 materialization robust to missing/duplicate upstream UUIDs and add a deterministic 512-example SFT validation split without changing the 10,000-example canonical training set or using validation for checkpoint selection.

**Architecture:** Treat the pinned OpenR1 dataset row index as the canonical source identity and retain upstream UUID only as metadata. Select a single deterministic contamination/verification/length-filtered pool, assign the first 10,000 selected rows to training and the next 512 to validation, and train with epoch-level evaluation while keeping epoch 2 as `pi_0` by definition.

**Tech Stack:** Python 3.12, Hugging Face datasets/transformers, TRL SFTTrainer, pytest

**Spec:** `docs/superpowers/specs/2026-08-26-qwen3-controlled-rlvr-design.md` plus `docs/superpowers/specs/2026-08-27-qwen3-controlled-rlvr-design-amendment.md`

## Global Constraints

- Canonical SFT training remains exactly 10,000 OpenR1 examples.
- Validation is 512 additional deterministic held-out OpenR1 examples from the same eligible pool.
- Validation metrics are diagnostic only and must not select the SFT checkpoint; `pi_0` remains the final epoch-2 checkpoint by definition.
- GSM8K train/test remain excluded from SFT train and validation through the existing contamination screens.
- Source identity must not depend on OpenR1 `uuid`, because live data contains missing and duplicate UUID values.
- The pinned dataset revision plus `source_index` defines immutable row identity; problem/completion hashes continue to verify row contents.

---

### Task 1: Replace UUID primary key with source index

**Files:**
- Modify: `controlled_run/data.py`
- Modify: `controlled_run/prepare_data.py`
- Modify: `tests/test_controlled_run_data.py`

**Interfaces:**
- Each OpenR1 row consumed by `build_sft_manifest` carries an integer `source_index` assigned from the pinned dataset order.
- Manifest rows contain `source_index` as the primary identity and retain `uuid` as metadata.
- `materialize_sft_records` resolves source rows by `source_index` and verifies UUID metadata (when present), problem hash, completion hash, and selected generation index.

- [ ] Add failing tests showing repeated/missing UUID metadata is accepted when source indices are unique, and duplicate source indices are rejected.
- [ ] Run focused tests and confirm RED on the current UUID-keyed implementation.
- [ ] Implement source-index ordering/lookup and annotate pinned OpenR1 rows during `prepare_data`.
- [ ] Run focused tests and the full controlled-run suite.

### Task 2: Add deterministic 512-example SFT validation split

**Files:**
- Modify: `controlled_run/prepare_data.py`
- Modify: `controlled_run/train_sft.py`
- Modify: `tests/test_controlled_run_sft.py`
- Modify or create focused prepare-data tests as needed.

**Interfaces:**
- Materialization writes `sft_10k_manifest.jsonl`, `sft_val_512_manifest.jsonl`, `sft_10k_records.jsonl`, and `sft_val_512_records.jsonl`.
- `source_revisions.json` records `target_size=10000` and `validation_size=512`.
- SFTTrainer receives the validation dataset and evaluates once per epoch with `load_best_model_at_end=False`; final epoch 2 remains canonical `pi_0`.
- `pi_0` lineage additionally records the validation manifest SHA256.

- [ ] Add failing tests for 10k/512 disjoint splitting and SFT eval configuration/lineage.
- [ ] Run focused tests and confirm RED.
- [ ] Implement combined 10,512 deterministic selection, split/write both manifests and record files, and wire validation into SFTTrainer.
- [ ] Run focused tests and full suite.

### Task 3: Document the live-data correction

**Files:**
- Modify: `docs/superpowers/2026-08-27-controlled-qwen3-rlvr-checkpoint.md`

- [ ] Record the observed OpenR1 identity issue (664 missing-like UUID rows, including 661 literal `NaN`, plus duplicate nonmissing UUIDs) and the source-index correction.
- [ ] Record the 512-example diagnostic-only SFT validation split and explicitly state that it cannot select `pi_0`.
- [ ] Verify documentation matches implementation and rerun the full test suite before completion.
