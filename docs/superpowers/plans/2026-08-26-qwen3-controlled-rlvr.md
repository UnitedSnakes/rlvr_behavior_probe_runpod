# Qwen3 Controlled RLVR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fully traceable `Qwen3-0.6B-Base -> reasoning SFT -> exact pi_0 -> GRPO -> pi_t -> held-out evaluation` pipeline whose saved artifacts support the predefined reachability and frozen-pool analyses without relying on ambiguous checkpoint lineage.

**Architecture:** Add a new `controlled_run/` package for data preparation, provenance, SFT, GRPO, checkpointing, and canonical-run orchestration while leaving the existing Qwen2.5 pilot workflow intact. Reuse the existing numeric scorer and CUDA vLLM inference machinery, but make prompt/tokenizer selection model-profile aware so locally saved Qwen3 policies can be evaluated through the same result schema. Keep training in a separate A40 environment from the existing vLLM 0.27.1 inference image, then analyze stored rollout banks with small pure functions under `analyses/`.

**Tech Stack:** Python 3.12, PyTorch, Transformers, TRL 0.28.0, vLLM 0.25.1 for colocated GRPO generation, Hugging Face Datasets/Hub, FlashAttention 2, PyYAML, pytest, CUDA on one NVIDIA A40 48 GB, existing vLLM 0.27.1 CUDA inference image for canonical evaluation.

**Spec:** `docs/superpowers/specs/2026-08-26-qwen3-controlled-rlvr-design.md`

## Global Constraints

- Base model is `Qwen/Qwen3-0.6B-Base`; resolve and record an immutable Hugging Face commit SHA before training.
- SFT data is a deterministic, contamination-audited 10,000-example subset of `open-r1/OpenR1-Math-220k` with one complete verified-correct reasoning trace per problem and formatted length at most 2048 tokens.
- GSM8K train is the only RL training split; GSM8K test is evaluation-only and must not influence SFT checkpoint selection, GRPO hyperparameters, or checkpoint selection.
- SFT is full-parameter, 2 epochs, bf16, FlashAttention 2, gradient checkpointing, packing, fused AdamW, LR `2e-5`, cosine schedule, warmup ratio `0.03`, weight decay `0.01`, per-device batch 8, gradient accumulation 8, seed 42.
- The final epoch-2 SFT weights are `pi_0` by definition. GRPO must consume that exact local checkpoint and verify its saved fingerprint before trainer construction.
- GRPO uses GSM8K train, one epoch, binary final-answer correctness only, 8 generations/prompt, temperature `0.8`, top-p `0.95`, top-k `0`, repetition penalty `1.0`, max completion length `1024`, LR `1e-6`, cosine schedule, warmup `0.10`, fused AdamW, max grad norm `1.0`, bf16, FlashAttention 2, gradient checkpointing, colocated vLLM, `beta=0`, `epsilon=0.2`, `num_iterations=1`, `loss_type="dapo"`, `scale_rewards="group"`, seed 42.
- Implementation clarification: TRL 0.28.0 `GRPOConfig` has no `max_prompt_length` field. Preserve the spec's 512-token bound as a hard preflight invariant over every GSM8K train prompt and set `vllm_max_model_length=1536` (`512 + 1024`). Do not silently truncate prompts.
- Implementation clarification: make GRPO batching explicit as `per_device_train_batch_size=8`, `gradient_accumulation_steps=1`, and `generation_batch_size=8`. With `num_generations=8`, this is a valid effective prompt batch and is the first A40 pilot configuration; an OOM requires a documented design/config revision rather than a silent canonical change.
- Explicitly set TRL vLLM importance-sampling correction to `True`, mode `sequence_mask`, cap `3.0`; do not rely on library defaults for experiment-defining behavior.
- Save policy-only snapshots every 5% of canonical GRPO progress and resumable Trainer checkpoints at approximately 25%, 50%, 75%, and 100%. Initial scientific analysis uses only pi_0/pi_25/pi_50/pi_75/pi_100.
- Canonical evaluation is CUDA vLLM: full GSM8K test at K=8 for five primary policies, fixed 30-question subset at K=256 for five primary policies, plus pi_0 at K=1024 on the same 30 questions.
- Do not add LoRA/QLoRA, 8-bit optimizers, quantized training weights, FSDP, ZeRO, representation probes, or strategy embeddings in this implementation.
- Keep the existing Qwen2.5 pilot defaults and existing CUDA/Metal inference behavior working; the controlled Qwen3 run is a separate named profile.
- Canonical checkpoints and raw rollouts must be persisted outside ephemeral Pod storage before the Pod is destroyed.
- Implementation is test-first. Hardware pilots are acceptance evidence, not substitutes for unit tests.

## File Map

New controlled-run files:

```text
controlled_run/
  __init__.py
  constants.py              # fixed prompt/profile names and scientific constants
  config.py                 # YAML loading + schema/value checks
  provenance.py             # HF revision resolution, file hashes, manifests
  data.py                   # SFT subset selection + contamination audit + GSM8K RL rows
  rewards.py                # binary GSM8K reward using existing scorer
  checkpointing.py          # pi_0 fingerprints + 5%-policy callback
  artifacts.py              # collision-safe private model-artifact upload
  prepare_data.py           # deterministic materialization CLI
  train_sft.py              # full SFT + pi_0 freeze
  train_grpo.py             # pi_0-verified GRPO pilot/canonical entry point
  requirements-a40.in       # isolated training dependency intent
  configs/
    sft_qwen3_0_6b.yaml
    grpo_qwen3_0_6b.yaml
```

Generated local data and training output lives under `controlled_run_outputs/` and `data/controlled_run/generated/`; both are ignored by Git. Lightweight source-ID/hash manifests and audit summaries live under `data/controlled_run/manifests/` and may be committed after materialization.

Evaluation changes stay in the existing probe package:

```text
probe/prompts.py
probe/model.py
probe/vllm_model.py
probe/runtime.py
probe/runner.py              # reusable one-checkpoint execution
run_probe.py                 # old paired workflow delegates to runner
run_checkpoint_probe.py      # generic single-policy CLI
```

Analysis additions:

```text
analyses/reachability_depth.py
analyses/frozen_pool.py
```

---

### Task 1: Add the controlled-run configuration boundary and isolated A40 environment

**Files:**
- Create: `controlled_run/__init__.py`
- Create: `controlled_run/constants.py`
- Create: `controlled_run/config.py`
- Create: `controlled_run/configs/sft_qwen3_0_6b.yaml`
- Create: `controlled_run/configs/grpo_qwen3_0_6b.yaml`
- Create: `controlled_run/requirements-a40.in`
- Modify: `.gitignore`
- Create: `tests/test_controlled_run_config.py`

**Interfaces:**
- Produces: `CONTROLLED_SYSTEM_PROMPT: str`
- Produces: `load_config(path: Path) -> dict`
- Produces: `validate_sft_config(config: dict) -> None`
- Produces: `validate_grpo_config(config: dict) -> None`
- Later tasks consume the loaded dictionaries without redefining scientific defaults in Python.

- [ ] **Step 1: Write failing tests for the controlled prompt and exact YAML values**

Create `tests/test_controlled_run_config.py` with assertions like:

```python
from pathlib import Path

from controlled_run.config import load_config, validate_grpo_config, validate_sft_config
from controlled_run.constants import CONTROLLED_SYSTEM_PROMPT

ROOT = Path(__file__).resolve().parents[1]


def test_controlled_prompt_requires_boxed_final_answer():
    assert "Reason step by step" in CONTROLLED_SYSTEM_PROMPT
    assert r"\\boxed{}" in CONTROLLED_SYSTEM_PROMPT
    assert "Qwen2.5" not in CONTROLLED_SYSTEM_PROMPT


def test_sft_config_matches_precommitted_recipe():
    cfg = load_config(ROOT / "controlled_run/configs/sft_qwen3_0_6b.yaml")
    validate_sft_config(cfg)
    assert cfg["model_name"] == "Qwen/Qwen3-0.6B-Base"
    assert cfg["num_train_epochs"] == 2
    assert cfg["max_length"] == 2048
    assert cfg["learning_rate"] == 2e-5
    assert cfg["per_device_train_batch_size"] == 8
    assert cfg["gradient_accumulation_steps"] == 8


def test_grpo_config_matches_precommitted_recipe_and_api_clarifications():
    cfg = load_config(ROOT / "controlled_run/configs/grpo_qwen3_0_6b.yaml")
    validate_grpo_config(cfg)
    assert cfg["num_generations"] == 8
    assert cfg["temperature"] == 0.8
    assert cfg["max_prompt_tokens"] == 512
    assert cfg["max_completion_length"] == 1024
    assert cfg["vllm_max_model_length"] == 1536
    assert cfg["per_device_train_batch_size"] == 8
    assert cfg["gradient_accumulation_steps"] == 1
    assert cfg["generation_batch_size"] == 8
    assert cfg["beta"] == 0.0
    assert cfg["loss_type"] == "dapo"
    assert cfg["scale_rewards"] == "group"
    assert cfg["vllm_importance_sampling_correction"] is True
    assert cfg["vllm_importance_sampling_mode"] == "sequence_mask"
    assert cfg["vllm_importance_sampling_cap"] == 3.0
```

- [ ] **Step 2: Run the focused test and verify RED**

```bash
python -m pytest -q tests/test_controlled_run_config.py
```

Expected: import failure because `controlled_run` does not exist.

- [ ] **Step 3: Add the minimal package, loader, and exact configs**

`controlled_run/constants.py`:

```python
CONTROLLED_SYSTEM_PROMPT = (
    "You are a helpful assistant. Reason step by step and put your final answer "
    "within \\boxed{}."
)

BASE_MODEL = "Qwen/Qwen3-0.6B-Base"
SFT_DATASET = "open-r1/OpenR1-Math-220k"
GSM8K_DATASET = "openai/gsm8k"
SEED = 42
```

`controlled_run/config.py` loads YAML with `yaml.safe_load` and rejects missing/changed invariant fields. Keep validation literal and small; do not build a generic configuration framework.

The SFT YAML must explicitly contain the approved recipe and:

```yaml
attn_implementation: flash_attention_2
bf16: true
gradient_checkpointing: true
packing: true
packing_strategy: bfd
completion_only_loss: true
optim: adamw_torch_fused
lr_scheduler_type: cosine
seed: 42
```

The GRPO YAML must explicitly contain the approved recipe plus:

```yaml
per_device_train_batch_size: 8
gradient_accumulation_steps: 1
generation_batch_size: 8
max_prompt_tokens: 512
vllm_max_model_length: 1536
vllm_gpu_memory_utilization: 0.30
vllm_importance_sampling_correction: true
vllm_importance_sampling_mode: sequence_mask
vllm_importance_sampling_cap: 3.0
```

`max_prompt_tokens` is a project preflight field and must not be forwarded as a nonexistent `GRPOConfig` argument.

- [ ] **Step 4: Add the separate A40 dependency intent and output ignores**

Create `controlled_run/requirements-a40.in`:

```text
trl==0.28.0
vllm==0.25.1
transformers>=5,<6
datasets>=4,<5
accelerate>=1,<2
huggingface_hub>=0.36,<1
pyyaml>=6,<7
pytest>=8.3,<9
```

FlashAttention is installed as a separate A40 environment step with `--no-build-isolation`; record its resolved version in the run manifest. Do not modify `docker/requirements-vllm.txt`, because the canonical inference image stays on vLLM 0.27.1.

Append to `.gitignore`:

```gitignore
controlled_run_outputs/
data/controlled_run/generated/
```

- [ ] **Step 5: Run the focused test**

```bash
python -m pytest -q tests/test_controlled_run_config.py
```

Expected: PASS.

- [ ] **Step 6: Commit Task 1**

```bash
git add \
  .gitignore \
  controlled_run/__init__.py \
  controlled_run/constants.py \
  controlled_run/config.py \
  controlled_run/configs/sft_qwen3_0_6b.yaml \
  controlled_run/configs/grpo_qwen3_0_6b.yaml \
  controlled_run/requirements-a40.in \
  tests/test_controlled_run_config.py
git commit -m "Add controlled Qwen3 experiment configuration"
```

---

### Task 2: Add immutable provenance and checkpoint fingerprint helpers

**Files:**
- Create: `controlled_run/provenance.py`
- Create: `tests/test_controlled_run_provenance.py`

**Interfaces:**
- Produces: `resolve_hf_revision(repo_id: str, revision: str, repo_type: str) -> str`
- Produces: `sha256_file(path: Path) -> str`
- Produces: `directory_fingerprint(path: Path, exclude: set[str] | None = None) -> dict[str, str]`
- Produces: `write_json(path: Path, payload: dict) -> None`
- Produces: `verify_directory_fingerprint(path: Path, expected: dict[str, str]) -> None`

- [ ] **Step 1: Write failing pure tests**

Use a fake `HfApi` and temporary files:

```python
def test_resolve_model_revision_returns_server_sha(monkeypatch):
    class FakeApi:
        def model_info(self, repo_id, revision):
            return SimpleNamespace(sha="model-sha")

    monkeypatch.setattr(provenance, "HfApi", FakeApi)
    assert resolve_hf_revision("owner/model", "main", "model") == "model-sha"


def test_directory_fingerprint_detects_changed_weight(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"abc")
    fp = directory_fingerprint(tmp_path)
    verify_directory_fingerprint(tmp_path, fp)
    (tmp_path / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="fingerprint"):
        verify_directory_fingerprint(tmp_path, fp)
```

Also require `repo_type="dataset"` to call `dataset_info`, unsupported repo types to raise, and JSON output to be sorted/indented for stable diffs.

- [ ] **Step 2: Run RED**

```bash
python -m pytest -q tests/test_controlled_run_provenance.py
```

Expected: import failure for `controlled_run.provenance`.

- [ ] **Step 3: Implement the helpers minimally**

`directory_fingerprint()` recursively hashes regular files in lexical relative-path order. For `pi_0`, callers will exclude `pi0_manifest.json` itself so the manifest does not recursively hash itself.

`verify_directory_fingerprint()` must compare both path sets and SHA256 values and raise with the first missing/extra/mismatched relative path.

- [ ] **Step 4: Run GREEN and commit**

```bash
python -m pytest -q tests/test_controlled_run_provenance.py
git add controlled_run/provenance.py tests/test_controlled_run_provenance.py
git commit -m "Add controlled-run provenance fingerprints"
```

---

### Task 3: Build the deterministic OpenR1 SFT subset and contamination audit

**Files:**
- Create: `controlled_run/data.py`
- Create: `controlled_run/prepare_data.py`
- Create: `tests/test_controlled_run_data.py`
- Create directory at execution time: `data/controlled_run/manifests/`

**Interfaces:**
- Produces: `select_verified_trace(row: dict) -> tuple[int, str] | None`
- Produces: `normalize_problem(text: str, aggressive: bool = False) -> str`
- Produces: `word_shingles(text: str, n: int = 5) -> set[str]`
- Produces: `NearDuplicateIndex(reference_texts: list[str], n: int = 5, threshold: float = 0.80)` with `.find_match(text: str) -> int | None`
- Produces: `stable_candidate_key(uuid: str, seed: int) -> str`
- Produces: `build_sft_manifest(openr1_rows, gsm8k_reference_rows, tokenizer, target_size=10_000, seed=42) -> tuple[list[dict], dict]`
- `prepare_data.py` pins source revisions first, then writes tracked ID/hash metadata plus ignored full training records.

- [ ] **Step 1: Write failing tests for verified-trace selection**

Construct synthetic OpenR1 rows whose sequence fields mirror the real schema:

```python
row = {
    "uuid": "u1",
    "problem": "What is 2+2?",
    "generations": ["bad", "reasoning ... \\boxed{4}"],
    "is_reasoning_complete": [True, True],
    "correctness_math_verify": [False, True],
}
assert select_verified_trace(row) == (1, "reasoning ... \\boxed{4}")
```

Require incomplete or unverified traces to be skipped. If multiple traces are eligible, select the lowest generation index so the rule is deterministic.

- [ ] **Step 2: Write failing contamination tests**

Require:

```python
assert normalize_problem("How many? 1,000") == normalize_problem("  HOW many? 1,000  ")
assert normalize_problem("How many? 1,000", aggressive=True) == normalize_problem(
    "How many 1000", aggressive=True
)

index = NearDuplicateIndex(["alpha beta gamma delta epsilon zeta"])
assert index.find_match("alpha beta gamma delta epsilon zeta") == 0
```

Add a near-but-below-threshold case so the threshold is not accidentally treated as substring matching.

Implementation definition:

```text
basic normalization: Unicode NFKC, lowercase, collapse whitespace
aggressive normalization: basic + remove digit-group commas + map punctuation to spaces + collapse whitespace
near duplicate: Jaccard similarity over lowercase aggressive-normalized 5-word shingles >= 0.80
```

Use an inverted shingle index so each OpenR1 candidate is compared only with GSM8K references sharing at least one shingle.

- [ ] **Step 3: Write a failing deterministic-subset test**

Use a fake tokenizer whose `apply_chat_template(..., tokenize=True)` returns a controlled token list. Feed more eligible rows than the requested toy target and assert:

```python
manifest_a, audit_a = build_sft_manifest(..., target_size=3, seed=42)
manifest_b, audit_b = build_sft_manifest(list(reversed(openr1_rows)), ..., target_size=3, seed=42)
assert [r["uuid"] for r in manifest_a] == [r["uuid"] for r in manifest_b]
assert len(manifest_a) == 3
assert audit_a["final_count"] == 3
```

Candidate order must come from SHA256 of `f"{seed}:{uuid}"`, not input row order or Python hash randomization.

Require records longer than 2048 formatted tokens and contaminated candidates to be skipped and counted in the audit.

- [ ] **Step 4: Run RED**

```bash
python -m pytest -q tests/test_controlled_run_data.py
```

Expected: missing functions/module.

- [ ] **Step 5: Implement subset construction**

Each selected manifest row must contain only reproducibility metadata suitable for Git:

```json
{
  "uuid": "...",
  "generation_index": 1,
  "source": "...",
  "problem_sha256": "...",
  "completion_sha256": "...",
  "formatted_token_count": 1374
}
```

The ignored full training JSONL must contain exact prompt/completion records:

```json
{
  "uuid": "...",
  "prompt": [
    {"role": "system", "content": "<CONTROLLED_SYSTEM_PROMPT>"},
    {"role": "user", "content": "<problem>"}
  ],
  "completion": [
    {"role": "assistant", "content": "<selected verified trace>"}
  ]
}
```

The preparation CLI must:

1. resolve immutable SHAs for Qwen3 base/tokenizer, OpenR1, and `openai/gsm8k`;
2. load OpenR1 `default` and GSM8K train+test at pinned revisions;
3. use the pinned Qwen3 tokenizer and its native chat template;
4. build exactly 10,000 records or fail loudly if fewer survive;
5. write `data/controlled_run/manifests/sft_10k_manifest.jsonl`;
6. write `data/controlled_run/manifests/contamination_audit.json`;
7. write ignored `data/controlled_run/generated/sft_10k_records.jsonl`;
8. write `data/controlled_run/manifests/source_revisions.json`.

- [ ] **Step 6: Run GREEN**

```bash
python -m pytest -q tests/test_controlled_run_data.py tests/test_controlled_run_provenance.py
```

Expected: PASS.

- [ ] **Step 7: Commit Task 3 code**

```bash
git add controlled_run/data.py controlled_run/prepare_data.py tests/test_controlled_run_data.py
git commit -m "Add deterministic controlled-run data preparation"
```

Do not commit the real 10k manifest until `prepare_data.py` is executed from pinned source revisions and the audit has been inspected.

---

### Task 4: Implement full SFT and freeze an exact `pi_0`

**Files:**
- Create: `controlled_run/train_sft.py`
- Create: `controlled_run/checkpointing.py`
- Create: `tests/test_controlled_run_sft.py`
- Create: `tests/test_controlled_run_checkpointing.py`

**Interfaces:**
- Produces: `build_sft_arguments(config: dict, output_dir: Path) -> SFTConfig`
- Produces: `load_prompt_completion_jsonl(path: Path) -> Dataset`
- Produces: `freeze_pi0(source_trainer, tokenizer, pi0_dir: Path, lineage: dict) -> dict`
- Produces: `load_pi0_manifest(pi0_dir: Path) -> dict`
- Later GRPO task consumes `pi0_dir` and validates the manifest fingerprint before loading the model.

- [ ] **Step 1: Write failing SFTConfig mapping tests**

Monkeypatch/fake `SFTConfig` if TRL is unavailable in the local analysis environment; assert the constructed kwargs include exactly:

```python
{
    "num_train_epochs": 2,
    "max_length": 2048,
    "bf16": True,
    "gradient_checkpointing": True,
    "packing": True,
    "packing_strategy": "bfd",
    "completion_only_loss": True,
    "learning_rate": 2e-5,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 8,
    "optim": "adamw_torch_fused",
    "seed": 42,
}
```

- [ ] **Step 2: Write failing `pi_0` fingerprint tests**

Use a fake trainer/model that writes `model.safetensors` and a fake tokenizer that writes `tokenizer.json`. Require `freeze_pi0()` to create `pi0_manifest.json` with:

```text
policy_name = pi_0
base_model_sha
sft_dataset_sha
sft_data_manifest_sha256
sft_config_sha256
files = relative-path -> SHA256
```

Then mutate one saved file and require `load_pi0_manifest()` / verification to fail.

- [ ] **Step 3: Run RED**

```bash
python -m pytest -q \
  tests/test_controlled_run_sft.py \
  tests/test_controlled_run_checkpointing.py
```

- [ ] **Step 4: Implement SFT argument mapping and training entry point**

`train_sft.py` must accept:

```text
--config controlled_run/configs/sft_qwen3_0_6b.yaml
--records data/controlled_run/generated/sft_10k_records.jsonl
--source-revisions data/controlled_run/manifests/source_revisions.json
--output-dir controlled_run_outputs/sft
--smoke-steps N   # optional; marks run mode=smoke and never produces canonical pi_0
```

Canonical mode has no `--smoke-steps` and must refuse to start if the records file does not contain exactly 10,000 records.

Load the model/tokenizer from `Qwen/Qwen3-0.6B-Base` at the exact recorded model SHA. Load the model in bf16 with `attn_implementation="flash_attention_2"`. Pass the conversational prompt-completion Dataset and `completion_only_loss=True` to `SFTTrainer`.

Set `save_strategy="epoch"` so epoch-1 is retained for diagnostics. After epoch 2, save a fresh dedicated directory:

```text
controlled_run_outputs/sft/pi_0/
```

containing model/tokenizer/config files only, then fingerprint it and write `pi0_manifest.json`. Do not define `pi_0` as a symlink to a Trainer checkpoint.

- [ ] **Step 5: Run focused tests and old scorer tests**

```bash
python -m pytest -q \
  tests/test_controlled_run_sft.py \
  tests/test_controlled_run_checkpointing.py \
  tests/test_sampling_protocol.py
```

Expected: PASS.

- [ ] **Step 6: Commit Task 4**

```bash
git add \
  controlled_run/train_sft.py \
  controlled_run/checkpointing.py \
  tests/test_controlled_run_sft.py \
  tests/test_controlled_run_checkpointing.py
git commit -m "Add exact pre-RL SFT checkpoint pipeline"
```

---

### Task 5: Add GSM8K RL preparation, binary reward, and the 512-token prompt invariant

**Files:**
- Modify: `controlled_run/data.py`
- Create: `controlled_run/rewards.py`
- Create: `tests/test_controlled_run_rewards.py`
- Modify: `tests/test_controlled_run_data.py`

**Interfaces:**
- Produces: `build_gsm8k_rl_rows(dataset_rows: list[dict]) -> list[dict]`
- Produces: `assert_prompt_token_limit(rows: list[dict], tokenizer, max_tokens: int = 512) -> dict`
- Produces: `completion_text(completion) -> str`
- Produces: `gsm8k_binary_reward(completions, answer, **kwargs) -> list[float]`

- [ ] **Step 1: Write failing reward tests against the existing scorer**

```python
def test_binary_reward_scores_numeric_correctness_only():
    completions = [
        [{"role": "assistant", "content": "work... \\boxed{12}"}],
        [{"role": "assistant", "content": "work... \\boxed{13}"}],
    ]
    assert gsm8k_binary_reward(completions, answer=["12", "12"]) == [1.0, 0.0]
```

Also test plain-string completions, fractions, and an unparseable completion. Reuse `probe.scoring.extract_numeric_answer`, `_to_number`, and `numeric_equal`; do not implement a second answer parser.

- [ ] **Step 2: Write failing RL-row and prompt-limit tests**

Require GSM8K input:

```python
{"question": "1+1?", "answer": "reasoning\n#### 2"}
```

to produce:

```python
{
    "prompt": [
        {"role": "system", "content": CONTROLLED_SYSTEM_PROMPT},
        {"role": "user", "content": "1+1?"},
    ],
    "answer": "2",
}
```

Use a fake tokenizer to make one prompt 513 tokens and assert `assert_prompt_token_limit(..., 512)` raises with the offending row index and maximum observed token length. The function must tokenize with the same chat template and `add_generation_prompt=True` used for generation.

- [ ] **Step 3: Run RED**

```bash
python -m pytest -q tests/test_controlled_run_rewards.py tests/test_controlled_run_data.py
```

- [ ] **Step 4: Implement reward/data helpers**

Do not truncate RL prompts. Return prompt-length summary metadata:

```json
{
  "count": 7473,
  "max_tokens": 187,
  "p95_tokens": 104,
  "limit": 512
}
```

The real values come from the actual pinned dataset; the shape is fixed.

- [ ] **Step 5: Run GREEN and commit**

```bash
python -m pytest -q tests/test_controlled_run_rewards.py tests/test_controlled_run_data.py
git add controlled_run/data.py controlled_run/rewards.py tests/test_controlled_run_rewards.py tests/test_controlled_run_data.py
git commit -m "Add controlled GSM8K reward and prompt preflight"
```

---

### Task 6: Implement GRPO initialization proof and predefined checkpoint saving

**Files:**
- Modify: `controlled_run/checkpointing.py`
- Create: `controlled_run/train_grpo.py`
- Create: `tests/test_controlled_run_grpo.py`
- Modify: `tests/test_controlled_run_checkpointing.py`

**Interfaces:**
- Produces: `progress_step_map(total_steps: int, percentages=range(5, 101, 5)) -> dict[int, int]`
- Produces: `PolicySnapshotCallback(output_dir: Path, tokenizer)`
- Produces: `verify_pi0_for_grpo(pi0_dir: Path) -> dict`
- Produces: `build_grpo_arguments(config: dict, output_dir: Path) -> GRPOConfig`

- [ ] **Step 1: Write failing checkpoint-step tests**

For `total_steps=1000`, require exact percentage mapping:

```python
mapping = progress_step_map(1000)
assert mapping[5] == 50
assert mapping[25] == 250
assert mapping[100] == 1000
assert sorted(mapping) == list(range(5, 101, 5))
```

Add a small-total-steps case and require monotonically increasing unique saved steps; if requested percentages collapse to the same optimizer step, keep the later percentage associated with that step and record the actual map in metadata.

- [ ] **Step 2: Write a failing callback test**

With a fake Trainer state of `max_steps=100`, simulate `global_step=5`, `10`, and repeated `10`. Require policy directories `pi_005`, `pi_010` to be saved once each, only when `state.is_world_process_zero=True`, with model/tokenizer files and `policy_metadata.json` recording actual step, target percentage, and the `pi_0` lineage identifier.

- [ ] **Step 3: Write failing GRPOConfig mapping and lineage tests**

Require `build_grpo_arguments()` to pass explicit TRL fields:

```python
assert kwargs["num_generations"] == 8
assert kwargs["generation_batch_size"] == 8
assert kwargs["per_device_train_batch_size"] == 8
assert kwargs["gradient_accumulation_steps"] == 1
assert kwargs["max_completion_length"] == 1024
assert "max_prompt_length" not in kwargs
assert kwargs["vllm_max_model_length"] == 1536
assert kwargs["use_vllm"] is True
assert kwargs["vllm_mode"] == "colocate"
assert kwargs["vllm_gpu_memory_utilization"] == 0.30
assert kwargs["beta"] == 0.0
assert kwargs["epsilon"] == 0.2
assert kwargs["num_iterations"] == 1
assert kwargs["loss_type"] == "dapo"
assert kwargs["scale_rewards"] == "group"
assert kwargs["vllm_importance_sampling_correction"] is True
assert kwargs["vllm_importance_sampling_mode"] == "sequence_mask"
assert kwargs["vllm_importance_sampling_cap"] == 3.0
```

Construct a fake `pi_0` directory with a valid manifest, verify it passes, mutate a weight, and verify training initialization is rejected before any trainer/model construction call.

- [ ] **Step 4: Run RED**

```bash
python -m pytest -q \
  tests/test_controlled_run_grpo.py \
  tests/test_controlled_run_checkpointing.py
```

- [ ] **Step 5: Implement GRPO entry point**

`train_grpo.py` accepts:

```text
--config controlled_run/configs/grpo_qwen3_0_6b.yaml
--pi0-dir /workspace/.../pi_0
--output-dir controlled_run_outputs/grpo
--mode pilot|canonical
--pilot-steps 20..50   # required only for mode=pilot
```

Both modes perform, in this order:

1. verify `pi_0` fingerprint;
2. load the exact tokenizer from `pi_0`;
3. resolve/pin GSM8K dataset SHA and load train split;
4. construct controlled conversational prompts and numeric answers;
5. enforce the 512-token prompt invariant over all train rows;
6. write the run manifest and prompt-length audit;
7. construct `GRPOTrainer` with the explicit config and `gsm8k_binary_reward`.

Pilot mode writes under a directory whose manifest says `scientific_use=false`, runs the requested 20-50 optimizer steps, and must never create canonical `pi_*` artifact names.

Canonical mode refuses a max-step override, runs one epoch, attaches `PolicySnapshotCallback`, uses Trainer `save_strategy="steps"` and `save_steps=0.25` for resumable ~25% checkpoints, and records `state.max_steps` plus the exact 5%-to-step map once training begins.

Set model init kwargs to bf16 + FlashAttention 2 + exact local `pi_0`; do not load a reference model because `beta=0`.

- [ ] **Step 6: Run GREEN and controlled-run unit suite**

```bash
python -m pytest -q \
  tests/test_controlled_run_config.py \
  tests/test_controlled_run_provenance.py \
  tests/test_controlled_run_data.py \
  tests/test_controlled_run_sft.py \
  tests/test_controlled_run_rewards.py \
  tests/test_controlled_run_checkpointing.py \
  tests/test_controlled_run_grpo.py
```

Expected: PASS without network/GPU access.

- [ ] **Step 7: Commit Task 6**

```bash
git add controlled_run/checkpointing.py controlled_run/train_grpo.py tests/test_controlled_run_checkpointing.py tests/test_controlled_run_grpo.py
git commit -m "Add exact-lineage GRPO training and policy snapshots"
```

---

### Task 7: Make the probe evaluate arbitrary Qwen3 policy checkpoints without breaking the Qwen2.5 pilot

**Files:**
- Modify: `probe/prompts.py`
- Modify: `probe/model.py`
- Modify: `probe/vllm_model.py`
- Modify: `probe/runtime.py`
- Create: `probe/runner.py`
- Modify: `run_probe.py`
- Create: `run_checkpoint_probe.py`
- Modify: `tests/test_hf_sampler.py`
- Modify: `tests/test_vllm_sampler.py`
- Modify: `tests/test_runtime_metadata.py`
- Modify: `tests/test_run_probe_modes.py`
- Create: `tests/test_checkpoint_probe.py`

**Interfaces:**
- Produces: `PromptProfile(name: str, system_prompt: str, tokenizer_name: str | None, tokenizer_revision: str | None)`
- Produces: `get_prompt_profile(name: str) -> PromptProfile`
- Profiles: `qwen25-pilot` and `qwen3-controlled`
- Produces: reusable `run_checkpoint(...)` in `probe/runner.py`
- Produces generic CLI `run_checkpoint_probe.py --model-name-or-path ... --prompt-profile qwen3-controlled ...`

- [ ] **Step 1: Write failing prompt-profile tests**

Preserve the old exact constants as the `qwen25-pilot` profile. Define the controlled profile as:

```python
PromptProfile(
    name="qwen3-controlled",
    system_prompt=CONTROLLED_SYSTEM_PROMPT,
    tokenizer_name=None,
    tokenizer_revision=None,
)
```

`tokenizer_name=None` means load tokenizer/chat template from the evaluated model checkpoint itself.

- [ ] **Step 2: Extend fake sampler tests before production edits**

For the old profile, assert tokenizer loading remains:

```text
Qwen/Qwen2.5-1.5B-Instruct @ main
```

For a controlled local checkpoint `/tmp/pi_025`, assert both HF and vLLM tokenizer sources are `/tmp/pi_025` and revision is `None`.

For a controlled remote model + immutable revision, assert tokenizer source follows the same repo/revision; on Metal, after exact snapshot download, use the resolved snapshot directory for both model and tokenizer.

- [ ] **Step 3: Write a failing generic checkpoint CLI test**

Patch sampler/runtime functions and invoke:

```text
run_checkpoint_probe.py
  --model-name-or-path /tmp/pi_025
  --model-alias pi_025
  --prompt-profile qwen3-controlled
  --question-file data/gsm8k_subset.jsonl
  --questions 30
  --rollouts 256
  --engine vllm
  --device cuda
  --max-new-tokens 1024
  --temperature 0.8
  --top-p 0.95
  --top-k 0
  --repetition-penalty 1.0
```

Require `run_config.json` to identify the prompt profile, tokenizer source, model path, runtime, and full effective sampling policy.

- [ ] **Step 4: Run RED**

```bash
python -m pytest -q \
  tests/test_hf_sampler.py \
  tests/test_vllm_sampler.py \
  tests/test_runtime_metadata.py \
  tests/test_run_probe_modes.py \
  tests/test_checkpoint_probe.py
```

- [ ] **Step 5: Implement prompt profiles and reusable runner**

Move the current one-checkpoint loop from `run_probe.py` into `probe/runner.py` without changing its saved row schema. `run_probe.py` remains the historical paired SFT/RL wrapper and defaults to `qwen25-pilot`.

Update `Sampler` and `VLLMSampler` constructors to accept a `PromptProfile`. Their prompt formatting uses `profile.system_prompt`; tokenizer resolution follows the profile rule above.

Update runtime metadata collection so tokenizer information is passed in by the caller/profile instead of importing hardcoded Qwen2.5 constants.

For local model directories, normalize model revision to `None`; never pass a fake `main` revision to a local path.

- [ ] **Step 6: Implement the generic CLI**

`run_checkpoint_probe.py` supports one checkpoint and writes:

```text
<result-dir>/raw.jsonl
<result-dir>/run_config.json
```

Use the same JSON row schema as the old probe with `model_alias` set from CLI. Reuse the existing private Dataset `--upload-repo` semantics for raw result backup.

- [ ] **Step 7: Run focused and full regression tests**

```bash
python -m pytest -q \
  tests/test_hf_sampler.py \
  tests/test_vllm_sampler.py \
  tests/test_runtime_metadata.py \
  tests/test_run_config_runtime.py \
  tests/test_run_probe_modes.py \
  tests/test_sampling_protocol.py \
  tests/test_checkpoint_probe.py

python -m pytest -q
```

Expected: all tests pass; old Qwen2.5 defaults remain unchanged.

- [ ] **Step 8: Commit Task 7**

```bash
git add \
  probe/prompts.py \
  probe/model.py \
  probe/vllm_model.py \
  probe/runtime.py \
  probe/runner.py \
  run_probe.py \
  run_checkpoint_probe.py \
  tests/test_hf_sampler.py \
  tests/test_vllm_sampler.py \
  tests/test_runtime_metadata.py \
  tests/test_run_probe_modes.py \
  tests/test_checkpoint_probe.py
git commit -m "Support arbitrary controlled policy evaluation"
```

---

### Task 8: Add the predefined reachability and Level-1 frozen-pool analysis

**Files:**
- Modify: `analyses/reachability_depth.py`
- Modify: `tests/test_reachability_depth.py`
- Create: `analyses/frozen_pool.py`
- Create: `tests/test_frozen_pool.py`

**Interfaces:**
- Extend default reachability budgets to `[1,2,4,8,16,32,64,128,256,512,1024]`
- Produces: `tilted_probability(p: float, beta: float) -> float`
- Produces: `fit_global_beta(rows: list[dict]) -> float`
- Produces: `leave_one_out_predictions(rows: list[dict]) -> list[dict]`
- Produces: `binomial_nll(k: int, n: int, p: float) -> float`
- Produces: `delta_sse_explained(actual: list[float], predicted: list[float]) -> float`
- Produces: `delta_correlation(actual: list[float], predicted: list[float]) -> float`
- Produces: `jeffreys_probability(k: int, n: int) -> float`
- Produces: `bootstrap_frozen_pool(rows, repeats: int, seed: int) -> dict`

- [ ] **Step 1: Extend the reachability default test to 1024**

Add:

```python
from analyses.reachability_depth import DEFAULT_K_VALUES
assert DEFAULT_K_VALUES == [1,2,4,8,16,32,64,128,256,512,1024]
```

Keep `simulate_depth_curve` accepting arbitrary lists so old toy tests still run.

- [ ] **Step 2: Write failing frozen-pool math tests**

Use analytically simple cases:

```python
assert math.isclose(tilted_probability(0.5, math.log(2)), 2/3)
assert tilted_probability(0.0, 3.0) == 0.0
assert tilted_probability(1.0, -3.0) == 1.0
```

Create a synthetic 3-question dataset where one known beta generated the post counts and require `fit_global_beta` to recover it within tolerance.

- [ ] **Step 3: Write failing held-out metric tests**

Require each LOO prediction to fit beta without the held-out qid. Test NLL against a hand-computed Bernoulli expression and delta SSE explained against a small vector with known result.

For log-odds summaries use:

```python
jeffreys_probability(k, n) = (k + 0.5) / (n + 1)
```

so 0/n and n/n do not create infinite logits.

- [ ] **Step 4: Write a deterministic bootstrap test**

Bootstrap by resampling stored Bernoulli rollout outcomes within each pre/post question bank, refitting LOO beta each repeat. Same seed must produce identical percentile summaries; another seed may differ.

- [ ] **Step 5: Run RED**

```bash
python -m pytest -q tests/test_reachability_depth.py tests/test_frozen_pool.py
```

- [ ] **Step 6: Implement the pure analysis functions and CLI**

The CLI takes:

```text
--pre-file <pi_0 K=256 raw.jsonl>
--post-file <pi_t K=256 raw.jsonl>
--bootstrap 5000
--seed 123
--output-dir <analysis dir>
```

It writes:

```text
summary.json
per_question.csv
bootstrap_summary.json
```

`summary.json` must include at least global beta/odds multiplier, null vs LOO tilt NLL, null vs tilt MAE, delta SSE explained, delta correlation, and the largest absolute held-out residuals.

Do not implement difficulty-conditioned beta or embedding/strategy analysis in this task.

- [ ] **Step 7: Run GREEN and commit**

```bash
python -m pytest -q tests/test_reachability_depth.py tests/test_frozen_pool.py
git add analyses/reachability_depth.py analyses/frozen_pool.py tests/test_reachability_depth.py tests/test_frozen_pool.py
git commit -m "Add controlled reachability and frozen-pool analysis"
```

---

### Task 9: Add artifact persistence, runbook, and hardware acceptance gates

**Files:**
- Create: `controlled_run/artifacts.py`
- Create: `tests/test_controlled_run_artifacts.py`
- Create: `controlled_run/README.md`
- Modify: `README.md`
- Create after first successful A40 resolution: `controlled_run/requirements-a40.lock`

**Interfaces:**
- Produces: `upload_model_artifacts(local_dir: Path, repo_id: str, remote_path: str) -> None`
- Uses `HF_TOKEN`, a pre-existing private Hugging Face **model** repository, and collision refusal mirroring the existing Dataset-result uploader.

- [ ] **Step 1: Write failing artifact-upload tests**

Use a fake `HfApi`. Require:

```text
repo_type = model
HF_TOKEN required
existing remote prefix rejected
upload_folder called only after collision check
```

Do not create repositories automatically and never store the token in a manifest.

- [ ] **Step 2: Run RED**

```bash
python -m pytest -q tests/test_controlled_run_artifacts.py
```

- [ ] **Step 3: Implement the model-artifact uploader**

Keep this helper separate from `probe/results_upload.py` so the existing raw-result Dataset semantics do not change.

Canonical training commands require an explicit pre-existing model repo ID at upload time. Upload layout:

```text
controlled-qwen3/<run-id>/sft/pi_0/
controlled-qwen3/<run-id>/grpo/policies/pi_005/ ... pi_100/
controlled-qwen3/<run-id>/grpo/resumable/
controlled-qwen3/<run-id>/manifests/
```

- [ ] **Step 4: Write the A40 runbook with exact gates**

`controlled_run/README.md` must document this order:

```text
1. create a fresh isolated Python 3.12 training venv on the A40 Pod
2. install requirements-a40.in
3. install FlashAttention 2 with --no-build-isolation
4. run a version/import check and save pip freeze
5. run full pytest suite
6. run prepare_data.py and inspect contamination audit
7. run 2-step SFT smoke; reload saved smoke checkpoint
8. run canonical 2-epoch SFT; freeze/upload pi_0
9. verify pi_0 fingerprint from a fresh process
10. run 20-50-step GRPO engineering pilot from pi_0
11. inspect: nonzero reward variance, finite loss/gradients, truncation fraction, peak VRAM, vLLM colocate, reloadability
12. discard pilot scientific outputs
13. restart canonical GRPO from untouched pi_0
14. upload predefined policy/resumable checkpoints and manifests
15. run canonical CUDA evaluations
16. upload raw rollout result directories to the private Dataset repo
17. verify remote artifacts before terminating the Pod
```

For the environment lock, after the first successful A40 installation run:

```bash
python -m pip freeze | sort > controlled_run/requirements-a40.lock
```

Commit that exact freeze before the canonical training run. If TRL 0.28.0 + vLLM 0.25.1 cannot support Qwen3 colocate on A40, stop and revise the design/config rather than changing package versions inside a canonical run.

- [ ] **Step 5: Document canonical evaluation commands**

For each of pi_0/pi_25/pi_50/pi_75/pi_100, document wide K=8 and deep K=256 commands through `run_checkpoint_probe.py` with `--prompt-profile qwen3-controlled` and the explicit sampling policy. Document a separate pi_0 K=1024 command for `data/gsm8k_subset.jsonl`.

The runbook must state that all canonical evaluation uses the existing CUDA vLLM inference environment, not the training environment and not Metal.

- [ ] **Step 6: Add documentation regression checks**

Extend or create a small test that reads `controlled_run/README.md` and requires the strings:

```text
Qwen3-0.6B-Base
A40 48 GB
pi_0
pilot
canonical
K=1024
qwen3-controlled
TRL 0.28.0
vLLM 0.25.1
```

- [ ] **Step 7: Run all tests**

```bash
python -m pytest -q
```

Expected: full suite passes on CPU/local test doubles.

- [ ] **Step 8: Commit Task 9**

```bash
git add \
  controlled_run/artifacts.py \
  controlled_run/README.md \
  README.md \
  tests/test_controlled_run_artifacts.py
git commit -m "Document controlled Qwen3 experiment workflow"
```

Add `controlled_run/requirements-a40.lock` in the later environment-resolution commit after it is generated on the A40 host.

---

## Hardware Execution Gates After Implementation

These are execution checkpoints, not unit-test tasks. Do them only after the implementation branch has passed the full local suite.

### Gate A: Fresh inference-image acceptance

Because the recently built SHA image changed the RunPod inference stack, start a fresh Pod from that SHA before using/promoting it. Run both:

```text
canonical one-question smoke: top_k=0, repetition_penalty=1.0
non-canonical JIT smoke: top_k=20, repetition_penalty=1.1
```

Only promote the SHA to stable after both succeed.

### Gate B: A40 training environment

On one fresh A40 48 GB Pod:

```bash
python -c 'import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.is_bf16_supported())'
python -c 'import trl, vllm, transformers; print(trl.__version__, vllm.__version__, transformers.__version__)'
python -c 'import flash_attn; print(flash_attn.__version__)'
python -m pytest -q
```

Expected: GPU reports NVIDIA A40, BF16 support is true, requested packages import, full tests pass.

### Gate C: SFT smoke then canonical SFT

Run a 2-step SFT smoke first. Success requires finite loss, no OOM, a reloadable saved model, and the exact pinned base SHA in metadata. Then start the canonical 2-epoch SFT from the pinned base, never from the smoke output.

After canonical SFT:

```text
pi_0 directory exists
pi0_manifest.json fingerprint verifies in a fresh process
pi_0 uploaded to private model artifact storage
```

### Gate D: GRPO engineering pilot

Start from the untouched `pi_0`. Run 20-50 optimizer steps. Record:

```text
reward mean/std and fraction of zero-variance groups
loss and grad norm finiteness
completion length distribution
fraction hitting 1024-token cap
peak GPU memory
wall-clock generation/training throughput
checkpoint reload success
```

If widespread truncation or OOM occurs, stop. Do not carry pilot changes into the canonical run without updating the config/design and restarting from `pi_0`.

### Gate E: Canonical GRPO and evaluation

Restart from untouched `pi_0`, run one canonical epoch, verify the 5%-policy map and resumable checkpoints, upload artifacts, then run the predefined K=8/K=256/K=1024 CUDA evaluations. Only after remote checkpoint and raw-result verification may the Pod be terminated.

---

## Plan Self-Review

### Spec coverage

- Exact base/SFT/GRPO lineage: Tasks 2, 4, 6.
- Deterministic 10k OpenR1 subset and contamination audit: Task 3.
- Full-parameter SFT recipe and fixed epoch-2 `pi_0`: Task 4.
- Binary GSM8K correctness reward and untouched test split: Task 5.
- GRPO recipe, colocated vLLM, DAPO, beta=0, policy/resumable checkpoint cadence: Task 6.
- Canonical Qwen3 evaluation without breaking Qwen2.5 pilot: Task 7.
- K=1024 reachability and Level-1 frozen-pool metrics/uncertainty: Task 8.
- Artifact persistence and A40/fresh-Pod acceptance workflow: Task 9 + hardware gates.
- Difficulty-conditioned and strategy-level models remain intentionally unimplemented until Level-1 results justify them, matching the spec hierarchy and YAGNI.

### API/spec normalization

Two details are deliberately explicit rather than silently translated:

1. TRL 0.28.0 exposes no `max_prompt_length` in `GRPOConfig`; the design's 512-token maximum is implemented as a full-dataset preflight invariant, and vLLM context is fixed to 1536.
2. The design did not specify GRPO trainer batch fields. The initial A40 implementation fixes prompt batch 8, gradient accumulation 1, and generation batch 8 so the eight-generation grouping is valid and reproducible.

### Placeholder scan

The plan contains no `TBD`, `TODO`, “implement later,” or unnamed production functions. Runtime-dependent outputs such as actual Hugging Face SHAs, step counts, resolved package freeze, and observed GPU metrics are generated artifacts by design and are recorded by the defined interfaces rather than hard-coded guesses.

### Type/interface consistency

`pi_0` provenance flows from `freeze_pi0()` -> `pi0_manifest.json` -> `verify_pi0_for_grpo()` -> GRPO run manifest -> `PolicySnapshotCallback` metadata. Prompt identity flows from `CONTROLLED_SYSTEM_PROMPT` into SFT/RL rows and the `qwen3-controlled` evaluation profile. All analysis consumes the existing rollout JSON schema rather than introducing a second result format.