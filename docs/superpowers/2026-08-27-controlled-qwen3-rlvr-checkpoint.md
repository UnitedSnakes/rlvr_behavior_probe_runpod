# Controlled Qwen3 RLVR — Current Checkpoint (updated 2026-08-28)

> Active branch: `controlled-qwen3-rlvr-task6`
>
> Base scientific branch: `controlled-qwen3-rlvr`
>
> Original design: `docs/superpowers/specs/2026-08-26-qwen3-controlled-rlvr-design.md`
>
> Scientific amendment: `docs/superpowers/specs/2026-08-27-qwen3-controlled-rlvr-design-amendment.md`
>
> Long-context / compute amendment: `docs/superpowers/specs/2026-08-27-long-context-sft-compute-amendment.md`
>
> Infra separation design: `docs/superpowers/specs/2026-08-27-dev-a40-infra-separation-design.md`

## Scientific center

The project studies RLVR as behavior-probability dynamics rather than benchmark optimization:

> Given a behavior's pre-RL probability, how does the RL objective allocate learning signal and move that probability over training? How much of the resulting improvement is redistribution over pre-existing reachable behavior, and where do genuinely new answer-level or strategy-level behaviors emerge?

Evidence hierarchy:

```text
reachability / p0
    ↓
learning-signal allocation vs p0
    ↓
frozen-pool control
    ↓
objective intervention
    ↓
strategy redistribution
    ↓
representation probing
```

Controlled lineage:

```text
Qwen/Qwen3-0.6B-Base
  → deterministic contamination-audited OpenR1 reasoning SFT
  → exact fingerprinted pi_0
  → GRPO on GSM8K train
  → predefined pi_t checkpoints
  → held-out GSM8K / fixed-subset deep evaluation
```

Canonical SFT, `pi_0`, and canonical GRPO have not yet been completed. Green unit tests must not be confused with completed scientific runs.

## Frozen SFT recipe

The 2048-token placeholder is superseded. Canonical SFT is now frozen at:

```text
model: Qwen/Qwen3-0.6B-Base
OpenR1 subset: 10,000 train + 512 validation
selection: lowest-index complete verified-correct trace/problem
contamination: GSM8K train + test exact/basic/aggressive/near screen
max_length: 16,384
num_train_epochs: 2
bf16: true
attention: flash_attention_2
gradient checkpointing: true
packing: true, BFD
completion_only_loss: true
optimizer: fused AdamW
lr: 2e-5 cosine
warmup: 0.03
weight decay: 0.01
seed: 42
validation: diagnostic once/epoch, never best-model selection
```

Scientific batch invariant:

```text
global optimizer batch = 64
2 × A40 DDP
per-device microbatch = 1
gradient accumulation = 32
1 × 2 × 32 = 64
```

One-A40 SFT smoke may use the same microbatch/accumulation and therefore has effective batch 32; smoke is non-scientific.

## Evidence for the 16k cutoff

Pinned audit on the contamination-clean verified OpenR1 candidates:

```text
candidate_count: 93,733
removed_no_verified_trace: 28,765
pre_length_filter_count: 64,968

p50: 4,974
p75: 8,096
p90: 11,843.6
p95: 14,128.65
p99: 16,876
p99.5: 17,496.165
p99.9: 18,946.089
max: 22,319

>12,288: 8.8905%
>16,384: 1.6300%
>32,768: 0%
```

Therefore 12,288 would discard about 8.9% of eligible traces, while 16,384 discards only about 1.63% of the extreme tail. The selected canonical cutoff is 16,384.

Expected eligible pool after the 16k filter is approximately 63,909, far above the requested 10,512 train+validation examples.

## Contamination sanity

The full pinned GSM8K reference set contains:

```text
train: 7,473
test: 1,319
total: 8,792
unique exact refs: 8,792
unique basic-normalized refs: 8,792
unique aggressive-normalized refs: 8,792
```

The real OpenR1 scan produced zero exact/basic/aggressive hits. A known GSM8K positive was independently confirmed to hit exact/basic/aggressive and the production near-duplicate index. Together with synthetic matcher tests, zero real removals is treated as a credible data property rather than a broken matcher.

## Canonical data-bundle gate

Normal `controlled_run.prepare_data` writes:

```text
manifests/sft_10k_manifest.jsonl
manifests/sft_val_512_manifest.jsonl
manifests/contamination_audit.json
manifests/source_revisions.json
generated/sft_10k_records.jsonl
generated/sft_val_512_records.jsonl
manifests/data_bundle_manifest.json
```

Before canonical SFT, the runner now requires a verified bundle and rejects:

- missing or hash-mismatched artifacts;
- wrong train/validation counts;
- source-revision mismatch;
- `audit_only=true`;
- `max_formatted_tokens` different from the current SFT config;
- `selected_total_count` different from train+validation size;
- supplied trainer files whose SHA256 does not match the verified bundle.

Smoke SFT remains able to use arbitrary small engineering fixtures.

## Development / production lanes

### M5 Pro development lane

Use the project `.venv` with `controlled_run/requirements-dev.txt` for:

- pytest;
- deterministic `prepare_data`;
- contamination and token-length audits;
- data-bundle materialization/verification;
- provenance and analysis.

vLLM-Metal remains a separate optional environment and is not canonical evidence.

### A40 production lane

Use A40 only when CUDA-specific behavior matters:

- runtime acceptance;
- SFT engineering smoke;
- canonical two-A40 SFT;
- large p0 sampling;
- GRPO pilot/canonical training;
- canonical CUDA evaluation.

Current compute allocation:

```text
SFT engineering smoke: 1 × A40
canonical SFT:          2 × A40, ordinary DDP, global batch 64
preliminary p0:         1 × A40
deep p0 / eval:         up to 4 × A40 independent shards
GRPO pilot:             1 × A40
canonical GRPO:         1 × A40
```

Do not use context parallel for canonical SFT because the frozen recipe uses packing/BFD.

## A40 runtime acceptance

`python -m controlled_run.runtime_acceptance` now checks:

- CUDA is operational;
- the GPU is an NVIDIA A40;
- Torch, Transformers, datasets, Accelerate, TRL, and vLLM import successfully;
- the configured `flash_attention_2` backend is reported available by Transformers.

`docker/rlvr-bootstrap.sh` runs this acceptance gate after repository sync. If FA2 is unavailable, bootstrap fails loudly. It does not install an arbitrary FlashAttention build and does not silently fall back to SDPA.

The current Docker/runtime stack has **not yet been freshly verified with compatible FlashAttention2** after this change. A fresh A40 image/runtime acceptance remains a blocker before canonical SFT.

## Frozen GRPO recipe

Canonical GRPO remains unchanged:

```text
G = 16
generation_batch_size = 32
per_device_train_batch_size = 8
gradient_accumulation_steps = 4
reward = binary final-answer correctness
temperature = 0.8
top_p = 0.95
top_k = 0
repetition_penalty = 1.0
max prompt = 512 hard preflight
max completion = 1024
mask_truncated_completions = true
vllm_max_model_length = 1536
lr = 1e-6 cosine
warmup = 0.10
beta = 0
epsilon = 0.2
num_iterations = 1
loss_type = dapo
scale_rewards = group
bf16 + FlashAttention2 + gradient checkpointing
colocated vLLM, gpu memory utilization 0.30
vLLM importance-sampling correction = true
IS mode = sequence_mask
IS cap = 3
seed = 42
```

First canonical GRPO remains one A40 so generation/update batch semantics are not changed by data parallelism.

## Immediate execution order

1. Let the already-running live A40 16k data materialization finish; do not disturb it for the infra refactor.
2. Pull the latest branch after materialization finishes and run the full CPU pytest suite.
3. Verify the new canonical data bundle and inspect the final audit/counts/hashes.
4. Preserve/copy the verified bundle; no host-specific rematerialization is required.
5. Resolve and verify the pinned A40 runtime with FlashAttention2 through the explicit runtime-acceptance gate.
6. Run a 1×A40 16k SFT engineering smoke and record VRAM/throughput.
7. Run canonical 2×A40 SFT with effective global batch 64 and freeze exact `pi_0`.
8. Estimate GSM8K `p0`; shard deep rollout evaluation independently if more GPUs are used.
9. Run the 20–50-step one-A40 GRPO engineering pilot from untouched `pi_0`.
10. Only after diagnostics pass, run canonical one-epoch GRPO from untouched `pi_0`.

## Verification status

The most recent user-run suite before this infra-alignment batch reached 122/123-pass intermediate states while the 16k tests were being synchronized; subsequent infra commits add new tests and have not yet been freshly verified on the live A40 checkout in this checkpoint. Do not claim the branch is green until a fresh `python -m pytest -q` is observed after pulling the latest head.
