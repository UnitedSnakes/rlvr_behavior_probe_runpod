# Controlled Qwen3 RLVR — Current Checkpoint (2026-08-27)

> Branch under active review: `controlled-qwen3-rlvr-task6`
>
> Base scientific branch: `controlled-qwen3-rlvr`
>
> Original design: `docs/superpowers/specs/2026-08-26-qwen3-controlled-rlvr-design.md`
>
> Latest design amendment (authoritative where it differs): `docs/superpowers/specs/2026-08-27-qwen3-controlled-rlvr-design-amendment.md`

## Scientific center of gravity

The project studies RLVR as behavior-probability dynamics rather than benchmark optimization:

> Given a behavior's pre-RL probability, how does the RL objective allocate learning signal and move that probability over training? How much of the resulting improvement is redistribution over pre-existing reachable behavior, and where do genuinely new answer-level or strategy-level behaviors emerge?

Current evidence hierarchy:

```text
reachability / p0 estimation
    ↓
learning-signal allocation vs p0
    ↓
frozen-pool reweighting
    ↓
objective intervention (GRPO ↔ controls ↔ later MaxRL)
    ↓
strategy redistribution
    ↓
representation probing
```

The older Qwen2.5 pilot remains motivating evidence only because the exact PPO initialization revision is not proven. Its strongest observations remain: deep SFT sampling removed apparent answer-level RL-only successes on the fixed 30 GSM8K problems; a global reward tilt captures aggregate improvement but not strong per-problem heterogeneity; decoding protocol materially changes measured RL gain.

## Controlled lineage

The intended causal backbone remains:

```text
Qwen/Qwen3-0.6B-Base
  → deterministic contamination-audited 10k OpenR1 reasoning SFT
  → exact fingerprinted pi_0
  → GRPO on GSM8K train
  → predefined pi_t checkpoints
  → held-out GSM8K / fixed-subset deep evaluation
```

The live canonical 10k materialization, canonical SFT, `pi_0`, and canonical GRPO scientific run have **not** yet been executed in this implementation session. Do not confuse green unit/CI tests with completed scientific runs.

## Current frozen GRPO recipe

The 2026-08-27 amendment supersedes the older G=8 recipe:

```text
num_generations: 16
generation_batch_size: 32
per_device_train_batch_size: 8
gradient_accumulation_steps: 4

reward: binary final-answer correctness
temperature: 0.8
top_p: 0.95
top_k: 0
repetition_penalty: 1.0

max prompt: 512 hard preflight, never truncate
max completion: 1024
mask_truncated_completions: true
vllm_max_model_length: 1536

learning_rate: 1e-6
cosine; warmup 0.10
beta: 0
epsilon: 0.2
num_iterations: 1
loss_type: dapo
scale_rewards: group

bf16; FlashAttention2; gradient checkpointing
colocated vLLM; gpu memory utilization 0.30
vLLM importance-sampling correction: true
IS mode: sequence_mask
IS cap: 3.0
seed: 42
```

On the intended first single-A40 run this gives two independent prompts × sixteen completions per generation batch and an effective optimizer batch of 32.

`loss_type=dapo` means only DAPO-style global active-token loss normalization; this is not a full DAPO recipe. Group reward scaling is deliberately retained in the main GRPO arm but is recognized as a difficulty-dependent mechanism and is permitted to vary in predeclared controls (`none`/False or batch scaling).

## Task 6 implementation status

Implemented on `controlled-qwen3-rlvr-task6`:

- `controlled_run/train_grpo.py`
  - verifies exact `pi_0` manifest/fingerprint before constructing the trainer;
  - pins GSM8K revision;
  - hard-checks prompt token lengths;
  - maps explicit TRL GRPOConfig fields;
  - separates 20–50-step `pilot` from `canonical` mode;
  - records `scientific_use=false` for pilot;
  - saves canonical policy-only snapshots every 5% using a predefined step map;
  - writes `pi_0` lineage ID into snapshot metadata.
- GRPO config/invariants revised to G=16 / generation batch 32 / gradacc 4.
- `mask_truncated_completions=true` is explicit.
- reward scaling controls are accepted by the validator without changing the canonical `group` value.
- generation batches are required to contain at least two independent prompts on the intended single-device configuration.

The CPU CI test suite at commit `931c287f0ec8d861414e67fa17d417c46e4434b7` passed:

```text
107 passed
```

This verifies pure Python/config/checkpoint logic only. A real A40 TRL/vLLM smoke is still required.

## SFT 2048 audit improvement

The existing SFT selector still rejects formatted verified traces longer than 2048 tokens; the cutoff has **not** been changed.

The audit now additionally measures the contamination-clean, verified candidate distribution *before* the length cutoff:

```text
pre_length_filter_count
formatted_token_percentiles: p50, p75, p90, p95, p99
formatted_token_tail_fractions: >2048, >4096, >8192
removed_too_long
```

The denominator deliberately excludes candidates already removed for no verified trace or GSM8K contamination. Therefore the live audit isolates how much selection pressure is introduced by the length cutoff itself.

Do not decide 2048 vs 4096 until the real pinned materialization has been run and these values inspected.

## Pilot diagnostics

The first GRPO pilot should be 20–50 optimizer steps and should not be used as scientific evidence. Highest-priority diagnostics:

```text
frac_reward_zero_std
reward
completion clipped ratio / maximum terminated length
gradient norm
entropy
clip diagnostics
vLLM sampling-logp mismatch
importance-sampling ratio / masking
```

`frac_reward_zero_std` is central because binary group-relative training gives no useful centered gradient on all-correct/all-wrong groups. This is especially important for low-p prompts, which are part of the scientific target rather than nuisance examples.

Keep completion cap 1024 if clipping is negligible. If clipping is material, move to 2048 as a documented engineering correction before canonical training rather than allowing the cap to become an implicit anti-long-reasoning reward.

## Immediate execution order

1. Use one A40 48 GB first.
2. Run real pinned data materialization and inspect contamination + SFT length audit.
3. Decide whether the SFT 2048 cutoff is scientifically acceptable before canonical SFT.
4. Run an SFT engineering smoke, then canonical 2-epoch SFT and freeze exact `pi_0`.
5. Measure the `pi_0` GSM8K pass-rate distribution sufficiently to characterize low/medium/high-p regimes.
6. Run the 20–50-step GRPO engineering pilot from untouched `pi_0` and inspect zero-std/clipping/vLLM diagnostics.
7. Only after those gates, run canonical one-epoch GRPO from untouched `pi_0`.
8. Then implement/use generic checkpoint probing and the predefined K=8/K=256/K=1024 evaluation plan.

## Planned scientific extensions, not blockers for the first canonical run

- Analyze `p_i(0) → training signal → p_i(t)` directly, including probability/logit movement by pre-RL pass-rate bins.
- Frozen-pool control: restrict an offline learner to a fixed `pi_0` trajectory pool and measure how much online RL behavior can be recovered by reweighting alone.
- Predeclared reward-scaling intervention: canonical group scaling vs no std scaling (and possibly batch scaling), all else fixed.
- Later MaxRL arm from the same `pi_0`, data, rollout budget, and decoding protocol to test objective-dependent amplification of rare-but-reachable successes.
- Longer overtraining/data-scarce stress trajectory after the one-epoch canonical dynamics are understood.
- Strategy-level clustering/structural analysis after probability-level mechanisms are characterized.

## Compute note

Start with one A40. Two A40s are not automatically scientifically equivalent because naïve data parallelism changes global generation/update batch semantics. If a second A40 becomes necessary, explicitly remap batch sizes so the number of independent prompts, completions per prompt, and effective optimizer batch remain controlled.
