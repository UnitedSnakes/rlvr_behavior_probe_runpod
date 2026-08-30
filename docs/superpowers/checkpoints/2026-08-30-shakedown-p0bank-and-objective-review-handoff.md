# Controlled Qwen3 RLVR — Shakedown, p0 Bank, and Objective Review Handoff

Checkpoint date: 2026-08-30, evening. Supersedes
`2026-08-30-truncation-policy-handoff.md`.

Active branch: `controlled-qwen3-rlvr-task6`.

## Executive status

The truncation-policy amendment is implemented, tested, and has passed both
the 20-step engineering pilot and a disposable shakedown. Canonical `pi_0` is
frozen and uploaded. The K=1024 reachability bank is collected.

**Canonical GRPO has still not started.** Three blocking issues were found in
an objective/literature review after the shakedown and must be resolved first.
They are listed under "Blockers" below and are the first work of the next
session.

## What completed today

### Truncation-policy amendment applied

`truncation-policy.patch` applied on top of `31420eb`. Full CPU suite green
(167 passed after later additions). Changes:

- `rewards.py` — `is_truncated_completion`, `make_gsm8k_terminated_binary_reward`,
  `resolve_terminal_token_ids`.
- `train_grpo.py` — reward built from tokenizer terminal ids.
- `config.py`, `configs/grpo_qwen3_0_6b.yaml` — reward renamed,
  `mask_truncated_completions: false`.

### Gate: uncontaminated pi_0 required

The first pilot launch failed closed on
`verify_directory_fingerprint`: `controlled_run_outputs/sft_corrected/pi_0/`
had been polluted by an earlier diagnostic writing `p0_signal_budget/` **into
the frozen checkpoint directory**. The gate was not weakened. All GPU work
since uses the clean mirror at
`/workspace/hf_pi0_corrected_diagnostic_2026-08-30/pi_0`.

Lesson to carry: diagnostics must never write inside a frozen `pi_0` directory.

### 20-step engineering pilot — PASS

`controlled_run_outputs/grpo_pilot_20_truncation_policy`, 2 x A40, ~7 min.

```text
epoch final            0.005353   (matches intended ~0.00535 single-device rate)
reward        0.3438 -> 0.4094
reward_std    0.4489 -> 0.4287    (nonzero variance)
clipped_ratio 0.5094 -> 0.4844
frac_reward_zero_std        0.05
loss / grad_norm     finite throughout
```

Batch semantics matched `GRPO_INVARIANTS` exactly. No OOM. Checkpoint reload
was NOT verified: pilot mode saves no snapshots by design.

### 150-step then 400-step disposable shakedown

150-step trend was ambiguous: `clipped_ratio` flat for the first two thirds
(0.4962 / 0.5019 / 0.4769 by segment). 400 steps were run to reduce noise.

**Discovery: the pilot-mode LR schedule is an artifact.** `pilot_steps` is
passed as `max_steps`, so the cosine scheduler treats it as the whole run.
Measured LR: peak ~9.75e-07 at step 40, 1.20e-07 by step 320, **1.9e-11 by
step 400**. The last ~15-20% of any `--pilot-steps` run trains with the model
effectively frozen. Canonical GRPO spreads the same schedule over ~3736 steps
and does not have this shape at step 400.

Restricting to the high-LR window (step 40-320, 29 logged points):

```text
completions/clipped_ratio   0.4969 -> 0.4250
completions/mean_length     1538   -> 1431
reward                      0.3906 -> 0.4906
frac_reward_zero_std        0.05-0.09 throughout
```

All four acceptance criteria move in the intended direction in that window.
**The gate was called PASS by the operator**, with the LR artifact recorded as
the explanation for the weak tail. This is a judgment call on trend-based
acceptance, not a threshold pass; it is recorded here as such.

### Canonical pi_0 frozen and uploaded

```text
HKReporter/rlvr-behavior-probe-pi0-corrected-canonical-2026-08-30  (private, model)
```

Contains `pi_0/`, `trainer_metadata/`, `sft_run_manifest.json`,
`SHA256SUMS.txt`. Manifest hash matches the diagnostic repo byte for byte;
`pi0_lineage_id = f89fc90226a67a6a3c7374f9c13abadfcecda88f397ab812fa4130f1f425605b`.

The `-corrected-diagnostic-` repo is retained; canonical work should cite the
`-canonical-` repo.

### K=1024 deep p0 bank collected

`sample_p0.py` gained two capabilities (4 new tests):

- `--num-generations` override (diagnostic sampling only; does not touch
  `GRPO_INVARIANTS`).
- `--dataset-indices` for an explicit, non-contiguous question set, plus
  `select_indexed_rows` and an `indices` field in the run manifest.

The fixed 30-question pilot subset in `data/gsm8k_subset.jsonl` was matched
back to GSM8K test natural order at the pinned SHA
`740312add88f781978c0658806c59bc2815b9866`. All 30 matched uniquely. Mapping
qid -> dataset_index is recorded in the merged bank.

Two 15-question shards ran in parallel on GPU 0/1. **Shard 1 was killed by an
external SIGTERM at 14/15 questions** (vLLM aborted 849 in-flight requests);
`dataset_index=1252` was resampled separately. All 30 questions x 1024
rollouts are present in:

```text
controlled_run_outputs/p0_bank_k1024_canonical_pi0/
  shard0of2/          15 questions
  shard1of2/          14 questions
  shard1_retry_1252/   1 question
  p0_bank_k1024_merged.jsonl   30 questions, qid-labelled
```

Reachability-depth result vs the Qwen2.5 pilot curve:

```text
   K   Qwen2.5 pilot   Qwen3 corrected pi_0
   1       17.50/30           15.73/30
   4       25.82/30           23.86/30
  16       28.24/30           28.14/30
  64       29.03/30           29.89/30
 256       30.00/30           30.00/30
1024            n/a           30.00/30
```

Outputs: `analyses/reachability_depth_qwen3_pi0.csv`,
`analyses/reachability_depth_qwen3_pi0_per_question.csv`,
`figures/reachability_depth_qwen3_pi0.png`.

Note: the "RL-success unresolved" and "persistent cases" columns were produced
against a placeholder RL file and are meaningless. Canonical GRPO has produced
no `pi_t` yet.

## Blockers — resolve before canonical GRPO

### 1. vLLM importance-sampling correction may be a length filter

TRL documents `sequence_mask` as Masked Importance Sampling: ratios outside
the clip range are **set to zero and contribute no gradient**. Observed in the
shakedown:

```text
importance_sampling_ratio  mean 0.0013   max 0.0375   min 1.8e-19
grad_norm observed at      0.0 (exact), 2.07e-09, 5.75e-10, 7.03e-08
```

The sequence-level ratio is approximately `exp(sum of per-token logp diff)`,
and the logged per-token difference is ~0.0255 nats over ~1500 tokens. The
ratio is therefore effectively a decreasing function of completion length, and
length correlates with difficulty. If sequences are being masked, the surviving
set is difficulty-biased — reintroducing exactly the filter the truncation
amendment removed. If instead the ratio is applied multiplicatively, the
effective learning rate is ~1000x below nominal.

`train_grpo.py` passes only `vllm_importance_sampling_clip_max=3.0`; no
`clip_min` is configured, and the problem is entirely at the lower bound.

**Action:** instrument the masked/zeroed fraction and confirm TRL 1.12's
`clip_min` default before committing ~20 GPU-hours.

### 2. The K=1024 bank scores a different quantity than the GRPO objective

`sample_p0.py` uses `gsm8k_binary_reward` (correctness only). Canonical GRPO
uses `binary_terminated_final_answer_correctness` (terminated AND correct).
The amendment requires p0 sampling intended for comparison with canonical GRPO
to use the post-amendment definition.

The bank **cannot be rescored offline**: each rollout stores only
`{rollout, correct, text}` — no `finish_reason`, no token ids — and project
rules forbid inferring truncation from decoded text.

The collected bank remains valid as a correctness-only reachability/support
measurement. For the frozen-pool tilt analysis it must be recollected.

**Action:** patch the sampler to record `finish_reason` and token ids so one
bank yields both quantities, then recollect (~2 GPU-hours). Do this before any
`pi_t` deep evaluation, or the same defect propagates to every checkpoint.

### 3. `scale_rewards` is not frozen

`scale_rewards` is absent from `GRPO_INVARIANTS`, and `validate_grpo_config`
only checks membership in `{True, False, "group", "batch", "none"}` rather than
equality to a frozen value. It is currently `"group"` in the YAML and manifests.

This is the single most scientifically load-bearing knob in the project: with
binary rewards, dividing by the group std is exactly what makes the learning
signal depend on difficulty (`w(p) = 1/sqrt(p(1-p))`). It can currently be
changed with every test still green.

**Action:** add `scale_rewards: "group"` to `GRPO_INVARIANTS`.

## Objective/literature review findings (non-blocking)

Reviewed against current RLVR literature. Recorded for the design revisit, not
yet adopted as amendments:

1. **The literal research question has a closed-form answer.** GRPO's weight on
   marginal improvement at difficulty p is `1/sqrt(p(1-p))` (bathtub, diverging
   at the extremes); Dr. GRPO's is flat; MaxRL's is `(1-(1-p)^T)/p`. However the
   *total* advantage magnitude per group is `2*sqrt(k(G-k))`, an inverted U that
   is **zero at both extremes**. The two natural readings of "signal allocated
   to a prompt" have opposite shapes. Which one predicts realized `Delta p_i` is
   an open empirical question and is what this apparatus can actually answer.

2. **Finite-G degeneracy.** At G=16 a prompt with p=0.01 yields a
   non-degenerate group only ~14.9% of the time; at p=0.001, ~1.6%. Realized
   allocation to the low-p0 regime collapses even though the idealized weight
   diverges there. `frac_reward_zero_std` already measures this.

3. **Outcome C is currently unfalsifiable.** All 30 deep-eval questions are
   reachable by K=256 under `pi_0`; none is a boundary-expansion candidate.
   Current literature treats "track problems with base pass@256 = 0" as the
   decisive test. Consider extending the deep bank to low-p0 questions (the
   256-prompt train measurement found 16 with p0=0 at K=16).

4. **DAPO Soft Overlong Punishment was not in the amendment's option set.** The
   amendment compared masking / hard-0 / unchanged, but not DAPO's graded
   length penalty, whose stated motivation is that truncated samples carry
   reward noise that a hard penalty conflates with quality. The hard-0 rule
   still leaves 14/51 dead groups in the `(0, 1/4]` bin.

5. Minor: `sample_p0.py` resolves GSM8K at `revision="main"` rather than the
   pinned SHA (they matched today, but this is unpinned); `beta=0` with no
   predeclared entropy/diversity analysis, though entropy collapse is the direct
   mechanism behind the conclusions this project draws;
   `is_truncated_completion`'s docstring claims to mirror TRL's rule
   (`ids[-1] not in [eos, pad]`) but TRL may test for EOS anywhere in the
   sequence — worth one check.

## Code changes this session

- `controlled_run/sample_p0.py` — `--num-generations`, `--dataset-indices`,
  `select_indexed_rows`, manifest `indices` field.
- `controlled_run/train_grpo.py` — `validate_pilot_steps` upper bound raised
  50 -> 150 -> 500. This is an engineering safety cap, not a frozen scientific
  value; no `GRPO_INVARIANTS` entry was touched. The 150- and 400-step
  shakedowns were not runnable before this.
- `tests/` — 4 new p0 tests; pilot-step bound tests updated. 167 passed.

## Backup state

Pushed to GitHub (`controlled-qwen3-rlvr-task6`): all code, tests, docs,
`analyses/`, `figures/`, `CLAUDE.md`.

Uploaded to Hugging Face:

- `HKReporter/rlvr-behavior-probe-pi0-corrected-canonical-2026-08-30` — canonical `pi_0`
- `HKReporter/rlvr-behavior-probe-results` — K=1024 bank, p0 signal-budget
  rollouts, GRPO pilot/shakedown logs and manifests

**Deliberately not backed up** (regenerable or superseded, ~39 GB):
`sft_corrected/trainer` optimizer state (6.7 GB; `pi_0` is weight-identical to
checkpoint-112 and is preserved), `sft/` (old buggy `im_end` lineage),
`sft_terminal_ab/` (18 GB A/B checkpoints; the finding is documented),
`sft_native_eos_smoke/`. If the pod volume is destroyed these are gone; none is
required for canonical GRPO, which restarts from the uploaded `pi_0`.

## Next steps, in order

1. Resolve blockers 1-3 above.
2. Decide whether to recollect the p0 bank with `finish_reason` recorded.
3. Launch canonical one-epoch GRPO from untouched `pi_0`, 2 x A40, under
   `tmux` or `setsid` — a background job was killed by SIGTERM today, and a
   ~20-hour run cannot be patched up the way a sampling shard was.
4. Deep K=256 evaluation of the five primary policies; wide K=8 full-test
   evaluation.
5. Reachability and frozen-pool analyses; consider the `w(p)` prediction test
   described in the review section above.
