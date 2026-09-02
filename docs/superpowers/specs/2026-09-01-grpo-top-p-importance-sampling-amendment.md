# Controlled Qwen3 GRPO — Top-p / Importance-Sampling Amendment

## Status

Adopted in two stages on 2026-09-01/02.

1. Rollout sampling changed from `top_p=0.95` to `top_p=1.0` after diagnosing the known top-p / processed-logprob mismatch.
2. vLLM importance correction changes from `sequence_mask` to `token_truncate` after a preregistered token-level residual diagnostic showed that nearly non-degenerate local ratios become strongly degenerate only when multiplied over the full trajectory.

Canonical GRPO remains **blocked** until the final `token_truncate` revalidation pilot and the amended train-side `p_0` rebaseline pass.

The reward, completion cap, batch semantics, optimizer, GRPO loss, reward scaling, and exact `pi_0` lineage remain unchanged.

## Stage 1 — top-p diagnosis

A disposable 20-step pilot from the untouched canonical `pi_0` used the frozen pre-amendment recipe:

```text
temperature: 0.8
top_p: 0.95
top_k: 0
repetition_penalty: 1.0
vllm_importance_sampling_correction: true
vllm_importance_sampling_mode: sequence_mask
vllm_importance_sampling_cap: 3.0
loss_type: dapo
G: 16
generation_batch_size: 32
world_size: 2
```

The signal ledger recorded 640/640 rollouts across 20 generation batches, 40 prompt groups, and two ranks without missing raw or post-mask IS quantities.

### Sequence-level IS degeneracy under `top_p=0.95`

```text
log rho ~ completion_length
slope/token:  -0.0184845
slope/1000:  -18.4845
R^2:           0.81903
correlation:  -0.90500

mean rho:      0.0033637
median rho:    6.57e-12
ESS:           6.18 / 640
ESS/N:         0.00965
upper-cap masked: 0 / 640
```

TRL v1.12 configures colocated vLLM with `logprobs_mode="processed_logprobs"`. With top-p truncation, vLLM therefore reports sampled-token log-probabilities after nucleus restriction and renormalization, while the trainer-side recomputation uses a full-vocabulary log-softmax. For retained nucleus mass `S_t`, the idealized discrepancy is

```text
Delta_t = log p_train(a_t) - log q_vLLM(a_t) = log S_t <= 0
rho_sequence = exp(sum_t Delta_t) = product_t S_t.
```

This mechanism is already documented upstream in Hugging Face TRL issue #6789 and is not claimed as a novel discovery of this project.

The sequence-average discrepancy `raw_log_rho / completion_length` was fully one-sided in the pilot:

```text
mean:     -0.0165095
std:       0.0037209
min:      -0.0276801
median:   -0.0167720
max:      -0.0023332
inside [log(.95), 0]: 640/640
positive:                0/640
```

The marginal association between `k/G` and `log rho` disappeared after completion length was included in the model:

```text
log rho ~ completion_length + k/G
n rollouts:       640
n clusters:        40
R^2:            0.819030

length coefficient:       -0.0184808
cluster SE:                 0.0005039

k/G coefficient:            0.0213822
cluster SE:                  1.6098575
```

The earlier marginal `k/G` gradient is therefore treated as a response-length shadow, not evidence of an independent difficulty-dependent IS effect.

## Stage 1 amendment — full-policy sampling

The canonical rollout recipe uses:

```text
top_p: 1.0
```

This removes nucleus truncation from the behavior policy and makes the rollout distribution match the full temperature-scaled policy that the experiment intends to study. `top_p=1.0` is frozen in `GRPO_INVARIANTS` and must not be changed without another written amendment.

## Stage 2 — residual sequence-IS diagnosis under `top_p=1.0`

Two independent disposable 20-step pilots with `top_p=1.0` showed that the dominant nucleus mismatch was removed, but a smaller backend residual still became scientifically material under sequence multiplication.

Representative first `top_p=1.0` pilot:

```text
sequence-average Delta mean:  -0.0009002
positive sequence-average:     20.5%
log rho ~ length slope/token:  -0.0010387
R^2:                            0.11937
sequence ESS/N:                 0.36170
upper-cap masked:               47 / 640
```

The repeated token-instrumented pilot reproduced the residual:

```text
log rho ~ length slope/token:  -0.00107825
sequence ESS/N:                 0.37121
upper-cap masked:               31 / 640
```

Across an illustrative completion-length span from 500 to 2000 tokens, a slope near `-0.0011/token` corresponds to roughly a five-fold sequence-weight difference. This is larger than the approximately 1.5--1.6-fold nominal difficulty-weight contrasts the controlled signal-allocation experiment is intended to resolve, so `sequence_mask` cannot be used for canonical training.

A free check rejected the hypothesis that the sequence upper-cap mask separately targets longer rollouts:

```text
mean completion length, upper-masked / unmasked:   1508 / 1533
median completion length, upper-masked / unmasked: 1545 / 2017

mean completion length, raw log rho > 0 / <= 0:    1314 / 1588
median completion length, raw log rho > 0 / <= 0:  1207 / 2048
```

The upper-cap events must therefore not be described as a demonstrated long-response filter. The clear length-coupled effect is the continuous negative sequence accumulation.

## Preregistered token-level diagnostic

The token diagnostic was specified before collection in `2026-09-02-token-is-residual-prereg.md`.

Instrumentation consistency passed:

```text
rollouts: 640
total active tokens: 964,165
completion_length != active_token_count: 0
max |raw_log_rho - token_count * token_delta_mean|: 7.60e-7
```

### P1 — small centered local discrepancy: supported

```text
token Delta mean:  -0.00099085
95% prompt-group bootstrap CI:
  [-0.00106958, -0.00091432]
population sd:      0.0450292
min / max:         -0.780895 / +0.664681
positive fraction:  0.501738
mean exp(Delta):     1.0000208
```

### P2 — high token-level ESS: supported

```text
token ESS/N: 0.997967
per-generation ESS/N min / median / max:
0.997802 / 0.997978 / 0.998137
```

### P3 — token upper-cap events are rare: supported

```text
exp(Delta) > 3: 0 / 964,165
```

### P4 — no gross token heavy tail: supported

```text
|Delta| > 1: 0 / 964,165
```

### P5 — stationary bulk alone explains the remaining slope: rejected as preregistered

The preregistered 1% trimming test removed the largest `floor(0.01*n)` active tokens by `|Delta|` within each rollout.

```text
raw slope:                  -0.00107825
1%-trimmed slope:           -0.00072300
relative slope change:       32.95%
preregistered limit:         <20%
```

The raw slope also lay just outside the preregistered prompt-group bootstrap interval for the global token mean. P5 is therefore formally rejected.

A follow-up decomposition explains the rejection without invoking catastrophic token outliers or a small number of bad trajectories:

```text
bulk 99% signed mean:       -0.00069165
top 1% |Delta| signed mean: -0.0328284

kept mean ~ length slope:    -2.22e-8 +/- 9.43e-8
removed-top1% mean ~ length: +9.10e-7 +/- 5.95e-6
mean(Delta) ~ length:        -4.31e-8 +/- 8.78e-8
```

Both the bulk and the moderate tail have approximately stationary signed means with length. The length effect is therefore primarily a **count effect**: longer completions contain proportionally more tokens drawn from the same slightly negative bulk and the same more-negative moderate tail. The tail contribution is distributed rather than concentrated in a few trajectories:

```text
top 1% rollouts share of |tail contribution|:   3.8%
top 5%:                                        15.3%
top 10%:                                       27.3%
top 25%:                                       53.3%
```

One secondary trace is retained for future diagnosis: `std(Delta) ~ length` has slope `+2.46e-6/token +/- 0.64e-6`. This mild widening does not change the present IS-mode decision and P4 found no gross outlier tokens, but it is the first residual to revisit if unexplained length structure remains in canonical data.

## Stage 2 amendment — token-level vLLM importance correction

Canonical GRPO changes to:

```text
vllm_importance_sampling_correction: true
vllm_importance_sampling_mode: token_truncate
vllm_importance_sampling_cap: 3.0
```

`vllm_importance_sampling_clip_min` remains unset / `None`.

Under TRL v1.12, token modes construct `exp(Delta_t)` with shape `[B,T]`; `token_truncate` clamps those local ratios and DAPO multiplies the per-token loss by them. In the diagnostic pilot no observed token ratio exceeded 3, so the upper truncation would have activated zero times on that sample.

The scientific interpretation is therefore:

> The residual mismatch is not driven by catastrophic token-level outliers. Token-level ratios are nearly non-degenerate (ESS/N≈0.998), but a stable negatively biased moderate tail compounds with response length under sequence-level importance weighting. Token-level correction preserves the local vLLM–trainer correction while preventing this trajectory-length multiplication.

`token_truncate` is chosen over disabling IS because it retains local correction for genuine vLLM-versus-training log-probability discrepancies. It is chosen over `sequence_mask` because the latter converts those small local discrepancies into an implementation-induced trajectory-length multiplier larger than the difficulty-allocation signal the experiment is designed to resolve.

## Important claim boundary — DAPO still intentionally weights by token count

Changing the vLLM IS mode removes the **multiplicative, exponential** length coupling introduced by sequence-level importance weighting. It does **not** remove all length dependence from the training objective.

With `loss_type=dapo`, TRL sums per-token contributions and normalizes by the generation-window token count rather than normalizing each sequence to equal mass. Holding per-token quantities fixed, a longer rollout therefore contributes more terms to the global gradient. This is an intentional **additive, linear** token-count weighting.

```text
sequence IS: multiplicative / exponential / penalizes longer responses  <- removed
DAPO:        additive / linear / more tokens contribute more terms       <- retained by design
```

The two effects had opposite signs in the observed stack: sequence IS partially counteracted DAPO's token-count weighting. After switching to token-level IS, that accidental counterweight disappears. A later pilot or canonical run may therefore show increased `completions/mean_length`; such a change is not by itself evidence of regression or a new bug. It is a preregistered consequence of removing the reverse sequence-IS bias.

This distinction is part of the main pipeline analysis: truncation masking and sequence-level IS were unintended length-coupled transformations, whereas DAPO token normalization is a deliberate implementation choice that can also alter realized difficulty allocation.

## Frozen post-amendment recipe

```text
temperature: 0.8
top_p: 1.0
top_k: 0
repetition_penalty: 1.0
mask_truncated_completions: false
reward: binary_terminated_final_answer_correctness
max_completion_length: 2048
vllm_importance_sampling_correction: true
vllm_importance_sampling_mode: token_truncate
vllm_importance_sampling_cap: 3.0
beta: 0.0
epsilon: 0.2
loss_type: dapo
scale_rewards: group
learning_rate: 1e-6
seed: 42
```

Both `top_p=1.0` and `vllm_importance_sampling_mode=token_truncate` are frozen invariants.

## Final revalidation gate before the train-side `p_0` bank

From the untouched canonical `pi_0`, run one disposable 20-step 2 x A40 pilot under the frozen post-amendment recipe.

The signal ledger must preserve two distinct objects:

1. `raw_log_rho = sum_t Delta_t` as a **counterfactual sequence-product diagnostic**. Its old negative length slope is allowed to persist because changing the training IS aggregation does not change the underlying backend discrepancy.
2. the **actual training token ratios** after `token_truncate`, summarized only over active loss-mask tokens.

Acceptance checks:

1. Actual token-level ratio ESS/N remains near the preregistered diagnostic result and is materially non-degenerate relative to the old sequence `ESS/N=0.371`; per-generation token ESS/N must also be inspected.
2. The per-rollout mean actual training token ratio, and equivalently the mean local log-ratio, must show no scientifically material completion-length trend. Do **not** require the counterfactual raw sequence `log rho ~ length` slope to vanish.
3. No sequence is hard-masked by the vLLM correction under `token_truncate`; `upper_cap_masked` must be zero by construction, while the ledger separately reports any token fraction actually clamped at the upper cap. Based on the diagnostic pilot, the preregistered expectation is that the token upper-cap fraction is zero or negligible.
4. Loss, gradient norm, rewards, and ledger fields remain finite and complete; the pilot must not alter any sampling setting or the canonical `pi_0` lineage.
5. Record `completions/mean_length` and interpret an increase against the preregistered DAPO/sequence-IS interaction above rather than treating it automatically as failure.

If these checks fail, canonical GRPO remains blocked and the train-side `p_0` bank must not start.

## Consequences for probability banks and truncation evidence

Changing `top_p` from `0.95` to `1.0` changes the rollout sampling distribution. Earlier train signal-budget measurements under `top_p=0.95`, including the reported 51.12% completion-cap rate and its difficulty gradient, are historical diagnostics only.

After the final token-IS revalidation passes, collect one train-side `K=32` bank under the frozen `top_p=1.0` sampling recipe and the canonical terminated-and-correct reward. It must serve both the truncation/signal-budget revalidation and the confirmatory cross-fit baseline.

The two `K=16` cross-fit halves must be produced by separate generation calls with deterministic distinct seeds, not by splitting one `n=32` call after generation. Canonical group quantities remain `G=16`; the `K=32` bank improves estimation of `p_0` and does not redefine the GRPO group size.

Primary live-group probability remains

```text
P(live | p) = 1 - p^16 - (1-p)^16.
```

The rebaseline must recompute at minimum:

- terminated-and-correct train `p_0` distribution and fixed preregistered bins;
- correctness and termination decompositions;
- completion-cap rate;
- cap rate versus difficulty;
- `G=16` live-group probability versus difficulty;
- the two independent `K=16` halves required for cross-fitting.

Only after the token-IS revalidation and this train-side rebaseline may canonical GRPO launch.
