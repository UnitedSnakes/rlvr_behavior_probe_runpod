# Controlled Qwen3 GRPO — Token-Level IS Residual Preregistration

Date: 2026-09-02. Written after the `top_p=1.0` sequence-ledger revalidation pilot and before collecting token-level log-probability-difference diagnostics.

This note does not release the canonical GRPO gate and does not yet change the frozen IS mode. It freezes the next diagnostic predictions and the consequences of the `top_p` amendment for the train-side `p_0` baseline.

## Evidence entering this diagnostic

The disposable 20-step `top_p=1.0` pilot from the untouched canonical `pi_0` produced 640 rollouts across 40 prompt groups (`G=16`). Relative to the earlier `top_p=0.95` pilot, the known nucleus/processed-logprob mismatch was materially reduced:

```text
                              top_p=.95        top_p=1.0
log rho ~ length slope/token  -0.0184845       -0.0010387
R^2                            0.81903          0.11937
sequence-weight ESS/N          0.00965          0.36170
positive sequence-mean delta   0/640            131/640
upper-cap masked               0/640            47/640
```

The residual sequence-level discrepancy remains scientifically material. Across an illustrative completion-length span of 500 to 2000 tokens, the observed slope corresponds to

```text
exp(-0.00111 * 1500) ~= 0.19,
```

or roughly a 5-fold sequence-weight difference. This is larger than the approximately 1.5--1.6-fold nominal difficulty-weight contrasts the controlled signal-allocation experiment is intended to resolve. Canonical GRPO therefore remains blocked.

## Free length/mask check

A post-hoc check using only already-recorded ledger fields tested whether the positive upper-cap tail was itself concentrated among long completions.

```text
upper-cap masked: 47
mean completion length, masked / unmasked:   1508 / 1533
median completion length, masked / unmasked: 1545 / 2017

raw log rho > 0: 131
mean completion length, positive / nonpositive:   1314 / 1588
median completion length, positive / nonpositive: 1207 / 2048
```

These data do **not** support the hypothesis that continuous down-weighting and upper-cap masking are two tails that both preferentially remove long responses. The positive tail is concentrated in shorter responses, while the upper-cap-masked subset is not longer than the unmasked subset.

The working mechanism is therefore narrower: a small negative token-level drift may accumulate with sequence length and drive continuous down-weighting, while the sequence upper cap separately removes occasional positive-tail aggregate ratios. The upper-cap events must not be described as a demonstrated long-response filter.

## Token-level quantities to record

For every rollout, using only active loss-mask tokens, record compact statistics of

```text
Delta_t = log p_train(a_t) - log p_vLLM(a_t).
```

The main JSONL ledger will record:

- active token count;
- mean, population standard deviation, minimum, and maximum of `Delta_t`;
- positive-token fraction;
- `sum exp(Delta_t)` and `sum exp(2 Delta_t)` so token-level IS ESS can be aggregated exactly across rollouts;
- fraction of token ratios above the configured upper cap (`exp(Delta_t) > 3`);
- fraction with `|Delta_t| > 1`;
- the sequence log-ratio after removing the largest `floor(0.01 * n_tokens)` active tokens by `|Delta_t|` within each rollout, plus the retained-token count.

The final item operationalizes the outlier-removal test without storing full per-token arrays. If the heavy-tail diagnostic fails, a separate token-ID/position investigation may then be instrumented.

## Frozen predictions P1--P5

These predictions are written before the token-level pilot.

**P1 — small centered residual.** Token-level `Delta_t` is centered near zero with a small negative mean. The earlier sequence statistics suggest, but do not establish, that ordinary token ratios should be close to one.

**P2 — high token-level ESS.** The aggregate token-level importance-ratio ESS/N is close to 1, in sharp contrast to the sequence-level ESS/N of 0.3617.

**P3 — token upper-cap events are rare.** `frac(exp(Delta_t) > 3)` is near zero. If so, sequence-level `upper_cap_masked` events arise primarily from accumulation rather than individual-token ratios above 3.

**P4 — no substantial token heavy tail.** `frac(|Delta_t| > 1)` is expected to be below 1%. The 1% value is a preregistered diagnostic trigger, not a tuned canonical acceptance threshold. If it is exceeded, stop and investigate token identities, positions, and backend/runtime causes before changing the IS mode.

**P5 — accumulation rather than outliers explains most remaining length dependence.** Let `mu` be the measured token-level mean `Delta_t`. The sequence-level model predicts a `log rho ~ completion_length` slope of approximately `mu` when the token discrepancy is approximately stationary with length. P5 is supported only if both conditions hold:

1. the observed sequence-level slope (currently about `-0.00111/token` in the joint regression) falls within the uncertainty interval used for the measured token-level mean; and
2. replacing each rollout's raw sequence `log rho` with the preregistered 1%-trimmed sequence sum changes the fitted length slope magnitude by less than 20%.

For condition 2, "top 1%" is defined prospectively and deterministically within each rollout: remove the `floor(0.01 * n_active_tokens)` active tokens having the largest `|Delta_t|`, then regress the retained sum on the original completion length using the same prompt-group cluster structure as the untrimmed regression. This definition avoids choosing an outlier threshold after observing the token distribution.

If P1--P4 hold and P5 holds, the primary value of a token-level IS mode would be to prevent small backend discrepancies from multiplying across the full trajectory, not to clip frequent pathological individual tokens. If P4 or P5 fails, that interpretation is rejected and the residual mechanism must be investigated before canonical training.

## `p_0` rebaseline debt created by the top-p amendment

Changing `top_p` from `0.95` to `1.0` changes the rollout sampling distribution. The earlier 256-question train signal-budget measurements collected under `top_p=0.95` are historical diagnostics only. Their quantitative values, including the 51.12% completion-cap rate and the previously reported difficulty-conditioned truncation/live-group gradients, must not be used as confirmatory evidence for the amended recipe without remeasurement.

After the token-level pilot freezes the final sampling/runtime recipe and IS mode, collect one train-side `K=32` bank under `top_p=1.0` and the canonical terminated-and-correct reward. The bank should serve both the truncation/signal-budget revalidation and the confirmatory cross-fit baseline.

### Independent halves

The two `K=16` halves must be separate generation calls rather than a post-hoc split of one `n=32` call. For dataset index `i`, use deterministic distinct seeds such as

```text
half A: seed * 100000 + i
half B: seed * 100000 + i + 50000
```

with otherwise identical frozen sampling settings. This preserves the intended cross-fit separation against regression-to-the-mean artifacts.

### Canonical group size remains G=16

The `K=32` bank improves estimation of `p_0`; it does not redefine a GRPO group. All group-level quantities remain `G=16` quantities.

Primary model-implied live-group probability will be reported as

```text
P(live | p) = 1 - p^16 - (1-p)^16,
```

using the cross-fit train-side `p_0` estimate. The two actual independent `K=16` halves may additionally provide descriptive empirical live/dead group states, but `K=32` must never be treated as the GRPO group size.

The rebaseline must recompute at minimum:

- terminated-and-correct train `p_0` distribution and fixed preregistered bins;
- correctness-only and termination components as descriptive decompositions;
- completion-cap rate;
- cap rate versus difficulty;
- `G=16` live-group probability versus difficulty;
- the two independent `K=16` halves required for cross-fitting.

Do not launch this bank before the token-level pilot has cleared P4, because a failed P4 may force a backend/runtime sampling change that would invalidate the bank again.
