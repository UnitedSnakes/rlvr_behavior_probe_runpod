# Sampling-truncation scientific-validity amendment

Date: 2026-09-14

## Status

This is a **post-diagnosis, pre-follow-up scientific-validity amendment**.

The engineering problem itself is already known and already mitigated in the canonical pipeline. The earlier `2026-09-01-grpo-top-p-importance-sampling-amendment.md` established that `top_p=0.95` combined with processed rollout log-probabilities and full-vocabulary trainer log-probabilities created a severe sequence-level importance-weight mismatch, and canonical training was amended to use `top_p=1.0` plus token-level importance correction.

This document does **not** claim novelty for the existence of top-p / processed-logprob mismatch, truncated-sampling support mismatch, or the upstream engineering fix. Its purpose is narrower:

> Determine whether the implementation artifact was merely an engineering pathology, or whether it systematically distorted the scientific comparison between RLVR objectives along the same difficulty axis that the GRPO–MaxRL study is intended to measure.

No new training run is authorized by this amendment. The next stage is limited to zero-training-cost mechanistic and counterfactual analyses on existing data. A corrected training rerun is allowed only if those analyses show that the artifact materially changes a scientific conclusion.

## Existing evidence frozen before this follow-up

The prior top-p / importance-sampling diagnosis reported the following results.

### `top_p=0.95`, sequence-level importance correction

```text
log rho ~ completion_length
slope/token:  -0.0184845
R^2:           0.81903
correlation:  -0.90500

mean rho:      0.0033637
median rho:    6.57e-12
ESS:           6.18 / 640
ESS/N:         0.00965
```

At a nominal batch size of 512 trajectories, `ESS/N=0.00965` corresponds to an importance-weight ESS of approximately 4.94 trajectories. This quantity must be described specifically as **importance-weight ESS**, not as the literal gradient batch size.

### `top_p=1.0`, same sequence-product diagnostic

Representative first full-policy pilot:

```text
log rho ~ completion_length
slope/token:  -0.0010387
R^2:           0.11937
sequence ESS/N: 0.36170
```

Thus removing nucleus truncation changed the dominant length slope by roughly an order of magnitude and increased sequence-level importance-weight ESS/N from `0.00965` to `0.36170`.

The repeated token-instrumented `top_p=1.0` pilot showed the same residual scale:

```text
log rho ~ completion_length slope/token: -0.00107825
sequence ESS/N:                           0.37121
```

The prior amendment already established that the remaining full-policy residual is primarily a local backend/log-probability mismatch that becomes scientifically material only after sequence multiplication. Token-level ratios themselves were nearly non-degenerate (`ESS/N≈0.998`).

## Mechanistic model to test directly

For token `t`, let `p_t(a)` be the trainer-side full-vocabulary probability after the same temperature transformation, and let the nucleus sampler retain support `S_t` with total pre-renormalization mass

```text
Z_t = sum_{a in S_t} p_t(a).
```

For an action `a_t` retained by the nucleus sampler, the processed rollout probability is ideally

```text
q_t(a_t) = p_t(a_t) / Z_t.
```

Therefore the per-token log-ratio caused by support truncation alone is

```text
Delta_t^trunc = log p_t(a_t) - log q_t(a_t) = log Z_t <= 0.
```

Under sequence-product importance weighting,

```text
log rho_i^trunc = sum_t log Z_it.
```

If `Z_t` is roughly stationary within a run, this predicts an approximately linear relationship between `log rho` and completion length with slope close to `E[log Z_t]`. The previously observed `-0.0184845/token` slope implies an average multiplicative factor of

```text
exp(-0.0184845) ~= 0.9817
```

per generated token, while the `top_p=1.0` slope implies

```text
exp(-0.0010387) ~= 0.9990.
```

These numerical correspondences are suggestive but are **not yet treated as a point-identified mechanism test**. The direct test below supersedes the heuristic slope comparison.

## Analysis A — direct per-trajectory nucleus-mass prediction

For every rollout in the existing `top_p=0.95` diagnostic for which the pre-truncation token distribution can be reconstructed exactly enough to match the sampler semantics, compute the realized retained mass `Z_it` at each generated token and then

```text
x_i = sum_t log Z_it
y_i = observed raw_log_rho_i
```

The primary mechanistic regression is

```text
y_i = alpha + beta * x_i + epsilon_i.
```

### Primary prediction

If nucleus renormalization is the dominant source of the sequence-product collapse,

```text
beta ~= 1
alpha ~= 0
```

up to the residual trainer-vs-rollout backend mismatch already observed under `top_p=1.0`.

A more informative residual is therefore

```text
r_i = observed raw_log_rho_i - sum_t log Z_it.
```

The main diagnostic is whether the strong `top_p=0.95` completion-length dependence disappears after subtracting the predicted truncation term and falls toward the scale of the independently observed `top_p=1.0` residual.

At minimum, report:

```text
observed log rho ~ completion_length
sum_t log Z_t ~ completion_length
observed log rho ~ sum_t log Z_t
residual r_i ~ completion_length
```

with slope, uncertainty, R^2, and the same grouping/bootstrap convention used in the earlier diagnostics.

### Reconstruction discipline

`Z_t` must be computed using the actual sampling semantics used by the pilot, including temperature and the exact top-p inclusion rule. Do not substitute the nominal `top_p=0.95` constant for realized retained mass. Because nucleus sampling includes the token that crosses the cumulative threshold, realized `Z_t` should usually exceed or equal the nominal threshold and can vary across positions.

If exact reconstruction of the sampler's pre-truncation distribution is impossible from existing artifacts, record that limitation explicitly and use the closest faithful replay available. Do not describe an approximate reconstruction as exact.

## Analysis B — offline counterfactual correction of the scientific ledger

The scientific question is not whether the old implementation produced ugly importance weights; that is already established. The new question is whether the artifact changed conclusions about **where training signal was effectively allocated**.

For each historical rollout with a reconstructable truncation term, define a counterfactual sequence log-ratio with the nucleus-renormalization component removed:

```text
log rho_corrected_i = log rho_observed_i - sum_t log Z_it.
```

The sign convention must be verified against the stored definition of `raw_log_rho` before any result is interpreted. The formula above assumes the stored quantity is `log p_train - log q_rollout`, consistent with the earlier amendment.

Using these corrected weights, recompute the same summaries that were scientifically relevant in the original study, without retraining the model.

At minimum:

```text
importance-weight ESS and ESS/N
clip / mask / truncation rates, as applicable
clip rate by frozen p0 bin
weighted or effective advantage mass by frozen p0 bin
GRPO-versus-MaxRL effective signal-allocation contrast by frozen p0 bin
```

Where the canonical study used a specific realized-signal definition, reuse that definition rather than inventing a new one for this follow-up.

## Difficulty-axis validity test

The key scientific risk is a structured pathway of the form

```text
problem difficulty / p0
  -> response length and/or realized nucleus mass
  -> sequence-product importance attenuation
  -> clipping or effective gradient attenuation
  -> apparent realized signal allocation
```

This must not be inferred merely from a marginal trend in clip rate versus `p0`. Response length, actual policy drift, reward/advantage magnitude, termination behavior, and nucleus mass are all potential mediators or confounders of that marginal relationship.

The counterfactual correction above is therefore the primary validity test:

> After removing the analytically predicted nucleus-renormalization term from the historical importance weights, does the difficulty-wise realized signal allocation materially change?

A useful secondary decomposition is to compare, by frozen `p0` bin:

```text
mean completion length
mean / median sum_t log Z_t
observed importance-weight distribution
counterfactually corrected importance-weight distribution
observed versus corrected clipping / attenuation
observed versus corrected effective advantage mass
```

## Relationship to the GRPO–MaxRL scientific question

The original objective intervention asks whether MaxRL reallocates realized training signal toward initially low-success problems and whether behavioral change follows that reallocation.

This follow-up introduces a validity threat to that comparison:

```text
objective
  -> nominal advantage allocation
  -> implementation-induced IS attenuation
  -> realized effective training signal
  -> parameter update
  -> behavioral change
```

If the IS artifact attenuates long or difficult responses more strongly, then an objective designed to upweight low-`p0` problems can be systematically counteracted by the infrastructure. In that case, an observed GRPO–MaxRL signal difference is partly a property of the interaction between the objective and the old rollout/logprob stack, not solely a property of the objective.

The scientifically interesting result would therefore not be "top-p caused a bug." It would be one of the following stronger statements:

1. the artifact materially changes the measured difficulty-wise signal allocation;
2. it changes the magnitude or direction of the GRPO–MaxRL contrast;
3. after correction, an earlier algorithmic comparison weakens, disappears, strengthens, or reverses.

Only such a result would justify treating this as more than an engineering diagnosis.

## Frozen claim boundaries

The following claims are **not** permitted from the present evidence alone:

- that this project discovered truncated-sampling / full-softmax mismatch first;
- that upstream sampling-mask or distribution-replay fixes are novel contributions here;
- that `ESS/N=0.00965` means the literal gradient batch size was five trajectories;
- that the observed `p0` dependence of clipping already proves a difficulty-aligned causal bias;
- that MaxRL was unfairly evaluated unless the offline correction demonstrates a material effect on the objective comparison;
- that removing the old sequence-level artifact removes all response-length weighting, because DAPO retains an intentional additive token-count weighting.

The admissible current claim is narrower:

> In the historical `top_p=0.95` sequence-importance pilot, a severe sequence-level weight collapse was observed and is quantitatively consistent with compounding nucleus-renormalization factors; the next analysis tests whether that implementation artifact materially distorted difficulty-wise RLVR signal allocation.

## Stop / continue rule

The project branches according to the zero-training-cost analyses above.

### STOP at infrastructure diagnosis

Do **not** launch new corrected training runs if both are true:

1. direct nucleus-mass reconstruction explains the historical sequence collapse but
2. counterfactual removal of the truncation term leaves the scientifically relevant difficulty-wise allocation and GRPO–MaxRL comparison essentially unchanged.

In this world, retain the result as an engineering/debugging finding, workshop note, or methodological appendix if useful. Do not inflate it into a main scientific claim.

### CONTINUE to corrected training validation

A new matched training rerun becomes justified only if the offline correction materially changes a scientific conclusion, for example by:

- substantially changing low- versus high-`p0` effective signal allocation;
- substantially changing the GRPO–MaxRL signal-allocation gap;
- changing the ordering or qualitative interpretation of the compared objectives;
- revealing that an apparent difficulty-dependent effect was largely infrastructure-induced.

If this gate is passed, write a separate pre-run amendment specifying the minimum corrected rerun needed to test the altered conclusion. Do not jump directly from this document to a large multi-seed campaign.

## Minimal outputs before any new GPU training

Produce and archive:

```text
1. table: historical top_p=.95 and top_p=1.0 slopes, R^2, ESS/N
2. figure: observed log rho vs completion length
3. figure: sum_t log Z_t vs completion length
4. figure: observed log rho vs sum_t log Z_t
5. figure: residual [log rho - sum_t log Z_t] vs completion length
6. table/figure: observed vs corrected ESS and clipping by p0 bin
7. table/figure: observed vs corrected effective signal allocation by p0 bin
8. explicit statement of whether the GRPO–MaxRL scientific interpretation changes
```

No new model training is required for these outputs.

## Relationship to the existing top-p amendment

`2026-09-01-grpo-top-p-importance-sampling-amendment.md` remains the canonical engineering decision record. It established the pipeline correction and must not be rewritten to imply that the present scientific-validity question was preregistered before the original diagnosis.

This 2026-09-14 amendment is explicitly post hoc with respect to the discovery of the top-p pathology. Its confirmatory value lies only in freezing the follow-up mechanism test and scientific stop rule **before** running the new nucleus-mass reconstruction and counterfactual objective-comparison analyses.