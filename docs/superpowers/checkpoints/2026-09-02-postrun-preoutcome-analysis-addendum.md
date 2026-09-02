# Controlled Qwen3 — Post-run, pre-outcome analysis addendum

## Status

Written 2026-09-02 **after the canonical GRPO run completed and passed structural-integrity checks, but before inspecting substantive canonical outcome curves or checkpoint probability movement**.

This document is **not a preregistration** and must never be described as one. The formal preregistration remains `docs/superpowers/specs/2026-09-01-signal-allocation-analysis-prereg.md`. This addendum records analysis clarifications reached after later engineering discoveries and before substantive canonical outcome inspection.

## Why this addendum exists

The 2026-09-01 preregistration predates several later clarifications: canonical reward/termination analysis, token-level IS adoption, the K=32 train rebaseline, and discussion of shared-parameter transfer. The canonical run has now completed, but only structural completeness/provenance has been inspected. To prevent further analysis drift, the following hierarchy is frozen before opening the substantive canonical results.

## Canonical seed-42 role

The present canonical GRPO trajectory is a **reference/discovery arm and effect-size estimator**, not by itself a confirmatory objective comparison and not an estimate of training-seed uncertainty.

A finding newly selected from this seed-42 trajectory cannot subsequently be called independently confirmed by combining this run with fresh replications. Such findings require fresh training seeds for confirmation.

## Canonical primary descriptive outcomes

For the train allocation panel, the primary descriptive probability-movement outcome is bin-level movement under the already frozen K=32 cross-fit baseline and bins:

```text
p0 bin -> mean Delta p
```

Bins remain exactly:

```text
0
(0, 1/4]
(1/4, 1/2]
(1/2, 3/4]
(3/4, 1)
1
```

Bin-level movement includes both direct exposure and shared-parameter transfer. It must not be interpreted as the causal effect of that bin's own gradients.

The held-out test panel remains transfer/generalization only and must be reported separately from train movement.

## Temporal reward decomposition

The formal preregistration already froze:

```text
P_t(R)
P_t(T)
P_t(C | T)
```

This addendum adds two **post-run/pre-outcome diagnostic curves**, clearly labeled non-preregistered:

```text
P_t(C)
P_t(T | C)
```

The purpose is to guard against composition changes in the conditioning set. In particular, a rise in `P(C|T)` alone is insufficient to claim a correctness-learning phase because the set of terminated trajectories changes as termination is acquired.

A stronger descriptive two-timescale pattern would be:

```text
early: P(T) rises while unconditional P(C) is comparatively flat
later: unconditional P(C) rises
```

Even this remains descriptive in a single training seed and must not be overstated as a causal phase transition without replication or intervention.

## Prompt-local signal versus prompt-local movement

Prompt-level relationships of the form

```text
S_i -> Delta p_i
```

are **exploratory**.

Reason: GRPO updates shared parameters. `Delta p_i` contains the cumulative effect of the prompt's own training exposures plus transfer/interference from all other training prompts. Therefore prompt-local signal is not expected to identify prompt-local causal movement without additional locality assumptions or interventions.

A null or weak prompt-level association is not, by itself, a failure of the canonical analysis.

## Future objective-comparison estimand

The intended confirmatory objective-level quantity for future matched GRPO-versus-MaxRL work is the seed-paired, bin-level contrast

```text
D_s(b) = Delta p_MaxRL,s(b) - Delta p_GRPO,s(b)
```

with the same exact `pi_0`, dataset, reward semantics, sampling policy, batch semantics, checkpoint schedule, and matched training/data seeds wherever mechanically possible. Because the two policies diverge after the first update, matched random seeds do not imply matched realized trajectories; pairing controls initialization/data-order/random-number schedules, not rollout identity.

The number of independent paired training seeds is not frozen here. It should be chosen only after at least one matched GRPO/MaxRL pair provides an empirical scale for both the objective contrast and paired-run variability. Three pairs are a plausible minimum design target, not a predeclared universal sufficiency threshold.

## Predeclared interpretation worlds for future objective comparison

These are recorded now to reduce outcome-dependent reframing later.

### World A — realized allocation materially deviates from nominal allocation

This is the strongest mechanism result. Analysis should ask which implementation transformations account for the deviation and whether realized ledger quantities predict probability movement better than nominal population theory.

### World B — realized allocation approximately matches nominal allocation

Do not search post hoc for a deviation story. The legitimate conclusion is that, under the controlled stack, population-level allocation theory survives finite-G sampling, termination reward, token-level importance correction, and DAPO normalization reasonably well. This may be scientifically useful but is not assumed in advance to be sufficient for a standalone main-conference paper.

### World C — local/objective allocation contrasts are dominated by transfer or training variability

Do not rescue a null by reclassifying arbitrary prompt-level correlations as primary. The next scientific question becomes why local allocation fails to determine local learning, motivating explicit transfer/representation/interference interventions.

## Training-seed uncertainty

Rollout bootstrap/cross-fit uncertainty does **not** estimate between-training-run uncertainty. The present seed-42 canonical curve has no training-seed error bar. Small wiggles or bin differences should not drive follow-up interventions merely because they look visually structured.

Follow-up decisions should prioritize effects that are large, mechanistically interpretable, and connected to independently established stack effects.

## Planned implementation intervention

A matched `sequence_mask` control from the same untouched `pi_0` is planned as a separate implementation-level intervention. It should change only the vLLM importance-sampling aggregation mode relative to the final `token_truncate` recipe, with the same seed and otherwise matched settings.

This intervention can test whether implementation-level IS aggregation causally changes realized training dynamics in this stack. It does **not** by itself justify claims about specific published papers unless their exact relevant implementation conditions are independently verified.

## Freeze rule

After this addendum is committed, no new canonical analysis metric is promoted into the primary/descriptive hierarchy on the basis of seeing the seed-42 outcome. New ideas go to an exploratory idea log and remain explicitly post hoc.
