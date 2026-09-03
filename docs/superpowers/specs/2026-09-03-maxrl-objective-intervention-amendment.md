# MaxRL objective-intervention amendment

Date: 2026-09-03

## Status

This is a **post-GRPO, pre-MaxRL-outcome scientific amendment**.

It is informed by the canonical seed-42 GRPO results, including the exposed-vs-unexposed analysis. It therefore cannot make the GRPO seed42 finding confirmatory. Its purpose is to freeze the scientific question and interpretation hierarchy **before any MaxRL training outcome is observed**.

This document does **not** yet authorize a canonical MaxRL run. The exact finite-G MaxRL estimator, temperature/scale parameter if any, normalization interaction with the existing TRL/DAPO loss, and implementation tests must be derived and verified in a later implementation-specific amendment before GPU training.

## Why the objective comparison changed

The earlier plan treated GRPO-versus-MaxRL mainly as a difficulty-weighting comparison: change the objective, change the nominal weight as a function of pre-RL success probability, then inspect which difficulty bins improve.

The canonical GRPO exposure result makes the causal structure sharper. Direct own exposure is not a stable predictor of question-level correctness improvement, and substantial improvement occurs before a question receives direct training exposure. Therefore a change in **where training signal is allocated** need not imply a matching change in **where behavioral improvement appears**.

The next experiment should explicitly separate these two quantities.

## Scientific chain

The intervention is organized as:

```text
objective
  -> realized training-signal allocation
  -> parameter update
  -> shared-parameter transfer / interference
  -> behavioral-change allocation
```

The two central measurements are therefore:

1. **signal allocation:** where the realized MaxRL training signal lands across the frozen pre-RL difficulty bins;
2. **behavioral allocation:** where fixed-panel `DeltaC` appears across those same bins.

Do not collapse these into one quantity.

## Matched comparison target

The intended matched comparison preserves, wherever mechanically possible:

```text
exact pi0 lineage
same dataset and train order
same canonical reward semantics
same generation policy / sampling settings
same G
same optimizer-batch semantics
same checkpoint schedule
same fixed 256-question train panel
same K=32 A/B cross-fit baseline
same independent C-bank snapshot evaluation protocol
matched training/data/random seeds
```

Matched random seeds do not imply matched realized trajectories after the first divergent update. Pairing controls initialization and stochastic schedules, not rollout identity.

The corrected canonical pre-RL lineage remains:

```text
pi0_lineage_id = f89fc90226a67a6a3c7374f9c13abadfcecda88f397ab812fa4130f1f425605b
HF repo = HKReporter/rlvr-behavior-probe-pi0-corrected-canonical-2026-08-30
```

## Frozen pre-MaxRL hypotheses

### H1 — mechanism check: realized signal allocation should move

If the MaxRL estimator is implemented faithfully and its nominal weighting differs materially from GRPO in this finite-G binary-reward regime, the **realized** signal allocation should shift toward lower-`p0` questions relative to the canonical GRPO reference.

This is the first gate. If the realized signal does not move in the expected qualitative direction, do not interpret downstream behavioral differences as evidence about transfer. First inspect the estimator, finite-G behavior, normalization, clipping/importance correction, and implementation.

No numerical left-shift threshold is frozen here because the exact finite-G estimator and scale are not yet verified.

### H2 — primary behavioral prediction: signal may move while DeltaC allocation remains comparatively stable

The primary behavioral prediction, chosen **before observing MaxRL outcomes**, is:

> MaxRL can materially reallocate realized training signal across difficulty while the shape of question-level correctness improvement changes much less.

This is the objective-level follow-up to the GRPO own-exposure result. If observed, it would provide an independent intervention showing that changing local signal allocation does not automatically localize behavioral improvement to the questions receiving more signal.

Do not call such a result "proof that representations dominate." The supported interpretation would be that shared-parameter transfer/interference substantially mediates the mapping from local signal to local behavioral change.

### H3 — alternative: both signal and behavior move

If MaxRL shifts realized signal allocation and the `DeltaC` allocation shifts in the corresponding direction, conclude that objective-level allocation can influence behavioral-change allocation in this controlled stack.

This would not invalidate the GRPO exposure result. It would instead imply that cross-question transfer does not fully wash out objective-level allocation differences.

### H4 — diagnostic failure world: signal does not move

If the MaxRL run does not produce the expected qualitative signal reallocation, treat the behavioral comparison as mechanistically uninterpretable until the estimator/implementation is checked.

Do not rescue this world by redefining the behavioral target post hoc.

## Primary outcomes for the matched comparison

At minimum, report separately:

```text
realized signal allocation by frozen p0 bin
DeltaC by frozen p0 bin
DeltaT by frozen p0 bin
DeltaR by frozen p0 bin
```

For the objective contrast, the natural seed-paired bin-level quantity remains:

```text
D_s(b) = DeltaC_MaxRL,s(b) - DeltaC_GRPO,s(b)
```

with analogous signal-allocation contrasts.

The existing exposure split may be rerun under MaxRL as a secondary mechanism diagnostic, but it is not promoted here above the objective-level signal-vs-behavior comparison.

## What is not frozen yet

The following are intentionally **not** specified in this amendment:

- the exact finite-G MaxRL advantage/gradient estimator;
- the MaxRL scale/temperature/horizon parameter (including whether it should equal `G`);
- how the estimator composes with the existing DAPO token/loss normalization;
- any numerical prediction for the magnitude of signal reallocation;
- the number of independent matched training-seed pairs;
- GPU launch commands.

These must not be guessed from memory or inferred from the population-limit formula. Before implementation, re-derive the exact estimator from the source paper and verify it against small synthetic finite-G cases.

## Implementation gate before GPU

Before any canonical or shakedown MaxRL GPU run:

1. verify the exact finite-G estimator from the source paper;
2. write an implementation-specific amendment fixing the estimator and all scale parameters;
3. add RED tests on synthetic binary groups where the expected finite-G weights can be computed exactly;
4. implement the objective and make those tests GREEN;
5. run the full CPU suite;
6. run a very short GPU pilot and inspect ledger quantities;
7. run a longer shakedown only if the mechanism check behaves as expected;
8. only then launch a matched canonical run from the untouched corrected `pi0`.

A passing CPU suite establishes code semantics only. It is not scientific validation.

## Reporting discipline

- The GRPO exposure finding was discovered on seed42 and must remain labeled as such.
- This amendment is pre-outcome for MaxRL and may be cited as the frozen hypothesis hierarchy for that intervention.
- Do not choose between H2 and H3 after looking at MaxRL and then describe the chosen one as the original prediction. H2 is primary; H3 is the predeclared alternative.
- If H1 fails, stop mechanistic interpretation and debug the objective implementation before discussing H2/H3.
- Do not use a new MaxRL result as a reason to rewrite the historical GRPO preregistration.

## Relationship to the workshop paper

The MATH-AI workshop submission does not depend on MaxRL completing. The current GRPO exposure result is sufficient for the paper's main mechanism claim.

MaxRL is a follow-up intervention. If a clean result becomes available before the workshop deadline, it may be included only if it is fully validated and does not displace the already-frozen GRPO claim hierarchy. Otherwise it remains future work.
