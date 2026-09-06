# ATTRIB deadline post-outcome sensitivity and writing correction

Date: 2026-09-06

## Status

This checkpoint is written after the canonical seed42 GRPO exposure result and
the canonical seed42 GRPO-vs-MaxRL result were already visible.

It records two **post-outcome exploratory** CPU diagnostics prepared for the
ATTRIB deadline, plus corrections to writing summaries. It does not rewrite the
historical pre-outcome records and does not add new training, generation, or
gradient measurements.

For paper drafting, this checkpoint supersedes stale summary numbers in the
2026-09-05 ATTRIB writing handoff where they conflict.

## Correction 1 — exposure classification counts

The accepted
`analyses/canonical_exposure_split_adjusted/adjusted_symmetric.csv`
contains 15 symmetric cutoff x p0-bin cells.

Recomputing the frozen descriptive labels gives:

```text
transfer_compatible:      7
unexposed_higher:         3
mixed_or_uncertain:       2
not_classifiable:         3
own_exposure_candidate:   0
```

The earlier summary value `8 / 3 / 2 / 2 / 0` was a documentation counting
error. The three not-classifiable cells are:

```text
25%  (.75,1)
45%  0
65%  (.75,1)
```

This correction does not change the continuous estimates or the fact that
there are zero `own_exposure_candidate` cells. The paper should rely on the
continuous effect estimates rather than treat these descriptive labels as
hypothesis tests.

Historical result checkpoints are left unchanged; this file is the correction
record.

## Correction 2 — MaxRL aggregate DeltaC rounding

The shared initial aggregate correctness is:

```text
C0 = 0.5045166015625
```

The canonical MaxRL endpoint aggregate correctness is:

```text
C100 = 0.56396484375
```

Therefore:

```text
DeltaC = 0.0594482421875 = +5.94482421875 pp
```

Rounded to two decimals this is **+5.94 pp**, not +5.95 pp.

## Exploratory diagnostic 1 — paired question resampling

The exposure sensitivity analysis resamples the 256 question IDs with
replacement 3,000 times using random seed `20260906`.

One question multiplicity is shared across:

```text
both cross-fit directions
all three exposure cutoffs
all rows belonging to that question
```

Within each resample, the original frozen OLS adjustment is refit and adjusted
group means are evaluated at the resampled pooled covariate mean. Before
resampling, all 15 accepted symmetric point estimates are reproduced to less
than `1e-12` absolute error.

For the primary low-nonzero reward bin `(0,.25]`:

| cutoff | not-yet-exposed DeltaC | pointwise 2.5--97.5% range | U-E point | U-E range |
|---:|---:|---:|---:|---:|
| 25% | +10.02 pp | [6.64, 13.30] | +3.77 pp | [-3.33, 10.60] |
| 45% | +12.80 pp | [7.55, 17.87] | +5.38 pp | [-1.41, 11.85] |
| 65% | +12.80 pp | [7.68, 18.13] | +1.90 pp | [-4.26, 8.09] |

The not-yet-exposed gain stays positive in these question-composition
resamples, while the U-E gap ranges cross zero at all three cutoffs.

Leave-one-question-out sensitivity for the same low bin gives minimum
not-yet-exposed gains:

```text
25%:  9.49 pp
45%: 11.78 pp
65%: 11.48 pp
```

Thus the strongest paper-safe statement is:

> In the observed seed42 trajectory, substantial correctness improvement is
> already present before a question's own direct RL exposure.

Do not upgrade the U-E comparison into a claim that not-yet-exposed questions
are statistically better, that own exposure has zero effect, or that the two
groups are equivalent.

### Resampling interpretation boundary

These are pointwise, post-outcome sensitivity ranges conditional on:

```text
the observed training trajectory
the frozen panel and bin memberships
the existing response banks
```

They primarily describe sensitivity to question composition.

They do **not** estimate:

```text
between-training-seed variance
a randomized exposure treatment effect
a complete repeated-response sampling distribution on the fixed panel
statistical equivalence
```

There is no multiplicity correction.

All three low-bin cells retain all 3,000 valid fits. Small high-reward cells can
become rank deficient under resampling; their valid-replicate counts are
retained rather than silently dropped.

## Exploratory diagnostic 2 — absolute versus relative correctness allocation

The existing objective-comparison outcome is the within-bin absolute contrast:

```text
d_b(t) = DeltaC_MaxRL,b(t) - DeltaC_GRPO,b(t)
```

A supplementary descriptive contrast centers this by the whole-panel
between-objective correctness difference:

```text
q_b(t)
  = d_b(t)
    - [C_MaxRL,panel(t) - C_GRPO,panel(t)]
```

Because the two objectives share the same initial whole-panel baseline, the
bracket is also the whole-panel difference in correctness change.

This is an algebraic comparison, **not** a causal intervention that holds
overall model quality fixed.

At the 100% endpoint:

```text
whole-panel MaxRL - GRPO correctness = -1.343 pp
```

| p0 bin | absolute d_b | centered q_b | mean q_b over 20 snapshots | q_b > 0 |
|---|---:|---:|---:|---:|
| 0 | -4.712 | -3.369 | -0.162 | 9/20 |
| (0,.25] | -0.437 | +0.906 | +0.226 | 13/20 |
| (.25,.5] | -1.908 | -0.565 | -0.671 | 3/20 |
| (.5,.75] | -0.620 | +0.723 | +0.147 | 9/20 |
| (.75,1) | -0.156 | +1.187 | +0.768 | 15/20 |

This separates two claims that must not be conflated:

1. **absolute within-bin correctness advantage:** MaxRL does not show a
   persistent advantage matching its persistent low-p0 advantage-mass
   reallocation;
2. **relative behavioral allocation after subtracting the whole-panel
   objective difference:** the evidence is mixed rather than a clean persistent
   low-p0 relocation.

Therefore the paper should not use the endpoint low-bin absolute contrast alone
to claim that relative behavioral allocation did not move.

The safer objective-intervention statement is:

> MaxRL persistently reallocates realized scalar advantage mass. A persistent
> matching absolute correctness advantage is not observed in the same bins;
> evidence about relative behavioral reallocation is more mixed.

Snapshot sign counts and trajectory means are descriptive summaries of two
training trajectories, not independent trials.

## Terminology

For writing, call the primary ledger quantity:

```text
realized scalar advantage mass
```

or simply:

```text
advantage mass
```

It is cumulative `sum |A|` over realized groups. It is not a gradient norm,
source-to-target influence estimate, or attributable behavioral effect.

Shared-parameter transfer/interference remains a compatible explanation, but
the current evidence does not identify it as a statistically estimated
mediator.

## Code and outputs

The reviewed scripts and deterministic outputs are tracked under:

```text
analyses/attrib_deadline_exploratory/
```

They read the canonical tracked CSVs directly and do not duplicate the source
data.

Run:

```bash
python -m analyses.attrib_deadline_exploratory.exposure_sensitivity
python -m analyses.attrib_deadline_exploratory.allocation_comparison
```

The supplied package was independently checked before integration:

- all five packaged input CSVs match the corresponding tracked canonical CSVs
  after normalizing line endings and removing one extra trailing blank line;
- both scripts run successfully under Python 3 with NumPy;
- regenerated CSV/JSON outputs match the supplied results semantically;
- the bootstrap weighted-fit implementation was independently compared against
  literal duplicated-row resampling with re-standardization over 400 fits,
  with maximum discrepancy about `1.5e-15`.

## Paper priority

These diagnostics strengthen uncertainty/sensitivity reporting but do not
replace the single-training-seed limitation.

Do not start new GPU experiments or a new attribution estimator for the
deadline paper merely because compute is available.
