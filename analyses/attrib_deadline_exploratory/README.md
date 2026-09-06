# ATTRIB deadline post-outcome exploratory diagnostics

Date: 2026-09-06.

These analyses were designed **after the accepted seed42 primary results were
visible**. They are supplementary sensitivity/description analyses, not part of
the frozen pre-outcome protocol. They perform no new training, generation, or
gradient measurement.

## Source-of-truth inputs

The scripts read the tracked canonical exports directly rather than duplicating
them:

```text
analyses/canonical_exposure_split_adjusted/adjustment_input_rows.csv
analyses/canonical_exposure_split_adjusted/adjusted_symmetric.csv
analyses/canonical_snapshot_crossfit/aggregate_sanity.csv
analyses/canonical_maxrl_snapshot_crossfit/aggregate_sanity.csv
analyses/canonical_maxrl_grpo_objective_comparison/objective_comparison.csv
```

At the review commit, their Git blob SHAs were respectively:

```text
a3a4c25a669d64b4eeaeac136071a9d0ffb1bdd0
79a9f0bf949471c17e0e57bade36440b237292e8
837b1f27058fcb0b678e860c9b9b7bfdca3ccb08
9300bd09ffa0baca1c20862b0e073e6d10a7a4a1
a65b01eaee7b1fdbfa6ee175eace188b34c62e22
```

The externally supplied deadline-analysis package was checked against these
tracked inputs. After normalizing CRLF/LF and removing one extra trailing blank
line in each packaged CSV, the contents matched exactly.

## 1. Question-resampling exposure sensitivity

Run:

```bash
python -m analyses.attrib_deadline_exploratory.exposure_sensitivity
```

The script resamples the 256 question IDs with replacement 3,000 times using
seed `20260906`. One question multiplicity is shared across both cross-fit
directions and all three cutoffs. Each resample refits the frozen OLS adjustment
and evaluates adjusted group means at the resampled pooled covariate mean.

Before resampling, the code reproduces all 15 accepted symmetric point
estimates to `<1e-12` absolute error. An independent review also checked that
its weighted implementation is numerically equivalent to literally duplicating
resampled rows, re-standardizing covariates, and refitting OLS (maximum error
about `1.5e-15` over 400 checked fits).

Primary low-nonzero-bin descriptive sensitivity:

| cutoff | not-yet-exposed DeltaC | pointwise 2.5--97.5% range | U-E point estimate | U-E range |
|---:|---:|---:|---:|---:|
| 25% | +10.02 pp | [6.64, 13.30] | +3.77 pp | [-3.33, 10.60] |
| 45% | +12.80 pp | [7.55, 17.87] | +5.38 pp | [-1.41, 11.85] |
| 65% | +12.80 pp | [7.68, 18.13] | +1.90 pp | [-4.26, 8.09] |

Leave-one-question-out minima for the not-yet-exposed low-bin gain are
`9.49`, `11.78`, and `11.48` pp at the three cutoffs.

Interpretation boundary: these are pointwise question-composition sensitivity
ranges conditional on the observed training trajectory, frozen memberships,
and observed response banks. They do **not** estimate between-training-seed
variance, do not fully characterize repeated response-sampling uncertainty on
the fixed panel, have no multiplicity correction, and are not causal or
equivalence-test intervals.

## 2. Absolute versus whole-panel-centered correctness contrasts

Run:

```bash
python -m analyses.attrib_deadline_exploratory.allocation_comparison
```

For each bin and snapshot, the existing objective comparison contains

```text
d_b(t) = DeltaC_MaxRL,b(t) - DeltaC_GRPO,b(t).
```

The supplementary centered contrast is

```text
q_b(t) = d_b(t) - [C_MaxRL,panel(t) - C_GRPO,panel(t)].
```

Because both objectives share the same initial aggregate baseline, the bracket
is also the whole-panel difference in correctness change. This is an algebraic
descriptive centering, not a causal intervention that fixes overall model
quality.

At 100%, the whole-panel MaxRL-minus-GRPO correctness difference is `-1.343 pp`:

| p0 bin | absolute d_b | centered q_b | mean q_b over 20 snapshots | q_b > 0 |
|---|---:|---:|---:|---:|
| 0 | -4.712 | -3.369 | -0.162 | 9/20 |
| (0,.25] | -0.437 | +0.906 | +0.226 | 13/20 |
| (.25,.5] | -1.908 | -0.565 | -0.671 | 3/20 |
| (.5,.75] | -0.620 | +0.723 | +0.147 | 9/20 |
| (.75,1) | -0.156 | +1.187 | +0.768 | 15/20 |

This distinguishes two claims: the within-bin **absolute** correctness
advantage is not persistent, while evidence about **relative behavioral
allocation** is mixed. Snapshot counts are descriptive trajectory summaries,
not independent trials.

## Outputs

```text
exposure_question_bootstrap.csv
exposure_question_bootstrap_summary.json
centered_correctness.csv
centered_correctness_summary.json
```

Scientific interpretation and historical corrections are recorded in:

```text
docs/superpowers/checkpoints/2026-09-06-attrib-deadline-postoutcome-sensitivity.md
```
