# Canonical GRPO exposure-split post-outcome result and paper-claim freeze

Date: 2026-09-03

## Status

This checkpoint is written **after** deblinding the canonical seed-42 exposed-vs-unexposed outcomes and after the direction-specific raw/adjusted robustness check.

It does not retroactively modify the 2026-09-01 preregistration or the two pre-outcome exposure checkpoints. The relevant provenance is:

1. `docs/superpowers/specs/2026-09-01-signal-allocation-analysis-prereg.md`
2. `docs/superpowers/checkpoints/2026-09-02-postrun-preoutcome-analysis-addendum.md`
3. `docs/superpowers/checkpoints/2026-09-03-exposure-split-preoutcome-decision.md`
4. `docs/superpowers/checkpoints/2026-09-03-cutoff-balance-observed-and-adjustment.md`
5. this post-outcome checkpoint

The exposure split, 25/45/65 cutoffs, primary `gap_C = DeltaC_unexposed - DeltaC_exposed`, interpretation scale, measured-covariate balance audit, and OLS adjustment model were all fixed before reading the split outcomes.

The run remains a single canonical training seed. The exposure order is not randomized, and the measured covariates do not exhaust possible ordering structure. Therefore these results are descriptive evidence about locality/transfer, not a causal treatment effect of own exposure.

## Canonical lineage

```text
branch at analysis implementation: codex/signal-ledger
analysis code commit: 386f300e36562ad78063fcfd4b5ed4137325fd9d
mode: canonical
scientific_use: true
pi0_lineage_id: f89fc90226a67a6a3c7374f9c13abadfcecda88f397ab812fa4130f1f425605b
max optimizer steps: 3736
G: 16
global optimizer batch: 32
generation batch: 32
world size: 2
fixed train panel: 256 GSM8K train questions
baseline: K=32 split into independent A/B halves
snapshot evaluation bank: independent K=16 C bank
```

Canonical model repository:

```text
HKReporter/rlvr-behavior-probe-grpo-canonical-seed42-2026-09-02
```

Corrected canonical pre-RL policy repository:

```text
HKReporter/rlvr-behavior-probe-pi0-corrected-canonical-2026-08-30
```

## Primary post-outcome finding

The strongest result from the canonical exposure analysis is:

```text
no stable own-exposure advantage
```

Across the frozen 25%, 45%, and 65% cutoffs, substantial correctness improvement appears among questions that have **not yet been directly sampled for training**. Direct own exposure is not a stable predictor of larger question-level correctness improvement.

This changes the interpretation of the earlier prompt-local caveat. The 2026-09-02 pre-outcome addendum said that prompt-local `S_i -> Delta p_i` attribution was exploratory because shared parameters make local attribution noisy and require a locality assumption. The observed exposure result goes further: the data are not merely too noisy to identify a strong prompt-local effect; they are **inconsistent with a strong locality account in which direct sampling should systematically produce larger subsequent gains on the sampled problem**.

This does **not** prove that all improvement is caused by cross-question transfer. Exposure timing is not randomized, measured-covariate adjustment is incomplete, and shared-parameter interference is inherent. The safe conclusion is that substantial cross-question transfer is compatible with, and strongly suggested by, the observed pattern, while a stable own-exposure advantage is not observed.

A safe paper sentence is:

> Across snapshots, correctness gains are not systematically larger for questions that have already generated direct training signal.

A slightly more interpretive but still acceptable sentence is:

> Direct question exposure is not a stable predictor of question-level correctness improvement. Large behavioral changes occur among questions not yet directly sampled for training, consistent with substantial cross-question transfer under shared parameters.

Do not write that transfer is causally proven to be the dominant channel.

## Covariate-adjusted symmetric DeltaC outcomes

All values below are percentage-point changes relative to the independent opposite-half baseline. `gap` is adjusted unexposed minus adjusted exposed.

| cutoff | p0 bin | exposed | unexposed | gap | frozen label |
|---:|---|---:|---:|---:|---|
| 25% | `0` | +2.89 | +2.10 | -0.78 | transfer_compatible |
| 25% | `(0,.25]` | +6.25 | +10.02 | +3.77 | transfer_compatible |
| 25% | `(.25,.5]` | +8.46 | +4.40 | -4.06 | mixed_or_uncertain |
| 25% | `(.5,.75]` | +3.25 | +1.47 | -1.78 | transfer_compatible |
| 25% | `(.75,1)` | -1.69 | +3.00 | +4.69 | not_classifiable |
| 45% | `0` | +1.45 | +3.08 | +1.64 | not_classifiable |
| 45% | `(0,.25]` | +7.42 | +12.80 | +5.38 | unexposed_higher |
| 45% | `(.25,.5]` | +7.57 | +7.47 | -0.11 | transfer_compatible |
| 45% | `(.5,.75]` | +2.87 | +7.02 | +4.15 | unexposed_higher |
| 45% | `(.75,1)` | +3.98 | -0.71 | -4.69 | mixed_or_uncertain |
| 65% | `0` | +6.55 | +3.36 | -3.19 | transfer_compatible |
| 65% | `(0,.25]` | +10.90 | +12.80 | +1.90 | transfer_compatible |
| 65% | `(.25,.5]` | +8.49 | +12.20 | +3.71 | transfer_compatible |
| 65% | `(.5,.75]` | +3.45 | +7.57 | +4.12 | unexposed_higher |
| 65% | `(.75,1)` | +1.78 | +2.10 | +0.33 | not_classifiable |

Across these 15 symmetric cells:

```text
transfer_compatible:  8
unexposed_higher:     3
mixed_or_uncertain:   2
not_classifiable:     2
own_exposure_candidate: 0
```

The absence of an `own_exposure_candidate` cell under the pre-frozen rule is the compact descriptive summary. It is not a significance test.

## Direction-specific robustness

The symmetric result is not being created by averaging two large opposite-direction effects, and the OLS adjustment is not generally manufacturing the sign.

### Low-p0 `(0,.25]`

25%:

```text
A->B raw gap +4.79 pp; adjusted +5.55 pp
B->A raw gap +2.31 pp; adjusted +2.00 pp
```

45%:

```text
A->B raw gap +8.25 pp; adjusted +7.85 pp
B->A raw gap +3.81 pp; adjusted +2.91 pp
```

65%:

```text
A->B raw gap +4.17 pp; adjusted +4.27 pp
B->A raw gap +0.33 pp; adjusted -0.46 pp
```

Thus the low-p0 symmetric gap is positive at all three frozen cutoffs. The 65% B->A direction is essentially null/slightly negative, so do not claim that every direction at every cutoff favors unexposed questions.

### Medium-p0 `(.25,.5]`

25%:

```text
A->B raw gap -3.54 pp; adjusted -1.40 pp
B->A raw gap -6.80 pp; adjusted -6.72 pp
```

45%:

```text
A->B raw gap -4.91 pp; adjusted -3.11 pp
B->A raw gap +4.06 pp; adjusted +2.90 pp
```

65%:

```text
A->B raw gap +0.40 pp; adjusted -0.17 pp
B->A raw gap +4.33 pp; adjusted +7.59 pp
```

The 25% medium bin has an own-exposure-like negative gap, especially B->A, but it does not persist: the directions disagree by 45%, and by 65% one direction has a large unexposed advantage. This is why the scientific conclusion is **no stable own-exposure advantage**, not "unexposed always wins."

## Reward/termination/correctness context

The fixed-panel trajectory separates three outcomes:

```text
R = terminated-and-correct reward
T = termination
C = correctness independent of termination
```

From pi0 to the 100% snapshot:

```text
DeltaR = +18.12 pp
DeltaT = +31.63 pp
DeltaC =  +7.29 pp
```

Termination acquisition accounts for much of the global reward movement, while correctness still improves nontrivially. The exposure result should therefore be reported for `DeltaC` rather than inferred from reward alone.

## Signal allocation is not behavioral-change allocation

The canonical ledger analysis separately measures realized training-signal allocation. This is conceptually distinct from the fixed-panel movement result.

The current mechanism chain is:

```text
objective
  -> realized training-signal allocation
  -> parameter update
  -> cross-question transfer / interference
  -> behavioral-change allocation
```

Equivalently, for question `i`,

```text
Delta log p_i ~= sum_t <grad_theta log p_i, Delta theta_t>
```

and each `Delta theta_t` is produced by many training questions, not only question `i`.

Therefore:

```text
training-signal allocation != eventual behavioral-change allocation
```

This distinction is now central to the paper and to the next objective intervention.

## Paper claim hierarchy frozen here

For the MATH-AI workshop draft, use this order:

1. **Main mechanism result:** RLVR-induced question-level correctness improvement is not localized to direct own training exposure; no stable own-exposure advantage is observed.
2. **Supporting measurement result:** realized GRPO signal allocation across difficulty need not have the same shape as correctness movement.
3. **Instrumentation result:** implementation choices can impose objective-external difficulty-dependent weighting. Historical truncation masking and sequence-level IS are supporting validation/appendix material, not the headline phenomenon. Canonical token-level IS has ESS/N approximately 0.998 and should not be presented as a canonical difficulty distortion.
4. **Behavioral decomposition:** `DeltaT`, `DeltaC`, and `DeltaR` move differently despite one binary terminated-correct reward.

Do not lead the paper with the historical implementation-stack story. Do not claim a representation mechanism has been directly measured. "Shared model structure," "shared-parameter transfer," or "cross-question transfer" are acceptable; a specific latent representation explanation remains future work.

## Reporting discipline

- Show all three frozen 25/45/65 cutoffs when making the low-p0 exposure claim; do not report only the visually largest 45% gap.
- Keep raw and adjusted direction-specific results available in supplement/robustness material.
- State that cutoff assignment is not randomized and adjustment covers only three measured pre-exposure covariates.
- Do not attach p-values or significance language to the pre-frozen descriptive labels.
- Do not combine fresh training seeds with seed42 and retroactively call the seed42-discovered exposure finding independently confirmed.

## Next objective intervention

The next planned scientific intervention is a matched GRPO-versus-MaxRL comparison. Its purpose is no longer merely to ask whether two objectives nominally weight difficulty differently. It will ask whether an objective-induced change in **realized signal allocation** propagates to **behavioral-change allocation**.

The detailed pre-MaxRL hypotheses and gates are frozen separately in:

`docs/superpowers/specs/2026-09-03-maxrl-objective-intervention-amendment.md`

No MaxRL training result has been observed at the time of this checkpoint.
