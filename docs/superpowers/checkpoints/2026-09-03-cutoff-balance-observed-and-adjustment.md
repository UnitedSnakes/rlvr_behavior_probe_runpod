# Cutoff-balance observed result and pre-outcome adjustment freeze

Date: 2026-09-03

## Outcome-blinding status

This checkpoint is written after inspecting only the pre-outcome cutoff-specific balance table and before inspecting any exposed-vs-unexposed DeltaR / DeltaT / DeltaC split outcomes.

The cutoff-specific balance check showed that exposure ordering is not adequately balanced at every cutoff on the three measured covariates. Therefore the raw exposed-vs-unexposed outcome contrast will remain a secondary descriptive result; the primary descriptive contrast will be covariate-adjusted using a model frozen here before outcome deblinding.

This remains descriptive / quasi-experimental at best. The measured covariates do not exhaust possible ordering structure. In particular, semantic problem type, latent reasoning-step count, numerical structure, and other unmeasured features can remain imbalanced.

## Observed cutoff-specific balance

The three pre-exposure covariates are opposite-cross-fit-half p0 reward probability, opposite-cross-fit-half pi0 completion length, and canonical prompt token count. Differences are exposed minus unexposed. SMD uses the pooled sample standard deviation.

### 25% snapshot

Primary hard bin `(0,.25]`:

- A-bin/B-base: nE/nU = 24/65; dp0 = -4.64 pp (SMD -0.370); d(pi0 length) = +108.8 tokens (SMD +0.586); d(prompt tokens) = +6.0 (SMD +0.288).
- B-bin/A-base: nE/nU = 23/63; dp0 = +3.39 pp (SMD +0.245); d(pi0 length) = -27.4 tokens (SMD -0.148); d(prompt tokens) = +8.7 (SMD +0.447).

This cell is not sufficiently balanced to interpret a raw exposed-vs-unexposed outcome gap as a quasi-random contrast.

Primary medium-hard bin `(.25,.5]`:

- A-bin/B-base: nE/nU = 20/37; dp0 = -3.61 pp (SMD -0.195); d(pi0 length) = +8.8 (SMD +0.044); d(prompt tokens) = +0.8 (SMD +0.048).
- B-bin/A-base: nE/nU = 17/42; dp0 = -1.69 pp (SMD -0.088); d(pi0 length) = -7.7 (SMD -0.035); d(prompt tokens) = +1.0 (SMD +0.060).

This is the cleanest primary cutoff cell on measured covariates, but it is still not randomized.

Other 25% cells contain notable local imbalance, including p0=0 A->B prompt SMD +0.933, `(.5,.75]` B->A prompt SMD -0.888, and `(.75,1)` A->B pi0-length SMD +0.766.

### 45% snapshot

The primary `(0,.25]` bin remains moderately imbalanced:

- A->B: p0 SMD -0.186, pi0-length +0.484, prompt -0.142.
- B->A: p0 +0.247, pi0-length +0.008, prompt -0.048.

`(.25,.5]` remains relatively cleaner but not exact: A->B p0 SMD -0.275, B->A pi0-length -0.293. Other bins retain larger local imbalances, including `.75,1` A->B p0 SMD +0.855 and `.5,.75]` B->A prompt SMD -0.710.

### 65% snapshot

The primary `(0,.25]` bin is nearly balanced in A->B but not B->A:

- A->B: p0 SMD -0.029, pi0-length +0.116, prompt -0.101.
- B->A: p0 +0.313, pi0-length -0.352, prompt -0.046.

`(.25,.5]` has a substantial B->A p0 imbalance (SMD -0.428). High-p0 cells remain small and unstable, including `.75,1` B->A prompt SMD -1.279.

## Interpretation of the balance result

The continuous exposure-step correlations previously showed no universal strong monotone ordering on the three measured covariates. The cutoff analysis demonstrates why that is not equivalent to balance at the binary 25/45/65 exposure splits: finite-sample and non-monotone imbalance is present at several actual cutoffs.

Therefore:

1. do not call the exposure order random;
2. do not call the full set of cells balanced;
3. do not use the raw exposed-vs-unexposed contrast as the primary transfer estimate in imbalanced cells;
4. keep the raw contrast visible as a secondary descriptive check;
5. use a pre-frozen covariate-adjusted contrast as the primary descriptive contrast.

## Frozen covariate-adjusted estimator

For each snapshot percent x cross-fit direction x frozen p0 bin, fit the following ordinary least-squares model separately for X in {R, T, C}:

`DeltaX_i = alpha + tau * E_i + beta1 * z(p0_i) + beta2 * z(L0_i) + beta3 * z(P_i) + epsilon_i`

where:

- E_i = 1 iff the question's unique own-exposure ledger step is strictly less than the snapshot step;
- p0_i is the opposite-cross-fit-half baseline reward probability;
- L0_i is the opposite-cross-fit-half pi0 completion length;
- P_i is canonical prompt token count;
- each covariate is centered and standardized within the cell using all questions in that cell;
- a covariate with zero within-cell variance is dropped rather than assigned an arbitrary coefficient;
- no interaction terms are added;
- no semantic/problem-type features are added after outcome deblinding;
- no covariate is selected or removed based on outcome fit.

Adjusted unexposed and exposed means are the model predictions at the pooled within-cell covariate mean. Because the covariates are centered, these are alpha and alpha + tau respectively. The adjusted primary gap is therefore:

`gap_C_adjusted = adjusted_DeltaC_unexposed - adjusted_DeltaC_exposed = -tau_C`.

Direction-specific adjusted means are computed first. The symmetric estimate is the equal-weight average of the A-bin/B-base and B-bin/A-base adjusted direction means, preserving the previously frozen cross-fit symmetrization rule.

No analytic p-value, bootstrap interval, or significance threshold is introduced in this analysis pass.

## Frozen adjusted DeltaC interpretation

Apply the already-frozen descriptive difference rule to the adjusted symmetric DeltaC means, not the raw means:

- if adjusted DeltaC_exposed < +2 pp: `not_classifiable`;
- if adjusted gap_C >= +4 pp: `unexposed_higher`;
- if -4 pp < adjusted gap_C < +4 pp: `transfer_compatible`;
- if -8 pp < adjusted gap_C <= -4 pp: `mixed_or_uncertain`;
- if adjusted gap_C <= -8 pp: `own_exposure_candidate`.

The ratio adjusted_DeltaC_unexposed / adjusted_DeltaC_exposed may be reported only as a secondary reference when the adjusted exposed mean is at least +2 pp.

These are descriptive labels, not significance claims. The pre-outcome ~4 pp / ~8 pp narrative scales remain based only on a worst-case K=16 rollout-Monte-Carlo calculation for the 25% `(0,.25]` cell and do not include between-question heterogeneity or assignment uncertainty.

## Confidence hierarchy before deblinding

- 25% `(.25,.5]`: best measured balance; raw and adjusted contrasts may both be shown, with adjusted primary.
- `(0,.25]` at 25/45/65: adjusted contrast primary; raw contrast is secondary because measured imbalance is non-negligible.
- `(.5,.75]`, p0=0, and especially `(.75,1)`: lower-confidence descriptive cells due measured imbalance and/or small n. Adjustment does not erase unmeasured confounding or small-sample instability.
- p0=1: do not interpret as a population support-boundary result.

## Deblinding order from this point

1. Implement and test the frozen adjusted estimator without reading real split outcomes.
2. Verify code on synthetic fixtures and full CPU CI.
3. Only then run the canonical 25/45/65 outcome split.
4. Report raw and adjusted DeltaR / DeltaT / DeltaC, but use adjusted DeltaC gap for the primary descriptive classification.
5. Do not launch MaxRL or sequence-mask experiments based solely on one split cell.