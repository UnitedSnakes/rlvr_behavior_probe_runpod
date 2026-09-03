# Exposure-split pre-outcome decision checkpoint

Date: 2026-09-03

## Scope

This checkpoint is written before inspecting any exposed-vs-unexposed DeltaR / DeltaT / DeltaC split outcomes.

The purpose is to freeze how the canonical seed-42 fixed-panel exposure split will be checked and interpreted. The split is descriptive / quasi-experimental at best, not a randomized causal estimate.

## Why cutoff-specific balance is required

Continuous exposure-step correlation and balance at a binary snapshot cutoff are distinct properties. A small Pearson correlation can coexist with a substantial exposed-vs-unexposed baseline difference at a particular cutoff if ordering imbalance is non-monotone. Therefore the already-computed continuous diagnostics are necessary but not sufficient.

Before outcome deblinding, evaluate 25%, 45%, and 65% snapshot cutoffs within each frozen cross-fit p0 bin. Exposed means unique own-exposure ledger step < snapshot step; unexposed means >= snapshot step.

For each cutoff / direction / bin report exposed-minus-unexposed differences for these pre-exposure covariates:

1. opposite-cross-fit-half p0 reward probability;
2. opposite-cross-fit-half pi0 completion length;
3. canonical prompt token count.

For each covariate report raw mean difference and standardized mean difference (SMD) using the pooled sample standard deviation. If either group has fewer than 2 questions or pooled SD is zero, SMD is NA rather than fabricated.

No numeric balance pass/fail threshold is introduced post hoc. The result will be interpreted as balance only on the measured covariates, not as random exposure ordering. Unmeasured semantic structure, reasoning-step count, numerical structure, and other prompt features can still differ.

## Continuous pre-outcome diagnostics already observed

The measured continuous diagnostics showed no universal strong monotone relation between exposure step and the three measured covariates, but local exceptions must remain flagged.

Primary bins of interest:

- `(0,.25]`: A->B r(p0)=+0.136, r(pi0 length)=-0.223, r(prompt)=-0.032, KS D=0.114; B->A r(p0)=-0.128, r(pi0 length)=+0.111, r(prompt)=-0.061, KS D=0.088.
- `(.25,.5]`: A->B r(p0)=+0.069, r(pi0 length)=+0.022, r(prompt)=+0.202, KS D=0.126; B->A r(p0)=+0.061, r(pi0 length)=+0.043, r(prompt)=+0.067, KS D=0.083.

Predeclared local caveats:

- `(.75,1)`, A->B: r(p0)=-0.451 with n=20; both directions also have relatively large KS D (~0.21). Treat exposure-split results in this bin as low-confidence descriptive evidence.
- `(.5,.75]`: prompt-length ordering is non-negligible, especially B->A r(prompt)=+0.389 (A->B +0.252). Any exposed/unexposed outcome contrast in this bin requires cautious attribution.
- `p0=0`: A->B prompt-length ordering r(prompt)=-0.318; interpret cautiously.
- observed `p0=1`: one A-bin question only; do not interpret as a population support-boundary result.

These caveats remain in force regardless of later outcome values.

## 25% `(0,.25]` group size and rough Monte Carlo scale

The cross-fit bins contain 89 questions in A-bin/B-base and 86 in B-bin/A-base. The observed first-quartile exposure counts imply approximately:

- A->B: 24 exposed, 65 unexposed;
- B->A: 23 exposed, 63 unexposed.

This corrects a mistaken rough count of ~14 exposed questions that would come from using the wrong bin denominator.

Each fixed-panel snapshot probability and opposite-half baseline probability is estimated with K=16. Ignoring between-question heterogeneity and using worst-case Bernoulli p=0.5, the rollout-Monte-Carlo-only standard deviation for an exposed-minus-unexposed mean DeltaC difference is approximately:

- A->B: 4.22 percentage points (1 sigma), 8.44 pp (2 sigma);
- B->A: 4.31 percentage points (1 sigma), 8.61 pp (2 sigma).

This is not a formal uncertainty interval and does not include finite-question heterogeneity or assignment variability. It is only a pre-outcome narrative scale. Therefore differences below ~4 pp are too small to narrate; 4-8 pp are mixed / uncertain; >=8 pp are large descriptive gaps, not statistical significance.

## Frozen primary DeltaC contrast

The primary descriptive contrast is the difference, not a ratio:

`gap_C = DeltaC_unexposed - DeltaC_exposed`

The ratio `DeltaC_unexposed / DeltaC_exposed` may still be reported as a secondary reference only.

Before outcome deblinding, freeze:

- if DeltaC_exposed < +2 pp: `not_classifiable`;
- if gap_C >= +4 pp: `unexposed_higher`;
- if -4 pp < gap_C < +4 pp: `transfer_compatible`;
- if -8 pp < gap_C <= -4 pp: `mixed_or_uncertain`;
- if gap_C <= -8 pp: `own_exposure_candidate`.

Labels are deliberately cautious. `transfer_compatible` means no material exposed advantage at the predeclared ~4 pp narrative scale; it does not prove transfer causally dominates. `own_exposure_candidate` likewise means a large descriptive exposed advantage, not significance.

The earlier ratio thresholds (0.25 / 0.75) are demoted to secondary descriptive reference because the denominator is noisy and can become unstable when DeltaC_exposed is small.

## Outcome deblinding order

1. Run cutoff-specific pre-outcome balance at 25/45/65.
2. Inspect and record those results without reading split outcomes.
3. Only then run the exposed-vs-unexposed fixed-panel DeltaR / DeltaT / DeltaC split.
4. Apply the frozen difference-based DeltaC interpretation above.
5. Do not launch MaxRL or sequence-mask runs based solely on one noisy split cell.
