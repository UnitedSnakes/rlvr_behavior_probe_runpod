# Clipping diagnostics (POST HOC)

**POST HOC — not pre-registered.** Specified on 2026-11-15, after `RESULTS.md` and `analysis.json` were known. Descriptive numbers only; nothing here is a causal estimate, and no pre-registered sentence or result file is changed. Scope: the three box pairs (seeds 43–45), GRPO and MaxRL arms, existing group ledgers and per-step pre-clip `grad_norm` logs; no retraining. All numbers: `clip_diagnostics.json`. Code: `camera_ready/analysis/clip_diagnostics.py` (run 2026-11-15T21:19:27Z, HEAD `4090df9`).

## Definitions used

- Clip coefficient of optimizer step t: c_t = min(1, 1.0 / grad_norm_t); a step is *clipped* iff grad_norm_t > 1.0 (strict). A group generated at ledger step s is weighted by c_{s+1}. Check: the ledger's mean reward at step s equals the logged reward at optimizer step s+1 at every step of every run (max |diff| = 0.0e+00); offsets 0 and 2 do not match.
- Live group: 0 < K < 16. Identical to `analysis.json`'s `advantage != 0` in every group of every run; non-live groups carry zero |A|.
- Bins: the frozen A40 cross-fit bins of `analysis.json`. Bin mass S_b = average over the two directions of the within-bin mean, over panel questions, of cumulative |A| (groups generated before step T) — the `analysis.json` convention. Clip-weighted mass replaces |A| by |A|·c.
- Cell counts n = number of distinct panel groups with non-zero mass in the cell, per arm, over both directions (each panel question is sampled at one step, so groups = steps; per-direction counts in the JSON). A cell is marked † if the smaller arm has n < 30 and ‡ if n < 10 (thresholds chosen here; see ambiguity 7).

## Reproduction check: PASS

Raw and clip-weighted mass, mass ratios, share ratios and totals in `analysis.json` (4 eval steps × 5 bins × 3 pairs) were recomputed twice — with the existing pipeline functions and with this module's own code (using the pipeline's `c = min(1, 1/(g + 1e-6))`) — together with the clipping summaries and live-step shares. Largest relative difference over all values: 6.3e-16 (seed45, clip_weighted_mass__own_code); tolerance 1e-09. Switching to this note's c (no 1e-6) changes any clip-weighted ratio by at most 3.4e-07 (relative).

## 1. Bin shares (step 3736) — values for seed 43 / 44 / 45

Share ratio V1 = (S_b^M / total^M) / (S_b^G / total^G), total over all training groups before T (the `analysis.json` `share_ratio` convention; for clip-weighted shares the totals are clip-weighted). c̄_b = mass-weighted mean clip coefficient, pooled form S_b^clip / S_b^raw.

| Bin | Raw share ratio M/G | Clip-wtd share ratio M/G | c̄_b GRPO | c̄_b MaxRL | n GRPO | n MaxRL |
|---|---|---|---|---|---|---|
| 0 † | 2.43 / 2.37 / 1.90 | 1.45 / 1.33 / 1.30 | 0.849 / 0.870 / 0.840 | 0.423 / 0.406 / 0.475 | 26 / 28 / 34 | 30 / 30 / 32 |
| (0,.25] | 1.38 / 1.47 / 1.46 | 1.15 / 1.28 / 1.18 | 0.796 / 0.788 / 0.805 | 0.557 / 0.567 / 0.536 | 104 / 106 / 108 | 104 / 108 / 107 |
| (.25,.5] | 0.75 / 0.81 / 0.89 | 0.94 / 0.96 / 1.12 | 0.706 / 0.717 / 0.720 | 0.739 / 0.703 / 0.750 | 91 / 91 / 90 | 90 / 90 / 92 |
| (.5,.75] | 0.55 / 0.51 / 0.58 | 0.74 / 0.66 / 0.75 | 0.673 / 0.710 / 0.682 | 0.766 / 0.767 / 0.731 | 75 / 75 / 75 | 76 / 75 / 80 |
| (.75,1) † | 0.50 / 0.51 / 0.45 | 0.67 / 0.86 / 0.70 | 0.672 / 0.629 / 0.627 | 0.759 / 0.878 / 0.813 | 19 / 23 / 24 | 25 / 25 / 25 |

Clip-weighted share ratio V1 at each evaluated step (seed 43 / 44 / 45):

| Bin | T = 934 | T = 1681 | T = 2428 | T = 3736 |
|---|---|---|---|---|
| 0 | 0.81 / 1.32 / 1.31 | 1.13 / 1.20 / 1.27 | 1.28 / 1.30 / 1.37 | 1.45 / 1.33 / 1.30 |
| (0,.25] | 1.24 / 1.37 / 1.20 | 1.17 / 1.31 / 1.21 | 1.18 / 1.27 / 1.23 | 1.15 / 1.28 / 1.18 |
| (.25,.5] | 1.05 / 1.02 / 1.09 | 1.00 / 0.92 / 1.16 | 0.93 / 0.93 / 1.14 | 0.94 / 0.96 / 1.12 |
| (.5,.75] | 0.88 / 0.79 / 0.80 | 0.75 / 0.76 / 0.82 | 0.76 / 0.69 / 0.82 | 0.74 / 0.66 / 0.75 |
| (.75,1) | 0.75 / 1.06 / 0.88 | 0.75 / 1.00 / 0.92 | 0.76 / 0.91 / 0.78 | 0.67 / 0.86 / 0.70 |

Raw share ratios, shares themselves, mass ratios, the V2 share variant and the per-direction c̄ variant at every T are in the JSON (`pairs.*.shares`).

## 2. Split by live groups per step (step 3736) — seed 43 / 44 / 45

Type is each arm's own step type (ambiguity 4). Fraction = cell raw mass / bin raw mass, pooled form, per arm. Ratios are M/G of the cell's symmetric mass.

| Bin | Cell | Frac of bin raw mass GRPO | Frac MaxRL | Raw ratio M/G | Clip-wtd ratio M/G | n GRPO | n MaxRL |
|---|---|---|---|---|---|---|---|
| 0 | live1 ‡ | 0.09 / 0.09 / 0.13 | 0.19 / 0.06 / 0.15 | 5.72 / 1.63 / 2.49 | 2.62 / 0.73 / 1.38 | 3 / 3 / 4 | 6 / 2 / 4 |
| 0 | live2 † | 0.91 / 0.91 / 0.87 | 0.81 / 0.94 / 0.85 | 2.48 / 2.83 / 2.12 | 1.24 / 1.33 / 1.20 | 23 / 25 / 30 | 24 / 28 / 28 |
| 0 | live1_clipped ‡ | 0.00 / 0.00 / 0.04 | 0.19 / 0.06 / 0.15 | – / – / 7.66 | – / – / 4.63 | 0 / 0 / 1 | 6 / 2 / 4 |
| 0 | live1_unclipped ‡ | 0.09 / 0.09 / 0.09 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 3 / 3 / 3 | 0 / 0 / 0 |
| 0 | live2_clipped † | 0.80 / 0.87 / 0.67 | 0.81 / 0.94 / 0.85 | 2.82 / 2.95 / 2.75 | 1.45 / 1.40 / 1.66 | 20 / 24 / 23 | 24 / 28 / 28 |
| 0 | live2_unclipped ‡ | 0.11 / 0.04 / 0.20 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 3 / 1 / 7 | 0 / 0 / 0 |
| (0,.25] | live1 ‡ | 0.09 / 0.12 / 0.07 | 0.15 / 0.10 / 0.09 | 2.74 / 1.35 / 1.98 | 1.76 / 1.03 / 1.34 | 9 / 13 / 9 | 14 / 10 / 10 |
| (0,.25] | live2 | 0.91 / 0.88 / 0.93 | 0.85 / 0.90 / 0.91 | 1.47 / 1.73 / 1.65 | 1.03 / 1.25 / 1.09 | 95 / 93 / 99 | 90 / 98 / 97 |
| (0,.25] | live1_clipped ‡ | 0.00 / 0.05 / 0.02 | 0.13 / 0.07 / 0.07 | – / 2.20 / 5.07 | – / 1.58 / 3.52 | 0 / 5 / 2 | 11 / 6 / 6 |
| (0,.25] | live1_unclipped ‡ | 0.09 / 0.07 / 0.05 | 0.02 / 0.03 / 0.02 | 0.34 / 0.67 / 0.63 | 0.34 / 0.67 / 0.63 | 9 / 8 / 7 | 3 / 4 / 4 |
| (0,.25] | live2_clipped | 0.87 / 0.87 / 0.84 | 0.84 / 0.89 / 0.91 | 1.53 / 1.73 / 1.82 | 1.07 / 1.24 / 1.24 | 90 / 92 / 88 | 87 / 96 / 97 |
| (0,.25] | live2_unclipped ‡ | 0.04 / 0.01 / 0.09 | 0.01 / 0.01 / 0.00 | 0.34 / 2.13 / 0.00 | 0.34 / 2.13 / 0.00 | 5 / 1 / 11 | 3 / 2 / 0 |
| (.25,.5] | live1 ‡ | 0.09 / 0.08 / 0.06 | 0.11 / 0.03 / 0.10 | 1.10 / 0.39 / 1.80 | 1.13 / 0.41 / 1.93 | 8 / 8 / 6 | 12 / 2 / 9 |
| (.25,.5] | live2 | 0.91 / 0.92 / 0.94 | 0.89 / 0.97 / 0.90 | 0.84 / 0.97 / 0.96 | 0.87 / 0.97 / 0.98 | 83 / 83 / 84 | 78 / 88 / 83 |
| (.25,.5] | live1_clipped ‡ | 0.03 / 0.03 / 0.01 | 0.05 / 0.03 / 0.01 | 1.35 / 1.07 / 0.80 | 1.48 / 1.24 / 1.16 | 3 / 3 / 2 | 4 / 1 / 1 |
| (.25,.5] | live1_unclipped ‡ | 0.06 / 0.05 / 0.05 | 0.06 / 0.00 / 0.09 | 0.96 / 0.05 / 2.07 | 0.96 / 0.05 / 2.07 | 5 / 5 / 4 | 8 / 1 / 8 |
| (.25,.5] | live2_clipped | 0.91 / 0.92 / 0.93 | 0.83 / 0.91 / 0.84 | 0.79 / 0.91 / 0.91 | 0.80 / 0.88 / 0.91 | 83 / 83 / 83 | 69 / 77 / 72 |
| (.25,.5] | live2_unclipped ‡ | 0.00 / 0.00 / 0.01 | 0.05 / 0.06 / 0.06 | – / – / 8.80 | – / – / 8.80 | 0 / 0 / 1 | 9 / 11 / 11 |
| (.5,.75] | live1 ‡ | 0.12 / 0.16 / 0.06 | 0.11 / 0.07 / 0.05 | 0.62 / 0.26 / 0.60 | 0.66 / 0.27 / 0.73 | 10 / 12 / 5 | 13 / 8 / 6 |
| (.5,.75] | live2 | 0.88 / 0.84 / 0.94 | 0.89 / 0.93 / 0.95 | 0.63 / 0.65 / 0.67 | 0.72 / 0.73 / 0.71 | 65 / 63 / 70 | 63 / 67 / 74 |
| (.5,.75] | live1_clipped ‡ | 0.04 / 0.05 / 0.05 | 0.03 / 0.00 / 0.02 | 0.49 / 0.00 / 0.27 | 0.60 / 0.00 / 0.35 | 5 / 5 / 4 | 2 / 0 / 1 |
| (.5,.75] | live1_unclipped ‡ | 0.08 / 0.11 / 0.01 | 0.08 / 0.07 / 0.03 | 0.68 / 0.38 / 1.96 | 0.68 / 0.38 / 1.96 | 5 / 7 / 1 | 11 / 8 / 5 |
| (.5,.75] | live2_clipped | 0.88 / 0.84 / 0.92 | 0.73 / 0.77 / 0.78 | 0.52 / 0.54 / 0.57 | 0.55 / 0.57 / 0.56 | 65 / 63 / 69 | 49 / 50 / 53 |
| (.5,.75] | live2_unclipped ‡ | 0.00 / 0.00 / 0.02 | 0.16 / 0.15 / 0.16 | – / – / 4.42 | – / – / 4.42 | 0 / 0 / 1 | 14 / 17 / 21 |
| (.75,1) | live1 ‡ | 0.22 / 0.17 / 0.05 | 0.23 / 0.15 / 0.09 | 0.59 / 0.51 / 0.87 | 0.70 / 0.57 / 0.97 | 5 / 4 / 1 | 5 / 4 / 3 |
| (.75,1) | live2 † | 0.78 / 0.83 / 0.95 | 0.77 / 0.85 / 0.91 | 0.56 / 0.60 / 0.49 | 0.62 / 0.90 / 0.64 | 14 / 19 / 23 | 20 / 21 / 22 |
| (.75,1) | live1_clipped ‡ | 0.10 / 0.13 / 0.05 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | 3 / 3 / 1 | 0 / 0 / 0 |
| (.75,1) | live1_unclipped ‡ | 0.11 / 0.04 / 0.00 | 0.23 / 0.15 / 0.09 | 1.12 / 2.21 / – | 1.12 / 2.21 / – | 2 / 1 / 0 | 5 / 4 / 3 |
| (.75,1) | live2_clipped † | 0.78 / 0.83 / 0.95 | 0.68 / 0.62 / 0.73 | 0.49 / 0.44 / 0.39 | 0.51 / 0.61 / 0.48 | 14 / 19 / 23 | 16 / 13 / 15 |
| (.75,1) | live2_unclipped ‡ | 0.00 / 0.00 / 0.00 | 0.09 / 0.24 / 0.18 | – / – / – | – / – / – | 0 / 0 / 0 | 4 / 8 / 7 |

A ratio of 0.00 means MaxRL has no mass in the cell; – means GRPO has none. The mark after the bin or cell name refers to the smallest n among the three seeds. Values at the other evaluated steps: JSON `pairs.*.step_type`.

## 3. Clip coefficient by the step's smallest positive K (all 3,736 steps per run, three seeds pooled)

For steps with ≥ 1 live group the smallest positive K equals the smallest live K (checked in every run). Per-seed tables: JSON `pairs.*.low_K`.

**1 live group in the step**

| min K | n GRPO | n MaxRL | mean c GRPO | mean c MaxRL | clipped GRPO | clipped MaxRL |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 123 | 118 | 0.975 | 0.368 | 13.0 % | 100.0 % |
| 2 | 117 | 121 | 0.987 | 0.503 | 12.8 % | 100.0 % |
| 3 | 93 | 113 | 0.978 | 0.616 | 22.6 % | 100.0 % |
| 4 | 103 | 85 | 0.983 | 0.707 | 16.5 % | 100.0 % |
| 5 | 107 | 98 | 0.957 | 0.806 | 35.5 % | 91.8 % |
| 6 | 114 | 130 | 0.966 | 0.885 | 26.3 % | 56.9 % |
| 7 | 121 | 132 | 0.949 | 0.920 | 32.2 % | 34.8 % |
| 8 | 160 | 138 | 0.950 | 0.949 | 26.2 % | 26.8 % |
| 9 | 162 | 145 | 0.930 | 0.959 | 38.3 % | 28.3 % |
| 10 | 164 | 165 | 0.916 | 0.982 | 37.8 % | 17.0 % |
| 11 | 168 | 146 | 0.894 | 0.988 | 45.2 % | 11.0 % |
| 12 | 157 | 177 | 0.908 | 0.998 | 40.8 % | 3.4 % |
| 13 | 189 | 197 | 0.862 | 1.000 | 54.0 % | 0.5 % |
| 14 | 178 | 159 | 0.836 | 1.000 | 61.2 % | 0.0 % |
| 15 | 143 | 141 | 0.796 | 1.000 | 72.0 % | 0.0 % |
| all | 2099 | 2065 | 0.918 | 0.871 | 37.9 % | 37.6 % |

**2 live groups in the step**

| min K | n GRPO | n MaxRL | mean c GRPO | mean c MaxRL | clipped GRPO | clipped MaxRL |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1015 | 1000 | 0.864 | 0.315 | 76.4 % | 100.0 % |
| 2 | 828 | 849 | 0.833 | 0.420 | 88.9 % | 100.0 % |
| 3 | 782 | 744 | 0.815 | 0.504 | 93.5 % | 100.0 % |
| 4 | 796 | 798 | 0.784 | 0.572 | 97.0 % | 100.0 % |
| 5 | 696 | 724 | 0.762 | 0.629 | 99.1 % | 100.0 % |
| 6 | 713 | 744 | 0.724 | 0.690 | 100.0 % | 100.0 % |
| 7 | 637 | 738 | 0.703 | 0.740 | 100.0 % | 100.0 % |
| 8 | 664 | 673 | 0.677 | 0.791 | 100.0 % | 99.3 % |
| 9 | 669 | 648 | 0.650 | 0.849 | 100.0 % | 95.8 % |
| 10 | 635 | 597 | 0.612 | 0.898 | 100.0 % | 83.2 % |
| 11 | 530 | 541 | 0.582 | 0.943 | 100.0 % | 62.1 % |
| 12 | 439 | 400 | 0.550 | 0.978 | 100.0 % | 31.8 % |
| 13 | 323 | 346 | 0.517 | 0.990 | 100.0 % | 15.0 % |
| 14 | 192 | 180 | 0.485 | 0.999 | 99.5 % | 1.7 % |
| 15 | 49 | 42 | 0.441 | 1.000 | 100.0 % | 0.0 % |
| all | 8968 | 9024 | 0.717 | 0.676 | 95.4 % | 87.6 % |

Steps with no live group (no mass): GRPO 141 (mean c 1.000), MaxRL 119 (mean c 1.000).

## Cells with few groups / steps (all evaluated steps; every cell's counts are in the JSON)

- Section 1 (bin × step × seed): 32 of 60 cells n < 30, 9 n < 10; by bin at T = 3736: 0 n = 26–32, (0,.25] n = 104–107, (.25,.5] n = 90–90, (.5,.75] n = 75–75, (.75,1) n = 19–24 (smaller arm).
- Section 2, live1: 60 of 60 cells n < 30, 58 n < 10.
- Section 2, live2: 35 of 60 cells n < 30, 13 n < 10.
- Section 2, live1_clipped: 60 of 60 cells n < 30, 60 n < 10.
- Section 2, live1_unclipped: 60 of 60 cells n < 30, 60 n < 10.
- Section 2, live2_clipped: 38 of 60 cells n < 30, 15 n < 10.
- Section 2, live2_unclipped: 60 of 60 cells n < 30, 60 n < 10.
- Section 3: pooled, 0 of 64 arm × live × K cells n < 30; per seed, 11 of 192 (smallest per-seed n = 7).

## Definitional ambiguities (not resolved here)

1. Share denominator. V1 (shown in the note) divides the symmetric per-question bin mass by the arm's total over *all* training groups before T, as `analysis.json`'s `share_ratio` does; V2 uses the fraction of the *panel's* mass, averaged over directions (sums to 1 over the six bins). Largest |V2 − V1| in any share ratio: raw 0.212, clip-weighted 0.130. Both in the JSON.
2. Cross-fit averaging of derived ratios (c̄_b, fraction of bin mass). Pooled form = ratio of the two symmetric masses (shown; gives clip ratio = raw ratio × c̄^M/c̄^G exactly); alternative = mean over directions of the per-direction ratios. Largest difference: c̄ 0.089, fraction 0.373. Both in the JSON.
3. Cumulative vs interval mass. All T values are cumulative from step 0 (as in `analysis.json`); mass accrued only between evaluated steps is not computed.
4. Step type is arm-specific: the same question can sit in a 1-live step in one arm and a 2-live step in the other, so the M/G ratio within a type compares partly different question sets. A joint classification (by both arms' types, or by GRPO's type only) is not computed.
5. Clipped / not-clipped sub-split in section 2 was added so that the stated prediction (1-live and clipped steps) can be read; the request listed only the 1-vs-2 split.
6. Section 3 uses all 3,736 steps of each run (whole training set, no bins); restricting to steps that consumed a panel question would give far fewer steps. Pooling over seeds is done by pooling steps; per-seed tables are in the JSON.
7. Small-cell thresholds (10 and 30) are this note's choice; every cell reports its counts.
8. The pipeline's `c = min(1, 1/(g + 1e-6))` (the form torch's clip_grad_norm_ applies) differs from this note's c only at the 1e-6 level; the strict `> 1.0` rule excludes the steps whose logged norm is exactly 1.0 (count per run in `checks`). Logged norms are coarsely quantized (values such as 1.3515625).
9. grad_norm = 0 makes 1.0/grad_norm undefined. It occurs at exactly the steps with no live group (33–52 per run; checked both ways), which carry no mass; c is set to 1 there (the limit of min(1, 1/g), and the factor torch applies). It affects only the 'no live group' row of section 3, whose mean c is 1 by this rule.

