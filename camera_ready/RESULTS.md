# Camera-ready replication results

Status: results computed 2026-10-03 ~16:40 UTC by the pre-registered pipeline (`camera_ready/analysis/analyze.py`, `bridge.py`) on the GPU box with code commit `45d6130`; independent fact-check: see `FACTCHECK.md`. All numbers below come from `camera_ready/results/analysis.json` (SHA-256 `2f1fa340…4419`) and `bridge.json` (`e816c7f0…b082`). Units: percentage points (pp) for contrasts, ratios for advantage mass.

**Sample.** Main inference uses the three box-trained pairs (seeds 43, 44, 45; 2 × RTX 4080 SUPER per run). The A40 seed-42 pair (the submitted discovery run) is shown in its own row and never pooled into the main sample. Endpoint K for the primary cells is 16 (the 64-response version requires all 64 endpoint responses in every pair; seed 45's third extra batch could not be scheduled before the pre-registered cut-off, so K = 64 was not available for all pairs).

**What can be said, using only the pre-registered sentences.**

- Primary cells (step 3736; bins 0 and (0,.25]; C and R; d and q): all eight are **"unresolved at this number of runs"** — every two-sided 95 % t-interval across the three pairs contains 0 and reaches beyond ±3 pp. No interval lies entirely above or entirely below 0, so the paper's Test-2 sentence does not have to change on the rule's terms. The eight intervals carry no multiplicity adjustment.
- Reweighting is **realized** in every new pair: the MaxRL/GRPO advantage-mass ratio is above 1 in bins 0 and (0,.25] at step 3736, and also at all four EVAL_STEPS.
- The pre-exposure gain is **replicated** under the pre-registered rule: the adjusted not-yet-exposed correctness gain in (0,.25] is positive at all three cutoffs in every new seed.
- The random-schedule exposure contrast (U − E) is **uninformative** at three seeds: the per-seed values differ in sign and the across-seed intervals are wide.
- Bridge: **PASS** — box evaluations of the same frozen weights are compatible with the stored A40 evaluations within the pre-registered margins. This says nothing about whether training on the box behaves like training on A40.
- Supplementary, mixing hardware (A40 discovery pair pooled with the three box pairs, K = 16, n = 4, shown only because the bridge passed): reported in §2 next to the primary table. It is not the pre-registered main inference.

## 1. Bridge (evaluation compatibility)

Verdict: **PASS** over 9 checkpoint(s). Full tables: `BRIDGE_REPORT.md`.

## Validity of the three box pairs

All six runs trained to completion on the first attempt (no OOM, no restart), from the same checkout (`9814757`) and environment, both arms of each pair concurrently. All three pairs are valid under PREREG_RUNS §6.

| Seed | GPUs (GRPO / MaxRL) | Training h (GRPO / MaxRL) | Rows / groups / steps | Max advantage error (GRPO / MaxRL) | Token IS ESS/N (GRPO / MaxRL) | Grad norm logged |
|---|---|---|---|---|---|---|
| 43 | 0,1 / 2,3 | 17.08 / 17.21 | 119,552 / 7,472 / 3,736 each | 2.24e-7 / 1.59e-7 | 0.99808 / 0.99808 | 3,736 / 3,736 |
| 44 | 2,3 / 0,1 | 17.10 / 17.25 | 119,552 / 7,472 / 3,736 each | 2.24e-7 / 1.59e-7 | 0.99807 / 0.99808 | 3,736 / 3,736 |
| 45 | 0,1 / 2,3 | 17.11 / 17.21 | 119,552 / 7,472 / 3,736 each | 2.24e-7 / 1.59e-7 | 0.99809 / 0.99807 | 3,736 / 3,736 |

No non-finite ledger fields in any run. Every panel question was sampled in every run (no never-sampled panel question). K = 16 protocol evaluations exist for all four EVAL_STEPS of every run (24 evaluations, 34–36 min each). Extra endpoint batches: complete for seed 43, partial for seeds 44–45 at the pre-registered cut-off (see §5); they are not used for the primary cells.

Whole-panel rates at step 3736 (%, K = 16; R / T / C): seed 43 GRPO 51.78 / 75.22 / 56.76, MaxRL 51.54 / 74.46 / 56.98; seed 44 GRPO 51.17 / 74.41 / 56.25, MaxRL 51.05 / 74.68 / 56.47; seed 45 GRPO 51.54 / 74.58 / 56.45, MaxRL 52.71 / 74.90 / 57.81; seed 42 (A40) GRPO 53.03 / 77.25 / 57.74, MaxRL 51.54 / 74.73 / 56.40.

Compatibility alerts: none. The smoke comparison against the A40 seed-42 ledger (first 40 steps) raised no > 3 SE difference; the bridge passed.

## 2. Main sample: primary cells

Main sample: 3 box-trained pair(s) (seed43, seed44, seed45). Endpoint K for the primary cells: 16 responses per question (rule: 64 only if every pair has all 64). Bridge verdict: PASS; baselines from the A40 π0 banks; bins: the paper's frozen A40 cross-fit bins.

### Primary cells (step 3736; MaxRL − GRPO, pp)

Two-sided 95 % t-intervals across pairs (df = n − 1; none for n < 3); no multiplicity adjustment across the eight cells. In brackets after each pair's value: its within-pair conditional sampling interval (responses resampled within question), which is not seed uncertainty.

| Cell | seed43 | seed44 | seed45 | Mean | 95 % CI | One-sided 95 % upper | Wording (δ = 3 pp) | Discovery (A40 seed 42, K=16) |
|---|---|---|---|---:|---|---:|---|---|
| d, C, bin 0 | +0.97 [-2.12, +4.17] | +2.00 [-1.06, +5.05] | +0.48 [-2.46, +3.38] | +1.15 | [-0.77, +3.08] | +2.46 | unresolved at this number of runs | -4.71 (below the new pairs' range) |
| q, C, bin 0 | +0.75 [-2.28, +3.73] | +1.78 [-1.19, +4.80] | -0.88 [-3.71, +1.98] | +0.55 | [-2.79, +3.89] | +2.82 | unresolved at this number of runs | -3.37 (below the new pairs' range) |
| d, C, bin (0,.25] | +0.81 [-2.10, +3.71] | +1.23 [-1.77, +4.17] | +3.43 [+0.49, +6.25] | +1.82 | [-1.66, +5.31] | +4.19 | unresolved at this number of runs | -0.44 (below the new pairs' range) |
| q, C, bin (0,.25] | +0.59 [-1.51, +2.65] | +1.01 [-1.11, +3.09] | +2.06 [-0.04, +4.17] | +1.22 | [-0.66, +3.10] | +2.49 | unresolved at this number of runs | +0.91 (inside the new pairs' range) |
| d, R, bin 0 | -1.12 [-3.87, +1.58] | +1.37 [-1.29, +4.05] | +0.39 [-2.28, +2.91] | +0.21 | [-2.90, +3.32] | +2.32 | unresolved at this number of runs | -3.13 (below the new pairs' range) |
| q, R, bin 0 | -0.87 [-3.62, +1.83] | +1.49 [-1.16, +4.22] | -0.78 [-3.38, +1.89] | -0.05 | [-3.38, +3.27] | +2.20 | unresolved at this number of runs | -1.64 (below the new pairs' range) |
| d, R, bin (0,.25] | +0.13 [-2.70, +3.04] | +0.55 [-2.27, +3.39] | +3.15 [+0.39, +6.03] | +1.28 | [-2.79, +5.34] | +4.03 | unresolved at this number of runs | -0.62 (below the new pairs' range) |
| q, R, bin (0,.25] | +0.37 [-1.70, +2.48] | +0.67 [-1.38, +2.68] | +1.98 [-0.07, +4.09] | +1.01 | [-1.11, +3.13] | +2.45 | unresolved at this number of runs | +0.87 (inside the new pairs' range) |

### Advantage mass (MaxRL/GRPO), per pair

| Pair | Step | 0 ratio; share ratio | (0,.25] ratio; share ratio | (.25,.5] ratio; share ratio | (.5,.75] ratio; share ratio | (.75,1) ratio; share ratio |
|---|---|---|---|---|---|---|
| seed43 | 934 | 1.96; 1.53 | 1.83; 1.42 | 1.02; 0.79 | 0.90; 0.70 | 0.91; 0.71 |
| seed43 | 1681 | 2.29; 1.92 | 1.66; 1.39 | 0.90; 0.75 | 0.71; 0.59 | 0.81; 0.67 |
| seed43 | 2428 | 2.45; 2.11 | 1.61; 1.38 | 0.84; 0.73 | 0.66; 0.57 | 0.68; 0.59 |
| seed43 | 3736 | 2.78; 2.43 | 1.58; 1.38 | 0.86; 0.75 | 0.63; 0.55 | 0.57; 0.50 |
| seed44 | 934 | 2.45; 1.99 | 1.92; 1.56 | 1.23; 1.00 | 0.76; 0.62 | 0.76; 0.62 |
| seed44 | 1681 | 2.04; 1.72 | 1.70; 1.43 | 1.00; 0.84 | 0.66; 0.56 | 0.68; 0.57 |
| seed44 | 2428 | 2.66; 2.29 | 1.73; 1.49 | 0.98; 0.84 | 0.60; 0.51 | 0.59; 0.51 |
| seed44 | 3736 | 2.72; 2.37 | 1.69; 1.47 | 0.93; 0.81 | 0.58; 0.51 | 0.59; 0.51 |
| seed45 | 934 | 2.56; 2.04 | 1.93; 1.53 | 1.16; 0.92 | 0.88; 0.70 | 0.70; 0.56 |
| seed45 | 1681 | 2.24; 1.87 | 1.83; 1.52 | 1.12; 0.94 | 0.83; 0.69 | 0.73; 0.61 |
| seed45 | 2428 | 2.18; 1.86 | 1.69; 1.44 | 1.05; 0.90 | 0.75; 0.64 | 0.60; 0.51 |
| seed45 | 3736 | 2.17; 1.90 | 1.67; 1.46 | 1.01; 0.89 | 0.66; 0.58 | 0.51; 0.45 |
| seed 42 (A40) | 934 | 2.72; 2.14 | 2.04; 1.61 | 1.13; 0.89 | 0.71; 0.56 | 0.41; 0.32 |
| seed 42 (A40) | 1681 | 2.62; 2.19 | 1.94; 1.62 | 1.08; 0.90 | 0.64; 0.54 | 0.48; 0.40 |
| seed 42 (A40) | 2428 | 2.49; 2.13 | 1.82; 1.55 | 1.04; 0.89 | 0.61; 0.52 | 0.50; 0.42 |
| seed 42 (A40) | 3736 | 2.21; 1.93 | 1.72; 1.50 | 0.99; 0.86 | 0.59; 0.52 | 0.46; 0.40 |

Reweighting realized (ratio > 1 in bins 0 and (0,.25] at step 3736 in every new pair): **True**; also at all four EVAL_STEPS: True.

### Pre-exposure gain, GRPO arm, bin (0,.25] (frozen adjustment model), pp

| Pair | Cutoff | ΔC exposed | ΔC not yet exposed | U − E | n exposed / not yet (A; B) |
|---|---|---:|---:|---:|---|
| seed43 | 934 | +4.01 | +10.01 | +6.00 | 26/63; 25/61 |
| seed43 | 1681 | +8.46 | +12.39 | +3.94 | 47/42; 43/43 |
| seed43 | 2428 | +8.69 | +12.50 | +3.81 | 66/23; 62/24 |
| seed44 | 934 | +9.47 | +5.86 | -3.62 | 19/70; 25/61 |
| seed44 | 1681 | +11.54 | +6.65 | -4.89 | 39/50; 37/49 |
| seed44 | 2428 | +9.84 | +6.00 | -3.84 | 55/34; 55/31 |
| seed45 | 934 | +11.32 | +6.22 | -5.11 | 26/63; 26/60 |
| seed45 | 1681 | +8.04 | +7.13 | -0.92 | 43/46; 38/48 |
| seed45 | 2428 | +9.01 | +8.59 | -0.42 | 60/29; 55/31 |
| seed 42 (A40) | 934 | +6.25 | +10.02 | +3.77 | 24/65; 23/63 |
| seed 42 (A40) | 1681 | +7.42 | +12.80 | +5.38 | 49/40; 44/42 |
| seed 42 (A40) | 2428 | +10.90 | +12.80 | +1.90 | 62/27; 55/31 |

Pre-exposure gain replicated (ΔC_U > 0 at all three cutoffs in every new seed): **True**.

Random-schedule exposure contrast (supplementary; adjusted U − E in (0,.25]; across-seed mean and 95 % t-interval; estimand: the average effect, under random scheduling, of a question having entered training by the cutoff instead of a random other question taking its place — not pure self-influence, not a share of the total gain):

| Cutoff | Per seed | Mean | 95 % CI |
|---|---|---:|---|
| 934 | +6.00, -3.62, -5.11 | -0.91 | [-15.88, +14.07] |
| 1681 | +3.94, -4.89, -0.92 | -0.62 | [-11.61, +10.36] |
| 2428 | +3.81, -3.84, -0.42 | -0.15 | [-9.67, +9.37] |

### Pre-registered sentences for the primary cells

- d, C, bin 0: mean +1.15 pp, 95 % CI [-0.77, +3.08], one-sided 95 % upper bound +2.46 pp (n = 3) — **unresolved at this number of runs**.
- q, C, bin 0: mean +0.55 pp, 95 % CI [-2.79, +3.89], one-sided 95 % upper bound +2.82 pp (n = 3) — **unresolved at this number of runs**.
- d, C, bin (0,.25]: mean +1.82 pp, 95 % CI [-1.66, +5.31], one-sided 95 % upper bound +4.19 pp (n = 3) — **unresolved at this number of runs**.
- q, C, bin (0,.25]: mean +1.22 pp, 95 % CI [-0.66, +3.10], one-sided 95 % upper bound +2.49 pp (n = 3) — **unresolved at this number of runs**.
- d, R, bin 0: mean +0.21 pp, 95 % CI [-2.90, +3.32], one-sided 95 % upper bound +2.32 pp (n = 3) — **unresolved at this number of runs**.
- q, R, bin 0: mean -0.05 pp, 95 % CI [-3.38, +3.27], one-sided 95 % upper bound +2.20 pp (n = 3) — **unresolved at this number of runs**.
- d, R, bin (0,.25]: mean +1.28 pp, 95 % CI [-2.79, +5.34], one-sided 95 % upper bound +4.03 pp (n = 3) — **unresolved at this number of runs**.
- q, R, bin (0,.25]: mean +1.01 pp, 95 % CI [-1.11, +3.13], one-sided 95 % upper bound +2.45 pp (n = 3) — **unresolved at this number of runs**.

The eight intervals carry no multiplicity adjustment.

### Supplementary mixed estimate (bridge PASS only): A40 discovery pair pooled with the box pairs, K = 16

Labelled as mixing the discovery run with replications across hardware. Not the pre-registered main inference; no wording rule is applied; no multiplicity adjustment.

| Cell | Values (box pairs…, seed 42 A40) | n | Mean | 95 % CI |
|---|---|---:|---:|---|
| d, C, bin 0 | +0.97, +2.00, +0.48, -4.71 | 4 | -0.31 | [-5.09, +4.46] |
| q, C, bin 0 | +0.75, +1.78, -0.88, -3.37 | 4 | -0.43 | [-4.00, +3.14] |
| d, C, bin (0,.25] | +0.81, +1.23, +3.43, -0.44 | 4 | +1.26 | [-1.30, +3.82] |
| q, C, bin (0,.25] | +0.59, +1.01, +2.06, +0.91 | 4 | +1.14 | [+0.13, +2.16] |
| d, R, bin 0 | -1.12, +1.37, +0.39, -3.13 | 4 | -0.62 | [-3.74, +2.50] |
| q, R, bin 0 | -0.87, +1.49, -0.78, -1.64 | 4 | -0.45 | [-2.60, +1.70] |
| d, R, bin (0,.25] | +0.13, +0.55, +3.15, -0.62 | 4 | +0.80 | [-1.80, +3.41] |
| q, R, bin (0,.25] | +0.37, +0.67, +1.98, +0.87 | 4 | +0.98 | [-0.14, +2.09] |

## 3. Secondary results

### Secondary: d_b and q_b at all EVAL_STEPS (pp; across-pair mean and 95 % t-interval; no wording rule applies)

**C**

| Step | Stat | 0 | (0,.25] | (.25,.5] | (.5,.75] | (.75,1) |
|---|---|---|---|---|---|---|
| 934 | d | +0.27 [-1.37, +1.90] (+0.28, +0.92, -0.40) | -0.74 [-1.17, -0.31] (-0.58, -0.71, -0.93) | -0.55 [-2.09, +0.98] (-0.08, -0.32, -1.25) | -0.41 [-1.46, +0.63] (-0.90, -0.17, -0.17) | +1.44 [-1.81, +4.70] (+2.87, +1.16, +0.30) |
| 934 | q | +0.59 [-0.39, +1.58] (+0.45, +1.04, +0.28) | -0.42 [-0.84, +0.01] (-0.41, -0.59, -0.25) | -0.23 [-1.04, +0.59] (+0.09, -0.20, -0.57) | -0.09 [-1.64, +1.46] (-0.73, -0.05, +0.52) | +1.77 [-1.00, +4.53] (+3.04, +1.28, +0.98) |
| 1681 | d | +0.95 [-1.44, +3.33] (+0.64, +2.02, +0.18) | -0.88 [-5.96, +4.20] (+0.04, -3.22, +0.54) | +0.74 [-1.75, +3.24] (+1.71, +0.81, -0.29) | -0.04 [-0.87, +0.80] (-0.29, -0.16, +0.34) | -0.58 [-3.59, +2.42] (-1.46, +0.80, -1.09) |
| 1681 | q | +1.00 [-2.49, +4.48] (+0.33, +2.61, +0.06) | -0.83 [-4.81, +3.15] (-0.28, -2.64, +0.42) | +0.79 [-1.81, +3.39] (+1.39, +1.40, -0.42) | +0.01 [-1.34, +1.37] (-0.61, +0.42, +0.22) | -0.54 [-4.72, +3.65] (-1.77, +1.38, -1.22) |
| 2428 | d | +0.46 [-1.09, +2.01] (-0.14, +0.40, +1.11) | +0.10 [-1.53, +1.74] (-0.62, +0.26, +0.67) | +0.62 [-3.56, +4.81] (+2.30, +0.64, -1.07) | -0.02 [-3.61, +3.56] (+1.51, -0.23, -1.35) | -0.69 [-7.32, +5.94] (+2.39, -2.21, -2.25) |
| 2428 | q | +0.27 [-2.73, +3.27] (-0.97, +0.33, +1.45) | -0.08 [-3.19, +3.03] (-1.45, +0.18, +1.01) | +0.44 [-2.31, +3.18] (+1.47, +0.57, -0.73) | -0.21 [-2.32, +1.90] (+0.68, -0.31, -1.01) | -0.88 [-6.14, +4.39] (+1.56, -2.29, -1.90) |
| 3736 | d | +1.15 [-0.77, +3.08] (+0.97, +2.00, +0.48) | +1.82 [-1.66, +5.31] (+0.81, +1.23, +3.43) | -1.21 [-5.61, +3.20] (-2.23, -2.23, +0.84) | +0.98 [-1.48, +3.43] (+2.08, +0.18, +0.67) | -1.35 [-5.19, +2.48] (-1.63, +0.31, -2.74) |
| 3736 | q | +0.55 [-2.79, +3.89] (+0.75, +1.78, -0.88) | +1.22 [-0.66, +3.10] (+0.59, +1.01, +2.06) | -1.81 [-4.57, +0.95] (-2.45, -2.45, -0.53) | +0.37 [-2.93, +3.68] (+1.86, -0.04, -0.70) | -1.95 [-7.17, +3.27] (-1.85, +0.09, -4.11) |

**R**

| Step | Stat | 0 | (0,.25] | (.25,.5] | (.5,.75] | (.75,1) |
|---|---|---|---|---|---|---|
| 934 | d | -0.22 [-3.15, +2.72] (-0.59, -1.17, +1.11) | +0.16 [-0.27, +0.59] (-0.02, +0.18, +0.32) | -0.56 [-3.57, +2.45] (+0.18, -1.96, +0.10) | -0.98 [-2.87, +0.91] (-1.86, -0.52, -0.57) | +2.74 [+1.62, +3.87] (+2.56, +3.26, +2.41) |
| 934 | q | -0.12 [-2.16, +1.92] (-0.34, -0.80, +0.79) | +0.26 [-0.42, +0.94] (+0.22, +0.55, +0.00) | -0.46 [-3.02, +2.10] (+0.43, -1.59, -0.22) | -0.89 [-2.70, +0.93] (-1.62, -0.16, -0.88) | +2.84 [+0.94, +4.74] (+2.80, +3.62, +2.09) |
| 1681 | d | +1.36 [-0.52, +3.25] (+0.73, +2.20, +1.16) | -0.65 [-3.68, +2.39] (-0.50, -1.93, +0.49) | -0.22 [-3.45, +3.02] (+0.64, +0.43, -1.72) | -0.21 [-4.01, +3.59] (-1.97, +0.73, +0.62) | +0.87 [-5.23, +6.97] (-1.96, +2.43, +2.14) |
| 1681 | q | +1.40 [-0.10, +2.91] (+1.22, +2.08, +0.92) | -0.60 [-3.74, +2.53] (-0.01, -2.05, +0.25) | -0.18 [-4.15, +3.80] (+1.13, +0.31, -1.96) | -0.17 [-3.02, +2.68] (-1.49, +0.61, +0.38) | +0.91 [-4.24, +6.06] (-1.47, +2.31, +1.89) |
| 2428 | d | +0.42 [-2.23, +3.08] (+0.35, +1.53, -0.61) | -0.65 [-1.36, +0.06] (-0.94, -0.64, -0.37) | -0.57 [-4.41, +3.28] (+1.16, -1.03, -1.83) | -0.19 [-0.61, +0.23] (-0.00, -0.23, -0.33) | -0.97 [-6.79, +4.84] (+1.73, -2.22, -2.43) |
| 2428 | q | +0.84 [-1.64, +3.32] (+0.23, +1.99, +0.29) | -0.23 [-2.22, +1.75] (-1.06, -0.17, +0.54) | -0.15 [-2.75, +2.45] (+1.04, -0.57, -0.93) | +0.23 [-0.64, +1.09] (-0.12, +0.23, +0.57) | -0.56 [-5.22, +4.10] (+1.60, -1.76, -1.52) |
| 3736 | d | +0.21 [-2.90, +3.32] (-1.12, +1.37, +0.39) | +1.28 [-2.79, +5.34] (+0.13, +0.55, +3.15) | -1.43 [-4.89, +2.03] (-2.38, -2.08, +0.17) | +0.85 [-1.95, +3.64] (+2.14, +0.11, +0.28) | -0.48 [-1.27, +0.31] (-0.52, -0.15, -0.78) |
| 3736 | q | -0.05 [-3.38, +3.27] (-0.87, +1.49, -0.78) | +1.01 [-1.11, +3.13] (+0.37, +0.67, +1.98) | -1.70 [-3.22, -0.18] (-2.14, -1.96, -1.00) | +0.58 [-3.56, +4.71] (+2.39, +0.23, -0.89) | -0.75 [-3.36, +1.85] (-0.27, -0.03, -1.95) |

**T**

| Step | Stat | 0 | (0,.25] | (.25,.5] | (.5,.75] | (.75,1) |
|---|---|---|---|---|---|---|
| 934 | d | -0.00 [-4.93, +4.93] (+2.29, -1.03, -1.26) | -0.32 [-3.23, +2.60] (+0.44, +0.28, -1.67) | -0.23 [-6.43, +5.96] (+2.56, -2.23, -1.03) | -1.32 [-2.82, +0.19] (-1.79, -0.64, -1.53) | +1.67 [-0.53, +3.88] (+1.91, +0.69, +2.42) |
| 934 | q | +0.32 [-2.22, +2.85] (+1.48, -0.42, -0.11) | +0.00 [-1.93, +1.93] (-0.37, +0.89, -0.52) | +0.08 [-4.11, +4.28] (+1.76, -1.62, +0.12) | -1.00 [-4.47, +2.46] (-2.60, -0.03, -0.38) | +1.99 [-1.41, +5.39] (+1.10, +1.30, +3.57) |
| 1681 | d | +1.24 [-2.94, +5.43] (-0.66, +2.55, +1.83) | -0.90 [-3.00, +1.21] (-1.77, -0.84, -0.08) | +0.39 [-3.74, +4.52] (+1.94, +0.59, -1.37) | +0.35 [-3.48, +4.19] (-1.41, +1.46, +1.01) | +0.71 [-5.20, +6.63] (-1.96, +1.48, +2.62) |
| 1681 | q | +1.15 [-1.26, +3.55] (+0.05, +1.90, +1.49) | -1.00 [-2.34, +0.35] (-1.06, -1.50, -0.42) | +0.29 [-5.17, +5.76] (+2.65, -0.07, -1.71) | +0.26 [-1.81, +2.32] (-0.70, +0.80, +0.67) | +0.62 [-3.78, +5.01] (-1.25, +0.82, +2.27) |
| 2428 | d | -0.28 [-5.02, +4.45] (-2.05, +1.74, -0.55) | -1.78 [-4.19, +0.63] (-2.68, -1.90, -0.76) | -2.02 [-3.94, -0.10] (-1.49, -2.91, -1.67) | -0.28 [-3.51, +2.95] (-1.52, -0.39, +1.07) | -0.07 [-1.91, +1.77] (+0.76, -0.32, -0.65) |
| 2428 | q | +0.90 [-3.45, +5.24] (-0.22, +2.91, -0.01) | -0.60 [-1.44, +0.24] (-0.85, -0.73, -0.22) | -0.84 [-3.50, +1.82] (+0.35, -1.73, -1.13) | +0.90 [-0.73, +2.53] (+0.31, +0.78, +1.61) | +1.11 [-2.30, +4.52] (+2.60, +0.85, -0.11) |
| 3736 | d | +1.25 [-3.64, +6.13] (-1.02, +2.39, +2.38) | +0.27 [-4.70, +5.23] (-1.89, +2.06, +0.63) | -1.42 [-4.54, +1.69] (-0.48, -2.85, -0.94) | -0.11 [-0.85, +0.63] (+0.12, +0.01, -0.45) | +0.43 [-3.98, +4.84] (+1.59, -1.61, +1.32) |
| 3736 | q | +1.30 [-2.08, +4.68] (-0.27, +2.12, +2.06) | +0.32 [-3.31, +3.95] (-1.13, +1.79, +0.31) | -1.37 [-5.58, +2.85] (+0.27, -3.11, -1.26) | -0.05 [-2.13, +2.03] (+0.87, -0.26, -0.76) | +0.49 [-4.87, +5.85] (+2.34, -1.88, +1.00) |

### Whole-panel rates (%), per pair and arm (K = 16 protocol evaluation)

| Pair | Arm | Step | R | T | C |
|---|---|---|---:|---:|---:|
| seed43 | GRPO | 934 | 49.05 | 68.09 | 55.91 |
| seed43 | GRPO | 1681 | 51.56 | 73.80 | 56.98 |
| seed43 | GRPO | 2428 | 51.68 | 75.44 | 56.84 |
| seed43 | GRPO | 3736 | 51.78 | 75.22 | 56.76 |
| seed43 | MaxRL | 934 | 48.80 | 68.90 | 55.74 |
| seed43 | MaxRL | 1681 | 51.07 | 73.10 | 57.30 |
| seed43 | MaxRL | 2428 | 51.81 | 73.61 | 57.67 |
| seed43 | MaxRL | 3736 | 51.54 | 74.46 | 56.98 |
| seed44 | GRPO | 934 | 47.51 | 66.82 | 54.49 |
| seed44 | GRPO | 1681 | 50.76 | 72.83 | 56.27 |
| seed44 | GRPO | 2428 | 51.59 | 75.29 | 56.88 |
| seed44 | GRPO | 3736 | 51.17 | 74.41 | 56.25 |
| seed44 | MaxRL | 934 | 47.14 | 66.21 | 54.37 |
| seed44 | MaxRL | 1681 | 50.88 | 73.49 | 55.69 |
| seed44 | MaxRL | 2428 | 51.12 | 74.12 | 56.96 |
| seed44 | MaxRL | 3736 | 51.05 | 74.68 | 56.47 |
| seed45 | GRPO | 934 | 47.63 | 67.82 | 55.40 |
| seed45 | GRPO | 1681 | 50.95 | 73.46 | 56.13 |
| seed45 | GRPO | 2428 | 52.25 | 74.85 | 57.28 |
| seed45 | GRPO | 3736 | 51.54 | 74.58 | 56.45 |
| seed45 | MaxRL | 934 | 47.95 | 66.67 | 54.71 |
| seed45 | MaxRL | 1681 | 51.20 | 73.80 | 56.25 |
| seed45 | MaxRL | 2428 | 51.34 | 74.32 | 56.93 |
| seed45 | MaxRL | 3736 | 52.71 | 74.90 | 57.81 |
| seed 42 (A40) | GRPO | 934 | 49.00 | 68.33 | 55.62 |
| seed 42 (A40) | GRPO | 1681 | 51.95 | 75.42 | 57.13 |
| seed 42 (A40) | GRPO | 2428 | 54.13 | 76.88 | 58.50 |
| seed 42 (A40) | GRPO | 3736 | 53.03 | 77.25 | 57.74 |
| seed 42 (A40) | MaxRL | 934 | 46.46 | 65.67 | 54.39 |
| seed 42 (A40) | MaxRL | 1681 | 50.29 | 71.92 | 56.05 |
| seed 42 (A40) | MaxRL | 2428 | 52.29 | 74.54 | 57.23 |
| seed 42 (A40) | MaxRL | 3736 | 51.54 | 74.73 | 56.40 |

### Clipping diagnostics (log proxy; clipping rescales the whole batch gradient, relative weights within a batch are untouched)

| Pair | Arm | Steps logged | Share clipped | Mean coef. | Median | 5th pct |
|---|---|---:|---:|---:|---:|---:|
| seed43 | grpo | 3736 | 83.8 % | 0.755 | 0.740 | 0.492 |
| seed43 | maxrl | 3736 | 77.8 % | 0.716 | 0.723 | 0.305 |
| seed44 | grpo | 3736 | 83.7 % | 0.760 | 0.753 | 0.492 |
| seed44 | maxrl | 3736 | 77.8 % | 0.715 | 0.731 | 0.296 |
| seed45 | grpo | 3736 | 83.5 % | 0.759 | 0.746 | 0.492 |
| seed45 | maxrl | 3736 | 77.5 % | 0.717 | 0.731 | 0.308 |

Clip-weighted |A| mass ratio MaxRL/GRPO (each group's |A| × the clip coefficient of the step that consumed it; log proxy, not a measure of parameter contribution):

| Pair | Step | 0 | (0,.25] | (.25,.5] | (.5,.75] | (.75,1) |
|---|---|---:|---:|---:|---:|---:|
| seed43 | 934 | 0.79 | 1.21 | 1.02 | 0.86 | 0.73 |
| seed43 | 1681 | 1.10 | 1.14 | 0.97 | 0.73 | 0.73 |
| seed43 | 2428 | 1.23 | 1.14 | 0.90 | 0.74 | 0.73 |
| seed43 | 3736 | 1.39 | 1.11 | 0.90 | 0.71 | 0.64 |
| seed44 | 934 | 1.26 | 1.30 | 0.97 | 0.75 | 1.00 |
| seed44 | 1681 | 1.14 | 1.25 | 0.88 | 0.72 | 0.95 |
| seed44 | 2428 | 1.24 | 1.21 | 0.89 | 0.66 | 0.86 |
| seed44 | 3736 | 1.27 | 1.22 | 0.91 | 0.63 | 0.82 |
| seed45 | 934 | 1.25 | 1.15 | 1.05 | 0.77 | 0.84 |
| seed45 | 1681 | 1.22 | 1.15 | 1.11 | 0.78 | 0.88 |
| seed45 | 2428 | 1.30 | 1.17 | 1.09 | 0.78 | 0.74 |
| seed45 | 3736 | 1.23 | 1.11 | 1.06 | 0.71 | 0.66 |

### Discovery pair: A40 evaluation vs its checkpoints re-evaluated on the box (sensitivity row), C, pp

| Step | Stat | Source | 0 | (0,.25] | (.25,.5] | (.5,.75] | (.75,1) |
|---|---|---|---:|---:|---:|---:|---:|
| 934 | d | A40 | +0.04 | -2.80 | -1.84 | +0.90 | -0.64 |
| 934 | d | box | -2.57 | +0.64 | -2.40 | -3.67 | +2.11 |
| 934 | q | A40 | +1.26 | -1.58 | -0.62 | +2.12 | +0.58 |
| 934 | q | box | -1.28 | +1.93 | -1.10 | -2.37 | +3.40 |
| 1681 | d | A40 | -1.82 | +0.20 | -2.39 | -1.30 | -1.12 |
| 1681 | d | box | +2.70 | -0.44 | +0.47 | +1.97 | +1.64 |
| 1681 | q | A40 | -0.75 | +1.27 | -1.32 | -0.22 | -0.04 |
| 1681 | q | box | +1.82 | -1.32 | -0.41 | +1.09 | +0.76 |
| 2428 | d | A40 | -1.42 | -1.71 | -1.02 | -0.90 | -0.79 |
| 2428 | d | box | -0.23 | -1.93 | +0.80 | +0.51 | +0.16 |
| 2428 | q | A40 | -0.15 | -0.44 | +0.24 | +0.37 | +0.48 |
| 2428 | q | box | +0.13 | -1.56 | +1.17 | +0.88 | +0.52 |
| 3736 | d | A40 | -4.71 | -0.44 | -1.91 | -0.62 | -0.16 |
| 3736 | d | box | -0.16 | -0.54 | -1.47 | -2.03 | -0.51 |
| 3736 | q | A40 | -3.37 | +0.91 | -0.56 | +0.72 | +1.19 |
| 3736 | q | box | +0.84 | +0.46 | -0.46 | -1.03 | +0.49 |

## 4. Seed-42 discovery run: post-outcome analyses (F)

### F. Seed-42 discovery run, post-outcome descriptive analyses (A40)

Labelled post-outcome: these were specified in `PREREG_RUNS.md` §F before any new-seed outcome existed, after the seed-42 results were known. Intervals are conditional sampling intervals (responses resampled within question, 3,000 draws), not training-seed variability.

**F1/F2 — d_b and q_b for C (pp), with conditional 95 % sampling intervals**

| Step | Stat | 0 | (0,.25] | (.25,.5] | (.5,.75] | (.75,1) |
|---|---|---:|---:|---:|---:|---:|
| 934 | d | +0.04 [-3.22, +3.10] | -2.80 [-5.69, +0.21] | -1.84 [-4.98, +1.50] | +0.90 [-1.82, +3.64] | -0.64 [-4.00, +2.85] |
| 934 | q | +1.26 [-1.88, +4.22] | -1.58 [-3.61, +0.57] | -0.62 [-3.17, +1.97] | +2.12 [-0.35, +4.50] | +0.58 [-2.84, +4.06] |
| 1681 | d | -1.82 [-4.77, +1.09] | +0.20 [-2.75, +3.09] | -2.39 [-5.58, +0.90] | -1.30 [-3.93, +1.24] | -1.12 [-4.50, +2.26] |
| 1681 | q | -0.75 [-3.54, +2.16] | +1.27 [-0.87, +3.32] | -1.32 [-3.96, +1.31] | -0.22 [-2.52, +2.05] | -0.04 [-3.50, +3.36] |
| 2428 | d | -1.42 [-4.72, +1.94] | -1.71 [-4.68, +1.28] | -1.02 [-4.21, +2.24] | -0.90 [-3.50, +1.69] | -0.79 [-4.05, +2.42] |
| 2428 | q | -0.15 [-3.31, +3.04] | -0.44 [-2.64, +1.78] | +0.24 [-2.35, +2.90] | +0.37 [-2.02, +2.76] | +0.48 [-2.81, +3.76] |
| 3736 | d | -4.71 [-7.87, -1.67] | -0.44 [-3.32, +2.62] | -1.91 [-5.09, +1.40] | -0.62 [-3.29, +1.91] | -0.16 [-3.51, +3.36] |
| 3736 | q | -3.37 [-6.25, -0.44] | +0.91 [-1.09, +3.06] | -0.56 [-3.20, +2.10] | +0.72 [-1.71, +3.05] | +1.19 [-2.23, +4.62] |

**F1/F2 — d_b and q_b for R (pp), with conditional 95 % sampling intervals**

| Step | Stat | 0 | (0,.25] | (.25,.5] | (.5,.75] | (.75,1) |
|---|---|---:|---:|---:|---:|---:|
| 934 | d | +0.97 [-1.89, +3.54] | -3.62 [-6.37, -0.82] | -4.11 [-7.56, -0.47] | -1.75 [-4.90, +1.42] | -1.43 [-5.49, +2.95] |
| 934 | q | +3.51 [+0.71, +6.17] | -1.08 [-3.05, +1.07] | -1.57 [-4.21, +1.18] | +0.79 [-1.90, +3.42] | +1.11 [-2.85, +5.30] |
| 1681 | d | -2.09 [-4.73, +0.57] | -0.47 [-3.30, +2.13] | -2.33 [-5.70, +1.17] | -2.48 [-5.52, +0.45] | -2.08 [-5.88, +1.76] |
| 1681 | q | -0.43 [-3.03, +2.28] | +1.19 [-0.94, +3.19] | -0.67 [-3.28, +2.01] | -0.82 [-3.28, +1.63] | -0.42 [-4.10, +3.48] |
| 2428 | d | -2.42 [-5.42, +0.62] | -2.18 [-5.09, +0.76] | -0.99 [-4.27, +2.40] | -1.75 [-4.68, +1.17] | -2.08 [-6.13, +1.91] |
| 2428 | q | -0.59 [-3.50, +2.42] | -0.35 [-2.50, +1.82] | +0.84 [-1.85, +3.52] | +0.08 [-2.51, +2.53] | -0.25 [-4.22, +3.46] |
| 3736 | d | -3.13 [-5.94, -0.30] | -0.62 [-3.34, +2.42] | -1.20 [-4.54, +2.13] | -2.25 [-5.19, +0.56] | -1.60 [-5.44, +2.40] |
| 3736 | q | -1.64 [-4.29, +1.10] | +0.87 [-1.13, +3.00] | +0.29 [-2.39, +2.97] | -0.76 [-3.27, +1.75] | -0.11 [-3.90, +3.91] |

**F1/F2 — d_b and q_b for T (pp), with conditional 95 % sampling intervals**

| Step | Stat | 0 | (0,.25] | (.25,.5] | (.5,.75] | (.75,1) |
|---|---|---:|---:|---:|---:|---:|
| 934 | d | +2.15 [-2.63, +6.67] | -2.02 [-5.09, +1.24] | -5.40 [-8.48, -2.27] | -4.28 [-6.82, -1.64] | -1.43 [-4.61, +1.62] |
| 934 | q | +4.81 [+0.54, +8.95] | +0.64 [-1.51, +2.91] | -2.74 [-5.25, -0.13] | -1.62 [-4.07, +0.78] | +1.23 [-2.07, +4.47] |
| 1681 | d | -4.39 [-8.79, +0.49] | -4.43 [-7.36, -1.61] | -3.18 [-6.06, -0.22] | -2.36 [-4.61, -0.05] | -1.78 [-4.33, +0.65] |
| 1681 | q | -0.90 [-4.81, +3.41] | -0.94 [-3.05, +1.10] | +0.31 [-2.06, +2.75] | +1.13 [-1.10, +3.28] | +1.71 [-0.92, +4.39] |
| 2428 | d | -5.48 [-10.09, -0.83] | -1.61 [-4.61, +1.29] | -1.47 [-4.18, +1.23] | -2.32 [-4.46, -0.17] | -2.90 [-5.81, -0.14] |
| 2428 | q | -3.14 [-7.17, +1.04] | +0.74 [-1.38, +2.81] | +0.87 [-1.43, +3.15] | +0.03 [-2.10, +2.10] | -0.56 [-3.57, +2.36] |
| 3736 | d | -4.30 [-8.90, +0.37] | -3.96 [-6.83, -0.96] | -0.31 [-3.16, +2.48] | -1.86 [-3.94, +0.27] | -1.43 [-4.12, +1.27] |
| 3736 | q | -1.79 [-5.86, +2.36] | -1.44 [-3.54, +0.64] | +2.20 [-0.18, +4.52] | +0.66 [-1.33, +2.72] | +1.08 [-1.80, +3.95] |

**F1 across all 20 snapshots — number of snapshots whose conditional interval excludes 0 (C)**

| Stat | 0 | (0,.25] | (.25,.5] | (.5,.75] | (.75,1) |
|---|---:|---:|---:|---:|---:|
| d | 0 above / 1 below | 0 above / 0 below | 0 above / 1 below | 1 above / 0 below | 0 above / 0 below |
| q | 0 above / 1 below | 0 above / 0 below | 0 above / 0 below | 1 above / 0 below | 0 above / 0 below |

These counts describe dependent snapshots of one pair of trajectories and are not independent trials.

**F3 — advantage mass (MaxRL/GRPO): mass ratio; mass-share ratio**

| Step | 0 | (0,.25] | (.25,.5] | (.5,.75] | (.75,1) |
|---|---:|---:|---:|---:|---:|
| 934 | 2.72; 2.14 | 2.04; 1.61 | 1.13; 0.89 | 0.71; 0.56 | 0.41; 0.32 |
| 1681 | 2.62; 2.19 | 1.94; 1.62 | 1.08; 0.90 | 0.64; 0.54 | 0.48; 0.40 |
| 2428 | 2.49; 2.13 | 1.82; 1.55 | 1.04; 0.89 | 0.61; 0.52 | 0.50; 0.42 |
| 3736 | 2.21; 1.93 | 1.72; 1.50 | 0.99; 0.86 | 0.59; 0.52 | 0.46; 0.40 |

Total |A| over all training groups up to the step (GRPO; MaxRL): 934: 21847; 27748; 1681: 39070; 46588; 2428: 56135; 65694; 3736: 85619; 98004.

**F4 — batch structure: share of optimizer steps with 0 / 1 / 2 live groups**

| Arm | All steps | steps with a panel question in 0 | steps with a panel question in (0,.25] | steps with a panel question in (.25,.5] | steps with a panel question in (.5,.75] | steps with a panel question in (.75,1) |
|---|---|---|---|---|---|---|
| GRPO | 0.010 / 0.180 / 0.810 | 0.016 / 0.444 / 0.540 | 0.006 / 0.150 / 0.844 | 0.000 / 0.103 / 0.897 | 0.000 / 0.108 / 0.892 | 0.051 / 0.153 / 0.796 |
| MaxRL | 0.013 / 0.174 / 0.813 | 0.045 / 0.516 / 0.439 | 0.006 / 0.144 / 0.850 | 0.000 / 0.086 / 0.914 | 0.000 / 0.108 / 0.892 | 0.000 / 0.230 / 0.770 |

Reading rule (pre-registered): with identical weights, rollouts and loss handling, a one-live-group step gives loss gradients under the two estimators that differ only by a scalar. This is a statement about the estimators, not about the realized AdamW updates of the two runs.

**F5 — bin contents (opposite bank): mean baseline C, mean baseline T, share with baseline C ≥ 0.5**

| 0 | (0,.25] | (.25,.5] | (.5,.75] | (.75,1) |
|---:|---:|---:|---:|---:|
| 11.5 %, 21.5 %, 1.5 % | 29.6 %, 30.2 %, 20.6 % | 62.0 %, 51.8 %, 74.2 % | 81.1 %, 65.7 %, 100.0 % | 91.2 %, 81.3 %, 100.0 % |

**Clipping (seed 42).** MaxRL, every 10th optimizer step only (373 logged steps): share clipped (grad norm > 1) 78.0 %, mean clip coefficient 0.710, median 0.719, 5th percentile 0.315. GRPO: seed-42 GRPO training logs were not kept; not available.

## 5. Deviations and unresolved alerts

Every item is logged with time and evidence in `PROGRESS.md`.

**Approved by Sam before launch**

1. *Seed check relaxed at runtime.* The seed-42 code hard-wires `seed == 42` in trainer and evaluator. `camera_ready/launch_train.py` and `camera_ready/eval_run.py` replace only that invariant (42 → S) at runtime; every other config, manifest, lineage and data check runs unchanged. The training and evaluation checkouts stay clean at `9814757` / `1c26b1f`.
2. *Evaluation seed base = training seed* (S·100000 + i + 75000), identical within a pair. Extra endpoint batches add b·1,000,000 (collision check: none).
3. *Evaluation order.* Endpoint K=16 evaluations right after each pair; the 934/1681/2428 evaluations after all training; extra endpoint batches last, skipped if time runs out.
4. *Time rule.* A pair launches only if its training and all launched pairs' evaluations, analysis and fact-check fit before RESULTS_DUE with ≥ 2 h buffer, and training ends before TRAIN_CUTOFF (both moved by Sam).

**Resource-only or network-only (no recipe change)**

5. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for training (first-step OOM from allocator fragmentation on 32 GB cards without it). Peak allocated memory 29.9 of 31.5 GiB.
6. Colocated vLLM fraction kept at the frozen 0.30 instead of the A40 absolute budget (0.45 would starve training on 32 GB; changing it would also need a second invariant relaxed).
7. π0 bank collector `gpu_memory_utilization` 0.95 instead of 0.70 (closest feasible to the A40 absolute KV budget).
8. flash-attn compiled with `MAX_JOBS=24 NVCC_THREADS=2` instead of 1/1; userspace CUDA 13.0.0 toolkit (nvcc 13.0.48) because the box has only system CUDA 12.9.
9. uv default index: aliyun PyPI mirror over HTTPS (PyPI via the proxy dropped connections). Core versions identical to the A40 record; full lock in `box/pip_freeze.txt`.
10. Training processes look up the GSM8K `main` revision through `https://hf-mirror.com` (proxy off during training); resolved SHA `740312a…` verified in every manifest.
11. Logging-only instrumentation (`INSTRUMENTATION.diff`): `logging_steps = 1`, step log JSONL with read-only CUDA memory counters, resolved Trainer args dump.

**Scope limits that are not deviations but must be stated**

12. The GRPO seed-42 execution commit is not recorded; inferred `c664e26`. Image and pod identity of the two seed-42 arms cannot be verified (`INVENTORY.md` §8).
13. Seed-42 grad norms exist only for MaxRL at every 10th step; GRPO seed-42 training logs were not kept.

12a. *Data movement (Sam, 10-03).* After seeds 43/44 were pulled in full, only small analysis outputs are pulled to the laptop (metered hotspot). The main analysis and the bridge statistics ran on the box with the committed code (commit `45d6130`); the discovery-pair results were computed on the laptop from the A40 data already there (`results/discovery_seed42.json`) and passed in. A box-side reproduction check (382 checks at the four EVAL_STEPS) gave the same numbers as the laptop.
12b. *Endpoint K.* The primary cells use K = 16 for every pair: by the pre-registered all-or-nothing rule K = 64 requires all 64 endpoint responses in every pair, and the extra-batch rounds that could start before RESULTS_DUE − 6.5 h did not cover seed 45's third batch.
12c. *Backups.* Private HF dataset and checkpoint repos (Sam-approved), uploaded from the box only; HF-side SHA-256 verification (`PROGRESS.md`).

**Operational issues (no effect on data)**

- The AutoDL SSH gateway was intermittent on 10-02/10-03 (minutes-long connect timeouts); the box-side queue was unaffected.
- The laptop agent session restarted once (no check-ins 10-01 17:52–21:59 UTC) and later missed the end of the required evaluations (finished 10-03 14:56 UTC; analysis started 16:34 UTC) because a watcher timed out during an SSH outage. No decision depended on these gaps.
- `camera_ready/results/` was initially not tracked (repo-root `.gitignore` rule `results*/`); fixed 10-03 08:20 UTC.

**Unresolved alerts**

*(none)*

## 6. What in the paper needs to change

Proposals only; the paper's prose and submitted figures are not edited here. Candidate tables are in `paper_candidates/`.

1. **Abstract, §1 "Findings", §4 "Scope and limitations", Appendix F** — "one matched seed" / "The present study supplies no additional training seeds" are no longer accurate. State that three additional matched seed pairs (43–45) were trained on different hardware (2 × RTX 4080 SUPER per run) with the frozen recipe, analysed with a pre-registered protocol (`PREREG_RUNS.md`, pushed before the first run), and that the A40 seed-42 pair is reported separately as the discovery run.
2. **§3 (pre-exposure)** — "Low nonzero-reward questions improve by 10.0–12.8 percentage points before their own GRPO exposure": the gain is positive at all three cutoffs in every new seed (replicated under the pre-registered rule), but the magnitude varies across seeds: 10.0–12.5 pp (seed 43), 5.9–6.7 pp (seed 44), 6.2–8.6 pp (seed 45). Report the per-seed values rather than the seed-42 range alone. The exposed-versus-not-yet-exposed gap (U − E) differs in sign across the new seeds and its across-seed intervals are wide — consistent with the paper's existing statement that this contrast is uncertain; the camera-ready can add that it is uninformative at three seeds.
3. **§4 (advantage reweighting), Figure 2 left** — the realized reweighting toward the two lowest bins holds in every new pair at all four evaluated steps (mass ratios 1.58–2.78 in bins 0 and (0,.25] at step 3736). This can be stated as replicated across seeds.
4. **§4 "Absolute correctness advantage does not persist", Table 2, Figure 2 right** — for the eight pre-registered endpoint cells (d and q, C and R, bins 0 and (0,.25]) the three-seed intervals are all "unresolved at this number of runs"; none lies entirely above 0, so on the pre-registered rule the paper's Test-2 sentence does not have to change. Table 2 should become a multi-seed table (candidate: `paper_candidates/table_endpoint_multiseed.tex`, `table_primary_cells.tex`) with the seed-42 row separate. Descriptively, the seed-42 values lie below the range of the three new pairs in 6 of the 8 primary cells (e.g. d for C in bin 0: −4.71 for seed 42 vs +0.48 to +2.00 for seeds 43–45); the camera-ready should not present the seed-42 endpoint values as typical. Figure 2's 20-snapshot curves remain single-seed; the new seeds were evaluated at four steps only.
5. **Relative contrasts (§4 "Whole-panel-centered", Appendix E)** — the same eight-cell result applies to q. The supplementary mixed estimate (n = 4, mixing hardware) has an interval entirely above 0 for q in C, bin (0,.25] ([+0.13, +2.16] pp); it is not the pre-registered main inference and carries no multiplicity adjustment, so at most it can be mentioned as a supplementary, hardware-mixing estimate.
6. **Appendix A (Table 3 "Hardware; seed")** — add the replication runs: seeds 43–45, 2 × RTX 4080 SUPER 32 GB per run, same software stack; the colocated vLLM engine uses TRL's fixed rank seed, not the training seed; training needed `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on 32 GB cards (allocator only). Add the hardware bridge (PASS; evaluation compatibility only).
7. **Appendix D (advantage mass)** — the paper notes that mass omits optimizer-time clipping. New log data: 77.5–83.8 % of optimizer steps had pre-clip gradient norm above 1.0 in every new run (GRPO 83.5–83.8 %, MaxRL 77.5–77.8 %); the clip-weighted mass ratio in the two lowest bins is 1.11–1.39 at step 3736 versus 1.58–2.78 unweighted. This is a log proxy, not a measure of parameter contribution, and can be added as a descriptive caveat.
8. **Limitations** — add: three replication pairs give wide intervals for the endpoint contrasts; the replications were trained on different hardware from the discovery run (the bridge establishes evaluation compatibility only); endpoint contrasts use 16 responses per question.
