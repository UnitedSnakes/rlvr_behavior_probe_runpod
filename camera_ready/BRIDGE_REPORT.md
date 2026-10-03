# Hardware bridge report: RTX 4080 SUPER (AutoDL) vs stored A40 evaluations

**Status: FINAL (2026-10-03 16:34 UTC).** All nine pre-registered checkpoints bridged (π0 both banks; seed-42 GRPO and MaxRL at steps 934, 1681, 2428, 3736). Computed on the box with `camera_ready/analysis/bridge.py` from `/root/autodl-tmp/analysis_data` (A40 reference outputs fetched read-only from the private HF repos; SHA-256 checked).

Pre-registration: `camera_ready/PREREG_BRIDGE.md` (commit `556a271`, pushed 2026-10-01 05:13:41 UTC, before any box evaluation output existed). Statistics: `camera_ready/analysis/bridge.py`.

**Scope.** The bridge compares generation and scoring of the same frozen weights on the two machines. It says nothing about whether *training* on this box behaves like training on A40; the main analysis never mixes A40-trained and box-trained runs whatever the verdict.

## Verdict: **PASS**

Gates: (a) pooled Δ intervals inside ±1.5 pp: True; (b) per-cell |Δ/SE| rule: True; (c) arm interaction for C and R: True; (d) ≥ 4 RL checkpoints from two arms: True; GROSS FAIL: False; all nine checkpoints bridged: True.

## Per-checkpoint differences (box − A40), pp

| Checkpoint | ΔR (SE) | ΔT (SE) | ΔC (SE) | max \|Δ/SE\| |
|---|---:|---:|---:|---:|
| π0 (both banks, 32/question) | -0.24 (0.62) | -0.28 (0.68) | -0.52 (0.64) | 0.83 |
| GRPO seed 42, step 934 | -0.73 (0.76) | -0.20 (0.86) | -0.05 (0.77) | 0.96 |
| GRPO seed 42, step 1681 | -1.10 (0.85) | -0.39 (0.82) | -1.05 (0.81) | 1.30 |
| GRPO seed 42, step 2428 | -1.27 (0.83) | -0.39 (0.74) | -1.25 (0.79) | 1.58 |
| GRPO seed 42, step 3736 | -0.24 (0.82) | -0.46 (0.83) | -0.39 (0.79) | 0.56 |
| MaxRL seed 42, step 934 | +0.29 (0.77) | +1.03 (0.85) | -0.12 (0.80) | 1.21 |
| MaxRL seed 42, step 1681 | +0.78 (0.78) | +0.54 (0.88) | +0.90 (0.79) | 1.14 |
| MaxRL seed 42, step 2428 | -0.98 (0.82) | -0.24 (0.85) | -0.34 (0.79) | 1.19 |
| MaxRL seed 42, step 3736 | -0.61 (0.80) | -0.59 (0.88) | -0.05 (0.81) | 0.76 |

## Pooled over checkpoints (equal weights), question-bootstrap 95 % interval (5,000 draws), pp

| Metric | Pooled Δ | 95 % interval |
|---|---:|---:|
| R | -0.46 | [-0.94, +0.02] |
| T | -0.11 | [-0.66, +0.44] |
| C | -0.32 | [-0.80, +0.15] |

## Arm interaction Δ(MaxRL) − Δ(GRPO), pooled over steps [25, 45, 65, 100], pp

| Metric | Interaction | 95 % interval |
|---|---:|---:|
| R | +0.71 | [-0.37, +1.78] |
| T | +0.54 | [-0.64, +1.73] |
| C | +0.78 | [-0.35, +1.94] |

## Not gating: completion length, cap-hit rate, per-bin Δ

| Checkpoint | Mean length A40 → box (tokens) | Cap-hit A40 → box (%) |
|---|---:|---:|
| π0 (both banks, 32/question) | 1593.7 → 1594.5 | 54.38 → 54.66 |
| GRPO seed 42, step 934 | 1279.7 → 1290.3 | 31.67 → 31.86 |
| GRPO seed 42, step 1681 | 1180.7 → 1177.1 | 24.58 → 24.98 |
| GRPO seed 42, step 2428 | 1147.9 → 1151.2 | 23.12 → 23.51 |
| GRPO seed 42, step 3736 | 1144.7 → 1139.2 | 22.75 → 23.22 |
| MaxRL seed 42, step 934 | 1339.7 → 1314.2 | 34.33 → 33.30 |
| MaxRL seed 42, step 1681 | 1228.0 → 1225.0 | 28.08 → 27.54 |
| MaxRL seed 42, step 2428 | 1195.8 → 1189.1 | 25.46 → 25.71 |
| MaxRL seed 42, step 3736 | 1188.3 → 1190.9 | 25.27 → 25.85 |

Per-bin ΔC (pp) on the frozen cross-fit bins (symmetric average of the two directions):

| Checkpoint | 0 | (0,.25] | (.25,.5] | (.5,.75] | (.75,1) |
|---|---:|---:|---:|---:|---:|
| π0 (both banks, 32/question) | +1.83 | +0.93 | -1.59 | -1.92 | -4.04 |
| GRPO seed 42, step 934 | +0.68 | -2.37 | +0.86 | +2.54 | -0.98 |
| GRPO seed 42, step 1681 | -1.09 | -0.64 | -0.54 | -2.20 | -1.13 |
| GRPO seed 42, step 2428 | -1.53 | -1.15 | -1.78 | -1.35 | +0.96 |
| GRPO seed 42, step 3736 | -2.26 | -0.07 | -0.93 | +0.62 | +0.00 |
| MaxRL seed 42, step 934 | -1.94 | +1.07 | +0.31 | -2.03 | +1.77 |
| MaxRL seed 42, step 1681 | +3.43 | -1.28 | +2.32 | +1.07 | +1.62 |
| MaxRL seed 42, step 2428 | -0.34 | -1.36 | +0.05 | +0.06 | +1.91 |
| MaxRL seed 42, step 3736 | +2.30 | -0.17 | -0.49 | -0.80 | -0.35 |

For π0 the bins are selected by the same A40 banks whose rates are subtracted, so per-bin π0 Δ contains regression toward the mean by construction (low bins up, high bins down); it is shown because it was pre-registered, not as a hardware signal. Per-bank π0 Δ (pp): bank A: R -0.78, T -0.49, C -0.51, bank B: R +0.29, T -0.07, C -0.54.

## Consequence (C4)

PASS: baselines and bins are the paper's stored A40 banks. A supplementary estimate pooling the A40 discovery pair with the box pairs at K = 16 is reported, labelled as mixing hardware.
