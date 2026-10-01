# Hardware bridge report: RTX 4080 SUPER (AutoDL) vs stored A40 evaluations

**Status: INTERIM (2026-10-01 06:30 UTC).** Only π0 has been bridged. The eight seed-42 snapshots run in idle-GPU windows between pairs and after training; this report is regenerated when they finish. Until then the verdict is NOT PASSED (bridge unfinished) for C4 purposes.

Pre-registration: `camera_ready/PREREG_BRIDGE.md` (commit `556a271`, pushed 2026-10-01 05:13:41 UTC, before any box evaluation output existed). Statistics: `camera_ready/analysis/bridge.py`.

**Scope.** The bridge compares generation and scoring of the same frozen weights on the two machines. It says nothing about whether *training* on this box behaves like training on A40; the main analysis never mixes A40-trained and box-trained runs whatever the verdict.

## Verdict: **NOT PASSED (bridge unfinished)**

Gates: (a) pooled Δ intervals inside ±1.5 pp: False; (b) per-cell |Δ/SE| rule: True; (c) arm interaction for C and R: False; (d) ≥ 4 RL checkpoints from two arms: False; GROSS FAIL: False; all nine checkpoints bridged: False.

## Per-checkpoint differences (box − A40), pp

| Checkpoint | ΔR (SE) | ΔT (SE) | ΔC (SE) | max \|Δ/SE\| |
|---|---:|---:|---:|---:|
| π0 (both banks, 32/question) | -0.24 (0.62) | -0.28 (0.68) | -0.52 (0.64) | 0.83 |

## Pooled over checkpoints (equal weights), question-bootstrap 95 % interval (5,000 draws), pp

| Metric | Pooled Δ | 95 % interval |
|---|---:|---:|
| R | -0.24 | [-1.46, +0.95] |
| T | -0.28 | [-1.62, +1.06] |
| C | -0.52 | [-1.76, +0.74] |

Arm interaction: not computable yet (no step with both arms bridged).

## Not gating: completion length, cap-hit rate, per-bin Δ

| Checkpoint | Mean length A40 → box (tokens) | Cap-hit A40 → box (%) |
|---|---:|---:|
| π0 (both banks, 32/question) | 1593.7 → 1594.5 | 54.38 → 54.66 |

Per-bin ΔC (pp) on the frozen cross-fit bins (symmetric average of the two directions):

| Checkpoint | 0 | (0,.25] | (.25,.5] | (.5,.75] | (.75,1) |
|---|---:|---:|---:|---:|---:|
| π0 (both banks, 32/question) | +1.83 | +0.93 | -1.59 | -1.92 | -4.04 |

For π0 the bins are selected by the same A40 banks whose rates are subtracted, so per-bin π0 Δ contains regression toward the mean by construction (low bins up, high bins down); it is shown because it was pre-registered, not as a hardware signal. Per-bank π0 Δ (pp): bank A: R -0.78, T -0.49, C -0.51, bank B: R +0.29, T -0.07, C -0.54.

## Consequence (C4)

Not PASS: baselines come from the π0 banks regenerated on this box (A-bin uses regenerated bank B and vice versa); bin membership stays the paper's frozen membership. No pooled estimate. The discovery pair is shown separately, plus one sensitivity row using its checkpoints as re-evaluated on this box where available.
