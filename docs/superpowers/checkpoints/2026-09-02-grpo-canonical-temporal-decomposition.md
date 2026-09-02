# Controlled Qwen3 — Canonical temporal reward decomposition

## Status

Completed 2026-09-02 from the sealed canonical signal ledger. This is the first substantive canonical outcome analysis.

The original preregistration specified `P(T)`, `P(C|T)`, and `P(R)` for temporal decomposition. A post-run, pre-outcome-inspection addendum additionally declared unconditional `P(C)` and `P(T|C)` as conditioning checks before this table was inspected.

## 5% windows

```text
window       P(T)     P(C)     P(R)    P(C|T)    P(T|C)   mean_len
  5%        48.23%   51.79%   36.85%    76.40%     71.15%     1559.3
 10%        54.71%   58.44%   43.17%    78.89%     73.86%     1499.9
 15%        57.49%   55.14%   43.06%    74.90%     78.09%     1447.0
 20%        65.24%   57.54%   49.00%    75.10%     85.16%     1345.2
 25%        69.67%   59.22%   51.45%    73.85%     86.88%     1264.8
 30%        70.10%   57.92%   51.27%    73.13%     88.52%     1264.8
 35%        75.85%   61.18%   55.72%    73.45%     91.07%     1167.3
 40%        75.52%   60.55%   55.24%    73.15%     91.23%     1165.6
 45%        76.75%   60.33%   55.67%    72.52%     92.27%     1137.4
 50%        76.40%   59.46%   54.41%    71.22%     91.51%     1152.5
 55%        78.11%   62.58%   57.47%    73.58%     91.83%     1116.7
 60%        79.19%   61.85%   57.15%    72.17%     92.41%     1109.9
 65%        76.68%   57.46%   53.23%    69.41%     92.63%     1148.7
 70%        78.66%   60.34%   56.15%    71.38%     93.05%     1116.2
 75%        79.01%   61.36%   57.20%    72.40%     93.22%     1107.8
 80%        77.99%   59.04%   54.95%    70.45%     93.07%     1139.8
 85%        79.34%   61.63%   56.77%    71.55%     92.11%     1104.1
 90%        76.43%   59.26%   54.45%    71.25%     91.89%     1152.8
 95%        76.85%   56.90%   52.14%    67.84%     91.63%     1156.3
100%        79.14%   58.69%   54.68%    69.09%     93.17%     1115.5
```

First 10% versus last 10%:

```text
P(T):       51.47% -> 78.00%   +26.53 pp
P(C):       55.11% -> 57.80%    +2.68 pp
P(R):       40.01% -> 53.41%   +13.40 pp
P(C|T):     77.65% -> 68.46%    -9.18 pp
P(T|C):     72.51% -> 92.40%   +19.89 pp
mean length:1529.6 -> 1135.9   -393.7 tokens
```

Quarter aggregates:

```text
quarter    P(T)     P(C)     P(R)    P(C|T)    P(T|C)
Q1         59.07%   56.43%   44.71%    75.83%     79.03%
Q2         74.93%   59.89%   54.46%    72.69%     90.92%
Q3         78.33%   60.72%   56.24%    71.79%     92.63%
Q4         77.95%   59.10%   54.60%    70.03%     92.37%
```

## Interpretation

The dominant canonical training change is termination acquisition, not a clear later correctness-learning phase.

- `P(T)` rises by 26.5 percentage points from the first to last 10% windows, while unconditional `P(C)` rises only 2.7 points.
- Mean completion length falls by about 394 tokens.
- `P(T|C)` rises by about 19.9 points: an increasing fraction of correctness-only trajectories also terminate within the budget.
- `P(C|T)` falls rather than rises. This conditional decline is expected to be composition-sensitive because the terminated population changes substantially during training, so it must not be interpreted as evidence that mathematical competence worsens.
- The largest movement occurs in the first half: Q1 -> Q2 gives `P(T)` +15.86 pp and reward +9.75 pp, whereas Q2 -> Q3 adds only +3.40 pp termination and +1.78 pp reward. Q3 -> Q4 is approximately flat/slightly down.

A simple counterfactual decomposition makes the same point. Holding early `P(C|T)` fixed, the observed increase in `P(T)` would by itself raise reward by roughly 20.6 percentage points; the contemporaneous decline in `P(C|T)` offsets roughly 7.2 points, leaving the observed net reward increase of about 13.4 points. This is descriptive decomposition, not a causal attribution.

## Decision

The preregistered hypothesis space allowed termination transition, correctness transition, or both. For this seed, the data support a strong termination transition with only modest unconditional correctness improvement and no distinct later correctness phase.

Do not upgrade this single-seed temporal shape into a population-level training claim. Next evaluate the frozen train and held-out probability panels across the saved policy snapshots to determine how probability movement is distributed by initial difficulty.
