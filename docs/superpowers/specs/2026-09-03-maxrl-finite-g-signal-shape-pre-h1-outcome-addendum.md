# MaxRL finite-G signal-shape pre-H1-outcome addendum

Date: 2026-09-03

## Status

This addendum is written after the disposable 150-step practical MaxRL-15
shakedown completed its structural checker, but **before inspecting the
shakedown's p0-binned realized signal-allocation outcome**.

It does not rewrite the earlier MaxRL objective preregistration. It refines H1
with exact finite-G predictions for the actual TRL 1.12 GRPO implementation and
freezes the matched first-150-step analysis before reading that result.

No numerical H1 pass/fail threshold is introduced post hoc. The earlier H1
remains qualitative: realized MaxRL signal should move toward lower p0 relative
to canonical GRPO if the objective intervention survives the outer stack.

## Matched prefix

The H1 shakedown comparison is frozen to:

```text
GRPO generation_global_step: 0..149
MaxRL generation_global_step: 0..149
G: 16
rows per generation step: 32
prompt groups per generation step: 2
expected rows/objective: 4800
expected groups/objective: 300
frozen panel: GSM8K train indices 0..255
p0 bank: canonical K=32 A/B cross-fit bank
```

The exact ordered `(generation_global_step, dataset_index)` prompt-group
schedule must be identical between GRPO and MaxRL. A schedule mismatch rejects
the matched-prefix interpretation rather than being averaged away.

## Conditional finite-G prediction

For one binary-reward group with `N=G=16` and `K` successes, practical
MaxRL-15 uses:

```text
K = 0: all A_i = 0
K > 0: A_i = (r_i - K/16) / (K/16)
```

Therefore its nominal group absolute-advantage mass is:

```text
M_MaxRL(K) = 0          if K = 0
M_MaxRL(K) = 2(16-K)   if 1 <= K <= 16
```

Canonical TRL 1.12 GRPO uses group reward standard deviation with Bessel
correction and divides by `std + 1e-4`. For `0<K<16`:

```text
s_K = sqrt[K(16-K)/(16*15)]

M_GRPO(K)
  = [2K(16-K)/16] / [s_K + 1e-4]
```

Selected exact-stack predictions are:

| K | TRL GRPO sum|A| | MaxRL sum|A| | MaxRL / GRPO |
|---:|---:|---:|---:|
| 1 | 7.4970 | 30 | 4.0016 |
| 4 | 13.4134 | 24 | 1.7893 |
| 8 | 15.4889 | 16 | 1.0330 |
| 9 | 15.3674 | 14 | 0.9110 |
| 12 | 13.4134 | 8 | 0.5964 |
| 15 | 7.4970 | 2 | 0.2668 |

Thus the actual-stack crossover lies between K=8 and K=9, not exactly at K=8.
The intervention sharply upweights low-but-nonzero empirical-success groups and
downweights high-K groups relative to canonical GRPO.

These are conditional-on-K predictions. They are not claims that the two
training trajectories realize the same K sequence after their policies diverge.

## K=0 boundary

Both objectives are locally silent for an all-failure training group:

```text
K = 0 -> direct group signal = 0
```

Therefore H1 is **not** "the closer p0 is to zero, the larger MaxRL signal must
be." The intended shift is toward low-but-nonzero reachability.

The empirical frozen `p0=0` bin is based on finite K=32 baseline sampling and
does not prove true success probability exactly zero. A question in that bin
may later produce K>0 due to sampling variation or may become more reachable
through shared-parameter transfer.

For a prompt with true success probability p, the expected nominal practical
MaxRL group mass under 16 independent rollouts is:

```text
E[M_MaxRL | p] = 32[(1-p) - (1-p)^16]
```

This is zero at p=0 and reaches its population maximum at:

```text
p* = 1 - 16^(-1/15) ~= 0.168762
```

which further motivates the pre-outcome expectation of increased allocation in
the frozen low-but-nonzero region, especially `(0,.25]`, rather than a
monotone increase into exact zero reachability.

## Frozen H1 measurements

The primary outcome remains the same frozen p0-bin quantity used by the
canonical signal analysis:

```text
cumulative sum|advantage| per panel question
```

computed separately under A-bin and B-bin definitions and symmetrized with
equal direction weight.

Supporting measurements, all frozen before inspection, are:

```text
active-group fraction
K=0 group fraction
all-success group fraction
mean K/G
actual token-IS ESS/N
exploratory |A| x token-IS numerator-mass proxy per panel question
```

A scalar descriptive summary is also frozen:

```text
signal-weighted mean p0
  = sum_group p0(question(group)) * group_sum|A|
    / sum_group group_sum|A|
```

It is computed separately for A-bin and B-bin p0 values and then averaged
equally across directions.

A negative:

```text
MaxRL signal-weighted mean p0 - GRPO signal-weighted mean p0
```

is predeclared as **compatible with a left shift**, but is not by itself an
automatic H1 pass. The binwise allocation table remains primary because no
numerical materiality threshold was frozen in the original preregistration.

## Interpretation discipline

The matched-prefix script must report measurements and must not automatically
label H1 PASS/FAIL.

After the table is observed:

- if the binwise signal shape moves clearly toward low-but-nonzero p0 and the
  scalar summary moves left, record H1 as supported/compatible before launching
  canonical MaxRL;
- if realized signal does not move in the expected qualitative direction,
  follow H4 and inspect estimator/normalization/clipping/IS interactions;
- do not inspect downstream DeltaC behavior to rescue an H1 failure;
- do not rewrite this addendum after seeing the shakedown allocation.

The 150-step shakedown remains disposable and is not a canonical scientific
trajectory.
