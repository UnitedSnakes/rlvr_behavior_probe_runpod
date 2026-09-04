# Practical MaxRL-15 — matched first-150 H1 post-outcome gate

Date: 2026-09-03

## Status

The pre-frozen matched first-150-step H1 analysis has been deblinded after the
disposable 150-step MaxRL shakedown passed structural acceptance.

Decision:

```text
H1 mechanism gate: SUPPORTED
canonical MaxRL seed42: AUTHORIZED AFTER FRESH TEST/CLEAN-CHECKOUT GATE
H2/H3 behavioral outcome: NOT YET OBSERVED
```

This is a post-outcome checkpoint. It does not rewrite the earlier MaxRL
objective preregistration or the pre-H1 finite-G addendum.

## Matched-prefix integrity

The analysis verified:

```text
GRPO prefix steps: 0..149
MaxRL prefix steps: 0..149
GRPO rows: 4800
MaxRL rows: 4800
GRPO groups: 300
MaxRL groups: 300
G: 16
ordered (generation_global_step, dataset_index) schedule: exact match
```

The comparison therefore uses the same 300 prompt-group positions under the
same frozen K=32 A/B p0 bank. The realized reward/K trajectories can diverge
after the objective changes; exact rollout outcomes are not assumed matched.

## Predeclared scalar left-shift summary

The pre-H1 addendum froze signal-weighted mean p0 as a scalar descriptive
summary, while retaining the binwise table as primary.

Observed:

```text
GRPO signal-weighted mean p0  = 0.3172371285
MaxRL signal-weighted mean p0 = 0.2295168070
MaxRL - GRPO                  = -0.0877203215
relative reduction vs GRPO   ~= 27.65%
```

The negative shift was predeclared as compatible with a left shift. Its
magnitude is substantial and agrees with the primary binwise pattern.

## Primary p0-bin realized signal result

Symmetric cumulative absolute-advantage mass per panel question over the first
150 generation steps:

| frozen p0 bin | GRPO | MaxRL | MaxRL / GRPO |
|---|---:|---:|---:|
| 0 | 0.3186 | 0.8026 | 2.5193 |
| (0,.25] | 0.3846 | 1.2346 | 3.2102 |
| (.25,.5] | 0.8863 | 1.1353 | 1.2810 |
| (.5,.75] | 0.6217 | 0.6471 | 1.0408 |
| (.75,1) | 0 | 0 | undefined |

The relative amplification is strongly concentrated at low p0. In particular,
the largest observed ratio is in the predeclared low-but-nonzero bin
`(0,.25]`, while the ratio falls toward one by `(.5,.75]`.

This matches the finite-G prediction that practical MaxRL-15 should reallocate
nominal signal toward low-but-nonzero reachability relative to canonical TRL
GRPO.

## K=0 boundary behaves as predicted

Observed K=0 group fractions:

| frozen p0 bin | GRPO K=0 | MaxRL K=0 |
|---|---:|---:|
| 0 | 0.25 | 0.25 |
| (0,.25] | 0.10 | 0.10 |
| (.25,.5] | 0 | 0 |
| (.5,.75] | 0 | 0 |

The empirical p0=0 bin is amplified less than `(0,.25]`, consistent with the
pre-frozen caveat that K=0 groups are locally silent under both objectives.
The result should not be described as "the harder the prompt, the larger the
MaxRL signal all the way to p=0."

For all exposed bins, the observed active-group fractions are the same between
GRPO and MaxRL in this prefix. Thus the qualitative allocation shift is not
explained merely by one objective having more active versus all-zero/all-one
groups in those bins. Mean K/G does differ after the policies diverge, as
expected for a realized-trajectory comparison.

## Supporting outer-stack diagnostic

The exploratory pre-PPO DAPO numerator-mass proxy
`sum |A| * token-IS mass` shows the same low-p0 emphasis:

| frozen p0 bin | MaxRL / GRPO proxy mass |
|---|---:|
| 0 | 2.1170 |
| (0,.25] | 2.9778 |
| (.25,.5] | 1.2877 |
| (.5,.75] | 0.9524 |

This proxy is not an exact gradient norm and must not be presented as one. It
is supporting evidence that the nominal advantage reallocation survives the
matched token-IS/DAPO outer stack in the intended qualitative direction.

Token-level importance sampling remains numerically healthy across the exposed
bins for both objectives, with ESS/N approximately 0.998. There is no observed
IS degeneracy that could plausibly explain the left-shift result.

## H1 decision

The frozen H1 mechanism gate asked whether realized signal allocation shifts
qualitatively toward lower p0 under practical MaxRL-15 relative to canonical
GRPO.

The answer for the disposable matched seed42 first-150-step shakedown is:

```text
SUPPORTED
```

Reason:

1. the primary binwise MaxRL/GRPO signal ratio is largest in the
   low-but-nonzero p0 region and decreases sharply with p0;
2. the predeclared signal-weighted mean p0 moves from 0.3172 to 0.2295;
3. the K=0 boundary behaves in the direction anticipated by the pre-outcome
   finite-G addendum;
4. the supporting token-IS numerator-mass proxy preserves the same qualitative
   low-p0 emphasis;
5. structural and numerical acceptance checks are already PASS.

No post hoc numerical threshold is introduced. This is a qualitative
predeclared mechanism judgment, not a claim of a sampling-theoretic p-value or
multi-seed population effect.

## Interpretation boundary

This shakedown result establishes that, in the matched controlled stack, the
objective intervention produces the intended realized training-signal
reallocation.

It does **not** establish H2 or H3. No canonical MaxRL fixed-panel behavioral
trajectory has yet been inspected.

Do not claim from this shakedown that:

- MaxRL improves final correctness;
- low-p0 questions improve more behaviorally;
- the realized signal shift necessarily causes a matching DeltaC shift;
- the effect generalizes across training seeds.

Those are downstream questions.

## Canonical launch authorization

The earlier implementation protocol required the 150-step shakedown to show
the intended mechanism before a full canonical MaxRL trajectory.

That gate is now satisfied.

A canonical seed42 MaxRL run is authorized only after:

```text
fresh full pytest PASS on the exact execution commit
clean git working tree
exact corrected untouched pi0 lineage
2 x A40 default NCCL preflight already valid for this host
fresh output directory
mode = canonical
no --pilot-steps override
```

The canonical run must restart from the untouched corrected pi0 and must not
resume or reuse either disposable pilot/shakedown trajectory.

The next scientific deblind after canonical training is the predeclared
signal-allocation and fixed-panel behavioral comparison for H2/H3.
