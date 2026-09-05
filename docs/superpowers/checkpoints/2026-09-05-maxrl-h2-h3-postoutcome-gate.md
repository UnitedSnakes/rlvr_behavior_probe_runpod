# Canonical GRPO vs practical MaxRL-15 — H2/H3 post-outcome gate

Date: 2026-09-05

## Status

The canonical practical MaxRL-15 seed42 trajectory passed structural
acceptance, completed the frozen train256 K=16 C-bank evaluation under the
same sequential evaluator structure used for canonical GRPO, and completed the
pre-frozen cross-fit signal-versus-behavior comparison.

This checkpoint is written after the complete 5%-through-100% MaxRL fixed-panel
trajectory was deblinded. It does not rewrite the pre-outcome files.

Decision:

```text
H1 mechanism gate: SUPPORTED
H2 primary behavioral prediction: SUPPORTED
H3 alternative as the primary explanation: NOT SUPPORTED
H4 diagnostic stop: NOT ACTIVE
```

The H2/H3 judgment is qualitative, as frozen before outcome inspection. No new
numerical materiality threshold is introduced here.

## Provenance and gates

Canonical MaxRL training provenance remains:

```text
training execution commit:
981475795538eee391c7e86aa022ee609b539770

pi0 lineage:
f89fc90226a67a6a3c7374f9c13abadfcecda88f397ab812fa4130f1f425605b

steps = 3736
ledger rows = 119552
prompt groups = 7472
G = 16
rank files = 2
policy snapshots = 20
aggregate token-IS ESS/N = 0.9980541312524671
max practical-MaxRL advantage identity error = 1.5894571969710114e-07
```

The completed fixed-panel JSONL audit found, for every snapshot from 5% through
100%:

```text
good questions = 256
unique questions = 256
bad JSONL rows = 0
missing questions = 0
duplicate questions = 0
```

The cross-fit movement analysis passed:

```text
CANONICAL SNAPSHOT CROSS-FIT ANALYSIS: PASS
```

The full MaxRL ledger join passed canonical geometry:

```text
files = 2
rows = 119552
groups = 7472
steps = 0..3735
ranks = [0, 1]
panel groups = 256
unique panel questions exposed = 256
CANONICAL LEDGER CROSS-FIT SIGNAL ANALYSIS: PASS
```

The two rank ledger filenames carry adjacent rank-local launch timestamps
(`20260904T015650Z` and `20260904T015651Z`). Canonical geometry and the
dedicated MaxRL acceptance checker establish that these are the two rank files
of the accepted distributed trajectory. Analysis support therefore permits
rank-local filename launch tokens only when explicitly requested; rank/file and
geometry checks remain fail-closed.

## Full-trajectory signal intervention

The realized signal intervention is persistent across the entire frozen
5%-through-100% trajectory.

For every snapshot:

- the `p0=0` bin has more MaxRL cumulative `|A|` than GRPO;
- the `(0,.25]` bin has more MaxRL cumulative `|A|` than GRPO;
- the `(.75,1)` bin has less MaxRL cumulative `|A|` than GRPO.

The intermediate bins converge toward or below the GRPO allocation as training
progresses. At 100%:

| frozen p0 bin | GRPO cum|A| | MaxRL cum|A| | MaxRL / GRPO |
|---|---:|---:|---:|
| 0 | 6.219 | 13.713 | 2.205 |
| (0,.25] | 11.945 | 20.551 | 1.720 |
| (.25,.5] | 13.792 | 13.624 | 0.988 |
| (.5,.75] | 11.659 | 6.901 | 0.592 |
| (.75,1) | 9.403 | 4.353 | 0.463 |

Thus the canonical run reproduces the H1 direction at full scale: practical
MaxRL reallocates realized training signal strongly toward lower frozen-p0
regions relative to GRPO.

## Behavioral comparison

The primary behavioral target is `DeltaC`. At 100%:

| frozen p0 bin | GRPO DeltaC (pp) | MaxRL DeltaC (pp) | MaxRL - GRPO (pp) |
|---|---:|---:|---:|
| 0 | 5.216 | 0.504 | -4.712 |
| (0,.25] | 9.993 | 9.556 | -0.437 |
| (.25,.5] | 8.678 | 6.770 | -1.908 |
| (.5,.75] | 4.962 | 4.343 | -0.620 |
| (.75,1) | 1.760 | 1.604 | -0.156 |

The endpoint is not a corresponding signal-to-behavior reallocation:

- `(0,.25]` receives 1.720x the GRPO cumulative signal but has essentially the
  same endpoint correctness gain;
- `(.5,.75]` receives only 0.592x the GRPO cumulative signal while its
  endpoint correctness gain changes much less;
- `(.75,1)` receives only 0.463x the GRPO cumulative signal while its
  endpoint correctness gain is nearly unchanged;
- `p0=0` receives 2.205x the GRPO cumulative signal but has substantially
  lower, not higher, endpoint `DeltaC`.

The full trajectory gives the same qualitative conclusion. The signal ratios
form a stable left-shifted allocation pattern, whereas
`DeltaC_MaxRL - DeltaC_GRPO` fluctuates in sign and magnitude across
snapshots rather than moving persistently in the corresponding direction.
There are transient cells compatible with H3, especially early in training,
but they do not form a stable trajectory-level mapping from signal
reallocation to correctness reallocation.

## Supporting aggregate behavior

The shared p0 baseline aggregate is:

```text
R = 0.349121
T = 0.456177
C = 0.504517
```

Canonical MaxRL reaches at 100%:

```text
R = 0.5154
T = 0.7473
C = 0.5640
```

so the aggregate MaxRL movement from p0 is approximately:

```text
DeltaR = +16.63 pp
DeltaT = +29.11 pp
DeltaC =  +5.95 pp
```

Termination improvement remains substantially larger than correctness
improvement, so `DeltaR`, `DeltaT`, and `DeltaC` must remain separated.

## H2 / H3 judgment

The pre-outcome H2 prediction was that MaxRL could materially reallocate
realized training signal while the shape/magnitude of correctness improvement
changed much less. That is the better-supported world.

The signal intervention is large, persistent, and ordered by frozen p0.
Correctness movement does not follow it with a corresponding persistent
reallocation. In several high-information endpoint cells, large signal ratios
coexist with small `DeltaC` contrasts, and the `p0=0` endpoint moves in the
opposite behavioral direction.

Therefore:

```text
H2: supported
H3: not supported as the primary explanation
```

This does not prove that objective allocation can never affect behavioral
allocation, nor that shared representations uniquely cause the observed
decoupling. The supported interpretation is the one frozen in the MaxRL
amendment: shared-parameter transfer/interference substantially mediates the
mapping from local training signal to local behavioral change, so local signal
reallocation does not automatically localize where correctness improves.

## Paper-facing implication

The original GRPO result remains the main result:

```text
no stable own-exposure advantage
```

The MaxRL intervention is now a clean supporting result rather than future
work: deliberately changing the objective produces a strong, persistent
reallocation of realized training signal without a correspondingly stable
reallocation of question-level correctness improvement.

Do not upgrade this to "representations dominate" or a causal proof of the
unique transfer mechanism.

## Historical evaluator note

A batched evaluator optimization was proposed and tested before the canonical
MaxRL behavior result. Exact sequential-versus-batched parity failed on 1147 of
1152 overlapping rollout token paths, and batching showed no compelling
wall-clock benefit. The canonical MaxRL fixed-panel result was therefore
restarted under the same sequential evaluator structure as GRPO.

See:

```text
docs/superpowers/checkpoints/2026-09-04-maxrl-cbank-batching-parity-fail.md
docs/superpowers/specs/2026-09-04-maxrl-canonical-fixed-panel-preoutcome-addendum.md
```
