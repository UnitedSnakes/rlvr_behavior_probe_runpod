# ATTRIB writing handoff — frozen project state

Date: 2026-09-05

## Purpose

This is the authoritative handoff for the next session, whose priority is
writing the ATTRIB workshop paper from the already accepted seed42 evidence.

Do not reopen GPU experimentation during paper drafting unless a concrete
writing blocker genuinely requires new evidence.

The active Git branch is:

```text
codex/signal-ledger
```

Do **not** merge or update `main` as part of this handoff.

## One-sentence project question

The project asks whether RLVR's local training signal predicts where behavior
improves, or whether shared-parameter updates substantially decouple
question-level signal allocation from question-level behavioral change.

The current causal/mechanistic chain is:

```text
objective
  -> realized training-signal allocation
  -> parameter update
  -> shared-parameter transfer / interference
  -> behavioral-change allocation
```

The central distinction is:

```text
where training signal is allocated
!=
where behavioral improvement appears
```

## Paper-level result hierarchy

The paper should present two independent but convergent evidence chains.

### Result 1 — GRPO own-exposure analysis is the main result

Canonical Qwen3-0.6B GRPO seed42 starts from the corrected untouched pi0 and
uses the frozen train-256 panel.

At the pre-frozen 25%, 45%, and 65% training cutoffs, panel questions were split
by whether their unique own training exposure had already happened. The split,
cutoffs, balance audit, interpretation scale, and measured-covariate adjusted
estimator were fixed before the exposed-vs-unexposed outcomes were inspected.

Frozen conclusion:

```text
no stable own-exposure advantage
```

Across 15 adjusted symmetric cutoff x p0-bin cells:

```text
transfer_compatible:      8
unexposed_higher:         3
mixed_or_uncertain:       2
not_classifiable:         2
own_exposure_candidate:   0
```

For the low-p0 bin `(0,.25]`:

| cutoff | exposed DeltaC | not-yet-exposed DeltaC | unexposed - exposed |
|---:|---:|---:|---:|
| 25% | +6.25 pp | +10.02 pp | +3.77 pp |
| 45% | +7.42 pp | +12.80 pp | +5.38 pp |
| 65% | +10.90 pp | +12.80 pp | +1.90 pp |

Paper-safe interpretation:

- substantial correctness improvement occurs before a question's own direct
  training exposure;
- there is no stable own-exposure advantage;
- this is evidence against a **strong prompt-local account**;
- it is consistent with substantial cross-question transfer.

Do **not** call this a randomized causal estimate. Exposure order may retain
unmeasured structure, and shared-parameter interference is intrinsic.

Authoritative checkpoint:

```text
docs/superpowers/checkpoints/2026-09-03-exposure-split-postoutcome-and-paper-claim-freeze.md
```

### Result 2 — MaxRL objective intervention is the independent follow-up

The pre-outcome MaxRL hypothesis asked whether an intervention that actually
moves realized signal allocation also moves the allocation of correctness
improvement.

Practical MaxRL-15 changes only the group advantage estimator while preserving
the matched GRPO outer stack.

Frozen finite-G semantics:

```text
G = 16
effective MaxRL order = 15
K = number of successes in group
K = 0: A_i = 0
K > 0: A_i = (r_i - K/16) / (K/16)
epsilon = 0
```

The canonical MaxRL seed42 run completed all 3736 steps and passed structural
acceptance:

```text
steps = 3736
rows = 119552
groups = 7472
rank_files = 2
snapshots = 20
max_advantage_error = 1.5894571969710114e-07
aggregate_token_is_ess_fraction = 0.9980541312524671
nonfinite_numeric_fields = 0
status = PASS
```

The frozen H1/H2/H3 decision is:

```text
H1 mechanism gate: SUPPORTED
H2 primary behavioral prediction: SUPPORTED
H3 alternative as the primary explanation: NOT SUPPORTED
H4 diagnostic stop: NOT ACTIVE
```

At the 100% endpoint:

| frozen p0 bin | MaxRL / GRPO cumulative |A| | GRPO DeltaC | MaxRL DeltaC | MaxRL - GRPO DeltaC |
|---|---:|---:|---:|---:|
| 0 | 2.205x | +5.216 pp | +0.504 pp | -4.712 pp |
| (0,.25] | 1.720x | +9.993 pp | +9.556 pp | -0.437 pp |
| (.25,.5] | 0.988x | +8.678 pp | +6.770 pp | -1.908 pp |
| (.5,.75] | 0.592x | +4.962 pp | +4.343 pp | -0.620 pp |
| (.75,1) | 0.463x | +1.760 pp | +1.604 pp | -0.156 pp |

The full 5%-through-100% trajectory is the evidentiary record, not only the
endpoint. Signal allocation remains persistently left-shifted under MaxRL,
while `DeltaC_MaxRL - DeltaC_GRPO` fluctuates rather than showing a matching
persistent reallocation.

Paper-safe interpretation:

> Changing the objective materially reallocates realized training signal, but
> question-level correctness improvement does not correspondingly and stably
> reallocate.

Do **not** claim that representations uniquely cause the effect or that
objective allocation can never affect behavioral allocation. The supported
mechanistic statement is that shared-parameter transfer/interference
substantially mediates the mapping from local training signal to local
behavioral change.

Authoritative checkpoint:

```text
docs/superpowers/checkpoints/2026-09-05-maxrl-h2-h3-postoutcome-gate.md
```

## Aggregate behavior must remain decomposed

The common pi0 K=32 aggregate is:

```text
R = 0.349121
T = 0.456177
C = 0.504517
```

Canonical GRPO at 100%:

```text
R = 0.530273
T = 0.772461
C = 0.577393

DeltaR = +18.12 pp
DeltaT = +31.63 pp
DeltaC =  +7.29 pp
```

Canonical MaxRL at 100%:

```text
R = 0.515381
T = 0.747314
C = 0.563965

DeltaR = +16.63 pp
DeltaT = +29.11 pp
DeltaC =  +5.95 pp
```

Do not use reward movement as a synonym for correctness movement. Termination
acquisition is the largest aggregate change under both objectives.

## Canonical lineage

Corrected pi0:

```text
HF repo:
HKReporter/rlvr-behavior-probe-pi0-corrected-canonical-2026-08-30

pi0_lineage_id:
f89fc90226a67a6a3c7374f9c13abadfcecda88f397ab812fa4130f1f425605b
```

Canonical GRPO seed42:

```text
HF repo:
HKReporter/rlvr-behavior-probe-grpo-canonical-seed42-2026-09-02

analysis implementation commit:
386f300e36562ad78063fcfd4b5ed4137325fd9d
```

Canonical MaxRL seed42:

```text
training execution commit:
981475795538eee391c7e86aa022ee609b539770

sequential evaluator implementation commit:
1c26b1f0f3c5f6ea1187fd00318587388a891272

model/checkpoint HF repo:
HKReporter/rlvr-behavior-probe-maxrl-canonical-seed42-2026-09-05
remote commit:
e67069c666ea372ce4fc4f0dc14617f35a1fce0f

analysis/raw-evaluation HF repo:
HKReporter/rlvr-behavior-probe-maxrl-analysis-seed42-2026-09-05
remote commit:
88b7f3244de0c61488ff53dab63d29e2f8669642
```

MaxRL remote backup verification:

```text
status = REMOTE_BACKUP_VERIFIED
expected_files = 313
verified_present = 313
size_verified = 313
LFS SHA256 verified = 99
```

## Fixed evaluation protocol

Train panel:

```text
GSM8K train indices 0..255
```

Baseline:

```text
K = 32 pre-RL rollouts/question
independent A/B halves of 16
cross-fit binning and baseline subtraction
```

Frozen p0 bins:

```text
0
(0,.25]
(.25,.5]
(.5,.75]
(.75,1)
1
```

Snapshot behavior:

```text
K = 16 C-bank rollouts/question/snapshot
temperature = 0.8
top_p = 1.0
top_k = 0
repetition penalty = 1.0
completion cap = 2048
sequential one-question/request evaluator
question seed = 42*100000 + dataset_index + 75000
```

The canonical MaxRL evaluator deliberately stayed sequential. A proposed
batched evaluator failed exact CRN token-path parity on 1147 / 1152 overlapping
rollouts and showed no compelling wall-clock benefit.

Do not silently change batching shape when exact common-random-number token
paths matter.

## Canonical training geometry

Both seed42 canonical objective runs use the matched 2xA40 topology:

```text
model = Qwen3-0.6B
world_size = 2
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
global optimizer batch = 32
generation batch = 32
G = 16
unique prompts / generation batch = 2
TRL steps_per_generation = 4
optimizer steps = 3736
```

The corrected pi0 is reused unchanged.

## ATTRIB paper framing

For the imminent workshop draft, use this hierarchy:

1. **Main result:** GRPO exposed-vs-unexposed analysis shows no stable
   own-exposure advantage.
2. **Second independent evidence:** MaxRL materially moves realized signal
   allocation (H1) while behavioral `DeltaC` allocation does not stably follow
   (H2).
3. **Supporting instrumentation:** pipeline distortions and evaluator parity
   checks motivate why realized signal and clean evaluation must be measured,
   but these are not the headline.
4. Keep `DeltaC`, `DeltaT`, and `DeltaR` separate.
5. Do not promote exploratory representation-level stories into causal claims.

Suggested core paper sentence:

> Local training-signal allocation is not a reliable map of where behavioral
> improvement appears under RLVR.

A slightly more conservative alternative:

> In this controlled seed42 study, question-level behavioral improvement does
> not stably track either own direct exposure or an objective-induced
> reallocation of realized training signal.

## Required limitation language

Use this formulation or an equivalent statement with the same scope:

> All behavioral and signal-allocation comparisons reported here are from a
> single matched training seed. We therefore do not estimate between-seed
> variability in either bin-level behavioral changes or objective-induced
> signal reallocation.

This limitation applies to both the GRPO exposure result and the MaxRL H1/H2
comparison.

## What is frozen / closed

Do not casually reopen these during writing:

- corrected pi0 lineage;
- canonical 2xA40 batch/topology semantics;
- G=16;
- reward = terminated and correct;
- C/T/R decomposition;
- p0 A/B cross-fit protocol;
- C-bank sequential evaluation;
- frozen p0 bins;
- GRPO exposure cutoffs 25/45/65;
- practical MaxRL-15 finite-G estimator;
- canonical token-level importance sampling semantics;
- H1/H2/H3 hypothesis ordering.

Historical sequence-level IS collapse and truncation-mask distortions are
instrumentation history, not a canonical phenomenon. Canonical token-level
ESS/N is approximately 0.998.

The Qwen3 thinking-control grammar question was measured and closed on
2026-08-30. Do not make it a paper blocker.

## A100 replication detour — closed

A single-A100 80GB path was engineering-qualified:

```text
world_size = 1
per_device_train_batch_size = 8
grad_accum = 4
global optimizer batch = 32
generation batch = 32
G = 16
gradient_checkpointing = True
vllm_gpu_memory_utilization = 0.30
```

GRPO and MaxRL 20-step pilots both passed. Approximate observed speed was
15.8 s/step.

A no-gradient-checkpointing speed experiment was rejected after OOMs at vLLM
memory fractions 0.30 and 0.20.

A six-run A100 seed42/43/44 replication suite was then planned. The initial
simultaneous launcher revealed that independent single-GPU vLLM jobs need
unique `MASTER_PORT` values: five processes collided on default TCPStore port
29500 while one acquired it.

The project decision is to **abort A100 full replication before scientific
use**. A complete A100 three-seed block would be scientifically usable, but the
long-term preferred replication topology remains A40 and producing an
additional topology block is not the current priority.

Any partial A100 full-run artifacts are provenance/engineering only and must
not enter scientific claims.

Authoritative record:

```text
docs/superpowers/checkpoints/2026-09-05-a100-full-replication-aborted-before-scientific-use.md
```

Operational note: the repo records the decision to stop. The current chat did
not independently verify the final RunPod process shutdown/termination, so if
the pod still exists, verify that the `a100full` tmux session and replication
processes are gone before deleting the pod.

## Future replication

Do not run new GPU experiments merely because hardware is available.

If this project continues after the workshop submission, the preferred
multi-seed extension is:

```text
A40 seed43: GRPO + MaxRL
A40 seed44: GRPO + MaxRL
```

under the original matched 2xA40 execution contract.

If the project later deliberately migrates to A100, write a new pre-outcome
topology decision and run a complete internally matched A100 suite.

## Test status

The last full pytest output observed on the A100 pod was:

```text
296 passed
1 failed
14 warnings
```

The single failure was not a logic failure: an existing test expected the old
exception-message substring after the new A100 MaxRL acceptance profile
generalized the checker. The fail-closed behavior itself was correct.

Message compatibility was fixed in:

```text
8a62690ed89ca67fbe9af21febbba80f463c3d7c
fix: preserve rank-count acceptance message
```

No post-fix full-suite output was observed in the chat before switching to
paper work. Before future code/GPU execution, run:

```bash
python -m pytest -q
```

Paper drafting itself does not require rerunning the suite.

## Paper-facing Git outputs

GRPO:

```text
analyses/canonical_snapshot_crossfit/
analyses/canonical_ledger_crossfit_signal/
analyses/canonical_exposure_split_transfer/
analyses/canonical_exposure_split_adjusted/
```

MaxRL:

```text
analyses/canonical_maxrl_snapshot_crossfit/
analyses/canonical_maxrl_ledger_crossfit_signal/
analyses/canonical_maxrl_grpo_objective_comparison/
```

Especially useful files:

```text
analyses/canonical_exposure_split_adjusted/adjusted_symmetric.csv
analyses/canonical_maxrl_grpo_objective_comparison/objective_comparison.csv
analyses/canonical_maxrl_grpo_objective_comparison/summary.json
analyses/canonical_snapshot_crossfit/aggregate_sanity.csv
analyses/canonical_maxrl_snapshot_crossfit/aggregate_sanity.csv
```

## Large-artifact boundary

Git contains code, configs, scientific specs/checkpoints, lightweight derived
tables/figures, and packaging/provenance.

Raw model checkpoints, rollout JSONL, signal ledgers, C-bank raw evaluations,
and large logs live in `controlled_run_outputs/` or private Hugging Face
storage.

The canonical MaxRL large-artifact backup is verified off-pod. The GRPO
canonical model repository remains the source of truth for its large training
artifacts.

Do not treat Git as a backup for `controlled_run_outputs/`.

## Next-session instruction

The next session should start by reading this file and the two paper-facing
post-outcome checkpoints:

```text
docs/superpowers/checkpoints/2026-09-05-attrib-writing-handoff.md
docs/superpowers/checkpoints/2026-09-03-exposure-split-postoutcome-and-paper-claim-freeze.md
docs/superpowers/checkpoints/2026-09-05-maxrl-h2-h3-postoutcome-gate.md
```

Then draft the ATTRIB paper. Do not spend the opening of that session
reconstructing experiment history or proposing new GPU runs.
