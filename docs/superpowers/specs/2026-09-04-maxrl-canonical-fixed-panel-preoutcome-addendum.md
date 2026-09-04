# Canonical MaxRL fixed-panel evaluation and H2/H3 pre-outcome addendum

Date: 2026-09-04

## Status

This addendum is written after the canonical MaxRL seed42 training trajectory
completed, but **before inspecting any canonical MaxRL fixed-panel K=16 C-bank
snapshot outcome**.

It does not rewrite the original MaxRL hypothesis amendment. It freezes the
evaluation transport and comparison table used to distinguish the predeclared
H2 primary prediction from H3.

The completed training trajectory itself may contain ordinary training reward
logs. Those are not substituted for the fixed-panel behavioral outcome defined
here.

## Required structural gate before evaluation

Before any fixed-panel deblind, the completed canonical MaxRL output must pass:

```text
mode = canonical
scientific_use = true
pilot_steps = null
execution commit = 981475795538eee391c7e86aa022ee609b539770
pi0 lineage = f89fc90226a67a6a3c7374f9c13abadfcecda88f397ab812fa4130f1f425605b
3736 generation steps, indices 0..3735
119552 ledger rows
7472 G=16 prompt groups
2 rank files
20 policy snapshots on the frozen 5% schedule
practical MaxRL advantage identities within 1e-6
finite token-level IS diagnostics
```

Failure of this gate blocks behavioral interpretation.

## Fixed-panel evaluation transport

The canonical MaxRL policies are evaluated with the **same transport** used for
canonical GRPO:

```text
panel: GSM8K train indices 0..255
snapshots: 5,10,...,100%
K: 16 completions/question/snapshot
sampling temperature: 0.8
top_p: 1.0
top_k: 0
repetition penalty: 1.0
completion cap: 2048
seed bank: C-bank
question seed = 42*100000 + dataset_index + 75000
reward: terminated and correct
C: correctness independent of termination
T: termination
R: terminated and correct
```

The C-bank seed is independent of the K=32 A/B baseline bank and is reused
across policy snapshots as common random numbers.

The MaxRL snapshot evaluator must verify the canonical MaxRL manifest,
snapshot schedule, snapshot policy lineage, pinned GSM8K revision, and frozen
GRPO/MaxRL shared sampling config before generation.

## Frozen baseline and p0 stratification

Use the same canonical K=32 A/B p0 bank already used for GRPO.

For every snapshot:

```text
A half defines p0 bin -> B half supplies baseline outcome
B half defines p0 bin -> A half supplies baseline outcome
symmetric value = equal average of the two directions
```

Frozen p0 bins remain:

```text
0
(0,.25]
(.25,.5]
(.5,.75]
(.75,1)
1
```

No bin boundaries may change after MaxRL outcomes are observed.

## Signal measurement

For both objectives, cumulative realized signal at each 5% snapshot is:

```text
cumulative sum|advantage| per panel question
```

using ledger rows with `generation_global_step < snapshot_step`.

The supporting outer-stack proxy remains:

```text
sum |advantage| * actual token-IS mass
```

and must continue to be labeled exploratory rather than an exact gradient norm.

## Behavioral measurement

At each snapshot and p0 bin, report separately:

```text
DeltaC
DeltaT
DeltaR
```

The primary behavioral target is DeltaC. DeltaT and DeltaR are supporting
outcomes and must not be collapsed into correctness.

## Frozen objective comparison table

For every 5% snapshot x p0 bin cell, the objective-comparison table contains:

```text
GRPO cumulative signal
MaxRL cumulative signal
MaxRL / GRPO signal ratio

GRPO DeltaC
MaxRL DeltaC
MaxRL - GRPO DeltaC

GRPO DeltaT
MaxRL DeltaT
MaxRL - GRPO DeltaT

GRPO DeltaR
MaxRL DeltaR
MaxRL - GRPO DeltaR

GRPO token-IS ESS/N
MaxRL token-IS ESS/N
```

The final 100% table is printed for compact inspection, while the complete
trajectory remains the evidentiary record.

## H2 / H3 decision rule

No new numerical materiality threshold is introduced after training.

The predeclared hypotheses remain:

- **H2 primary:** realized signal allocation differs materially across p0 while
  the shape/magnitude of DeltaC allocation changes much less.
- **H3 alternative:** signal allocation shifts and DeltaC allocation shifts in
  a corresponding direction.

The comparison script must therefore report measurements and end with:

```text
COMPLETE_NO_AUTOMATIC_H2_H3_VERDICT
```

It must not auto-label H2 or H3 from a post hoc threshold.

Interpretation is based on the frozen full binwise trajectory. A later
post-outcome checkpoint may state which predeclared world is better supported,
but this file must not be rewritten after the C-bank results are inspected.

## Evaluation artifact boundary

Raw MaxRL C-bank snapshot JSONL and manifests remain under
`controlled_run_outputs/` or private Hugging Face storage.

Lightweight post-analysis CSV/JSON tables and a post-outcome scientific
checkpoint may be committed after deblind.

The canonical MaxRL training execution commit remains the training provenance
even if evaluation/analysis support code is committed later.
