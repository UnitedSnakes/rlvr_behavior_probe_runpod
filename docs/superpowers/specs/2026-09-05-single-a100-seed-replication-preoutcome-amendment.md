# Single-A100 seed42/43/44 matched replication pre-outcome amendment

Date: 2026-09-05

## Status

This amendment is written after the canonical A40 seed42 H1/H2 result and
before any seed42/43/44 A100 replication outcome is inspected.

It defines a separate replication suite. It does not alter the canonical A40
seed42 execution contract or its provenance.

## Frozen replication matrix

Run matched GRPO and practical MaxRL-15 arms for:

```text
seed 42
seed 43
seed 44
```

All six A100 runs must use the same A100 80GB SKU and the same software image.
Each run must see exactly one CUDA device.

Seed42 on A100 is the hardware/topology bridge. Seeds43 and 44 are independent
training-seed replications within the same A100 execution contract.

## Frozen single-GPU batch geometry

Canonical A40 execution:

```text
world_size = 2
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
global_optimizer_batch_size = 32
generation_batch_size = 32
steps_per_generation = 4
num_generations = 16
unique prompts per generation batch = 2
```

A100 replication execution:

```text
world_size = 1
per_device_train_batch_size = 8
gradient_accumulation_steps = 4
global_optimizer_batch_size = 32
generation_batch_size = 32
steps_per_generation = 4
num_generations = 16
unique prompts per generation batch = 2
```

Thus the optimizer batch, generation batch, finite-G group size, number of
unique prompts per generation, and steps-per-generation remain unchanged.

The following execution details do change and must be disclosed:

```text
2-rank DDP -> single process
A40 -> A100 80GB
per-rank microbatch 4 -> single-device microbatch 8
gradient all-reduce -> none
```

The A100 suite is therefore not bitwise or sample-path-equivalent to the A40
canonical run.

## Frozen scientific invariants

Except for seed, world size, per-device batch size, and GPU SKU/topology, all
canonical GRPO/MaxRL scientific settings remain unchanged, including:

```text
same untouched corrected pi0 lineage
same GSM8K training data and ordering logic
same one-epoch schedule
same binary terminated+correct reward
same G=16
same temperature/top-p/top-k/repetition penalty
same completion and prompt caps
same learning rate and cosine schedule
same AdamW optimizer settings
same DAPO loss and group reward scaling
same token-level vLLM importance correction
same practical MaxRL advantage estimator
same snapshot schedule
```

## Engineering gate

Before any full A100 replication run, a 20-step single-A100 pilot must pass for
both GRPO and MaxRL on the new execution lane.

The pilot must establish at minimum:

```text
exactly one visible A100 80GB-class GPU
frozen package versions and CUDA runtime
BF16 + FlashAttention2 forward/backward probe
world_size = 1
per_device batch = 8
global optimizer batch = 32
generation batch = 32
steps_per_generation = 4
G = 16
finite ledger fields
MaxRL finite-G advantage identity for the MaxRL pilot
no OOM
```

A failed pilot blocks the six full replications. Do not respond to OOM by
silently changing a scientific hyperparameter; write a new amendment first.

## Frozen outcome use

The original A40 seed42 result remains the discovery/canonical result.

Within the A100 suite, the primary replication question is whether the
predeclared qualitative pattern repeats across seeds 42, 43, and 44:

1. H1: MaxRL materially reallocates realized signal toward lower frozen p0
   relative to GRPO.
2. H2: the corresponding DeltaC allocation changes much less and does not show
   a stable matching reallocation.

No new numerical materiality or equivalence threshold is introduced.

For deadline-sensitive reporting, the 100% endpoint may be evaluated first,
but all 20 policy snapshots remain saved for the complete trajectory analysis.
The same frozen K32 A/B p0 bank and sequential K16 C-bank evaluation protocol
must be used.

## Interpretation discipline

Seeds42-44 inside the A100 suite share hardware/topology and may be used to
assess training-seed robustness descriptively.

The original A40 seed42 versus A100 seed42 comparison is a hardware/topology
bridge, not a pure seed comparison. Any cross-platform difference combines GPU
and distributed-topology effects.

Do not claim a numerical seed variance estimate from three seeds without an
explicit estimator and uncertainty analysis. The immediate purpose is
qualitative replication of H1/H2.
