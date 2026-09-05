# Matched 2xA40 seed43/44 replication pre-outcome amendment

Date: 2026-09-05

## Status

This amendment supersedes the same-day single-A100 replication plan before any
A100 replication outcome was produced or inspected. The A100 plan is retained
as provenance only and is not the active execution contract.

The canonical seed42 A40 GRPO and MaxRL runs remain unchanged. The newly
available 8xA40 host permits two additional matched seed pairs under the same
2xA40 execution topology.

## Frozen replication matrix

Run four full training jobs concurrently on one 8xA40 host:

```text
GPU 0-1: GRPO  seed43
GPU 2-3: MaxRL seed43
GPU 4-5: GRPO  seed44
GPU 6-7: MaxRL seed44
```

Each arm must see exactly two A40 GPUs and run with WORLD_SIZE=2.

## Frozen changes

Relative to canonical seed42, the only scientific configuration field allowed
to change is:

```text
seed = 43 or 44
data_seed follows training seed
```

All other GRPO_INVARIANTS remain exactly equal to the canonical seed42 recipe.

The physical identity/UUID of the A40 devices is not frozen. GPU SKU, software
stack, DDP topology, and batch semantics are frozen.

## Frozen batch and generation geometry

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

## Engineering gate

Before launching the four full runs:

1. pull the approved exact execution commit;
2. run the full CPU test suite;
3. run the existing 2xA40 NCCL preflight separately on GPU pairs 0-1, 2-3,
   4-5, and 6-7;
4. run a 20-step GRPO seed43 and 20-step MaxRL seed43 pilot on two different
   GPU pairs;
5. require no OOM, finite numeric fields, the frozen runtime batch geometry,
   and the existing MaxRL finite-G structural identity.

A failed gate blocks the full runs. Do not alter batch semantics, NCCL
transport, or scientific hyperparameters to rescue a failed host without a new
written amendment.

## Frozen scientific question

The original seed42 result remains the discovery/canonical result.

Seeds43 and 44 are matched replications asking whether the predeclared
qualitative H1 -> H2 pattern survives training randomness:

1. H1: practical MaxRL reallocates realized signal toward lower frozen-p0
   regions relative to GRPO.
2. H2: correctness movement does not exhibit a stable corresponding
   reallocation.

No new numerical threshold is introduced after observing seed42.

## Evaluation deadline policy

All full runs save the same 20 policy snapshots. For deadline-sensitive
reporting, evaluate the 100% endpoint first after structural acceptance. Full
20-snapshot trajectories can follow.

The frozen baseline and behavior-evaluation protocol remains the same as the
canonical comparison. In particular, do not change the sequential evaluator
into a batched evaluator for paired comparisons.

## Interpretation

With seed42, seed43, and seed44 all run on 2xA40 using the same scientific
execution topology, the three matched objective pairs can be reported as a
three-seed robustness analysis. With only three seeds, report per-seed results
and descriptive mean/spread; do not overstate precision or equivalence.
