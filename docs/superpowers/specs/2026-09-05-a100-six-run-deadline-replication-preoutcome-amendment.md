# Single-A100 six-run deadline replication pre-outcome amendment

Date: 2026-09-05

## Status

This amendment supersedes the same-day matched-2xA40 seed43/44 execution plan
before any replication training outcome was produced on that plan.

Reason for supersession: the available 8xA40 host failed the default NCCL
collective gate and was rejected. With the ATTRIB submission deadline now
within roughly one day, waiting for another 8xA40 host is no longer
deadline-compatible.

No A100 replication outcome has been produced or inspected at the time of this
amendment.

## Frozen replication matrix

Run six independent single-GPU jobs concurrently on six identical A100 80GB
GPUs:

```text
GPU0: GRPO  seed42
GPU1: MaxRL seed42
GPU2: GRPO  seed43
GPU3: MaxRL seed43
GPU4: GRPO  seed44
GPU5: MaxRL seed44
```

Each run must see exactly one A100 80GB-class GPU.

## Frozen single-GPU execution geometry

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

Relative to the original A40 canonical execution, the A100 suite changes:

```text
A40 -> A100 80GB
WORLD_SIZE 2 -> 1
per-device train batch 4 -> 8
two-rank DDP/all-reduce -> single process
```

The optimizer batch, generation batch, finite-G group size, number of unique
prompts per generation batch, and steps-per-generation remain unchanged.

All other scientific GRPO/MaxRL settings remain frozen.

## Interpretation

The original A40 seed42 result remains the canonical discovery result.

Within the A100 suite, seeds42/43/44 share the same GPU SKU, software stack,
single-GPU topology, and batch geometry. Therefore the A100 suite supplies the
clean three-seed robustness panel.

A100 seed42 also supplies a hardware/topology bridge to the original A40
seed42 result. The A40-vs-A100 seed42 difference must not be interpreted as a
pure hardware effect because hardware and distributed topology change
together.

## Engineering gate

Before launching the six full jobs:

1. full CPU tests must pass on the approved execution commit;
2. one A100 must pass the A100 runtime acceptance and FA2 model probe;
3. a 20-step GRPO seed43 single-A100 pilot must pass;
4. a 20-step MaxRL seed43 single-A100 pilot must pass;
5. both pilots must show the frozen runtime batch geometry and no OOM;
6. the MaxRL pilot must satisfy the existing finite-G advantage identity and
   ledger structural checks.

Do not silently change batch size, generation batch, vLLM memory fraction,
completion cap, objective settings, or optimizer settings in response to OOM or
performance issues. Any such change requires a new amendment before outcomes.

## Deadline evaluation policy

All six full runs still save the full 20-snapshot training schedule.

For the deadline-sensitive ATTRIB version, evaluate the 100% endpoint first.
The full 5%-through-100% C-bank trajectory is deferred until after submission.

The endpoint analysis must use the same frozen p0 baseline bank and the same
sequential K16 C-bank protocol as the canonical comparison.

Critically, evaluation sampling randomness is frozen to the canonical
evaluation bank and must not inherit training seed 43/44. Training seed and
evaluation sampling seed are separate quantities.

## Frozen scientific question

For each A100 seed s in {42,43,44}, compare matched GRPO and MaxRL:

1. H1: MaxRL reallocates realized training signal toward lower frozen-p0
   regions relative to GRPO.
2. H2: correctness movement does not show a stable corresponding reallocation.

Report per-seed endpoint results first. With only three seeds, descriptive
mean/spread is allowed; do not overstate precision, equivalence, or a formal
seed-variance estimate.
