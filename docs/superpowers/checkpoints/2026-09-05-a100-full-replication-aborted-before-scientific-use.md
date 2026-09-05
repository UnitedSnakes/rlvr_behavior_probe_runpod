# A100 full replication aborted after engineering qualification

Date: 2026-09-05

## Status

The six-run single-A100 replication suite is intentionally stopped before any
completed full-run scientific outcome is used.

This is a project-prioritization / experimental-design decision, not a failed
scientific replication.

## Engineering status before stop

The single-A100 execution path had already passed the frozen 20-step GRPO and
MaxRL engineering gates under:

```text
GPU SKU = A100 80GB
world_size = 1
per_device_train_batch_size = 8
gradient_accumulation_steps = 4
global_optimizer_batch_size = 32
generation_batch_size = 32
steps_per_generation = 4
num_generations = 16
unique prompts per generation batch = 2
gradient_checkpointing = True
vllm_gpu_memory_utilization = 0.30
```

The successful pilot gate is recorded in:

```text
docs/superpowers/checkpoints/2026-09-05-a100-20step-replication-pilot-pass.md
```

## Full-run launch history

The planned matched suite was:

```text
GPU0: GRPO  seed42
GPU1: MaxRL seed42
GPU2: GRPO  seed43
GPU3: MaxRL seed43
GPU4: GRPO  seed44
GPU5: MaxRL seed44
```

The first simultaneous launch attempt exposed an infrastructure-only port
collision because independent single-GPU vLLM processes inherited the same
default torch distributed TCPStore port 29500. Five jobs exited before training
while one job acquired the port and began training.

This was diagnosed as a launch-level namespace collision, not an NCCL failure
and not a scientific outcome. Unique per-job MASTER_PORT values are sufficient
for future independent single-GPU launches.

## Decision

Stop the A100 full replication effort now.

Reason:

1. the long-term preferred training topology for this project remains the
   original A40 path;
2. the cleanest multi-seed extension is therefore to add matched A40 seeds
   43/44 to the existing A40 seed42 discovery runs when suitable A40 capacity is
   available;
3. a complete A100 seed42/43/44 block would be scientifically usable, but would
   create a separate hardware/topology block requiring additional explanation;
4. these A100 full runs cannot materially improve the imminent ATTRIB
   submission before endpoint C-bank evaluation;
5. researcher time before submission is higher priority than producing an
   additional topology block.

## Scientific-use boundary

Any partial A100 full-run artifacts are non-scientific and must not enter
behavioral, signal-allocation, or replication claims.

They may be retained only as engineering / provenance artifacts.

The accepted scientific evidence for the current workshop submission remains
the matched A40 seed42 GRPO/MaxRL result bundle.

## Future replication plan

If the project continues on A40, run matched seed43 and seed44 GRPO/MaxRL pairs
using the original A40 execution contract.

If the project later deliberately migrates to A100, make that a new explicit
pre-outcome topology decision and run a complete internally matched A100 suite.

## Submission limitation language

Use:

> All behavioral and signal-allocation comparisons reported here are from a
> single matched training seed. We therefore do not estimate between-seed
> variability in either bin-level behavioral changes or objective-induced
> signal reallocation.
