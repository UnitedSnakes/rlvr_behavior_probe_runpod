# A100 replication memory amendment: vLLM utilization 0.20

Date: 2026-09-05

## Status

This is a pre-outcome engineering amendment to the active single-A100
replication suite.

The first 20-step GRPO seed43 pilot with
`gradient_checkpointing=False` and
`vllm_gpu_memory_utilization=0.30` failed on the first backward pass with a
CUDA OOM before producing any training outcome.

Observed failure:

```text
A100 total capacity: 79.25 GiB
free at failure: ~0.97 GiB
PyTorch allocated: ~74.24 GiB
PyTorch reserved but unallocated: ~3.41 GiB
requested allocation: 9.27 GiB
```

The failure occurred before step 1 completed.

## Change

Keep gradient checkpointing disabled, and reduce only the colocated vLLM memory
reservation:

```text
vllm_gpu_memory_utilization: 0.30 -> 0.20
```

For an 80GB-class GPU this nominally returns about 8GB of GPU-memory budget
from vLLM to the training process, which is materially larger than the
remaining shortfall implied by the failed allocation.

## Frozen A100 execution contract

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
gradient_checkpointing = False
vllm_gpu_memory_utilization = 0.20
```

All reward, objective, sampling, completion length, optimizer, scheduler,
dataset, pi0, importance-sampling, and finite-G settings remain unchanged.

## Acceptance gate

Rerun disposable 20-step GRPO seed43 and MaxRL seed43 pilots.

Accept this configuration only if:

1. both pilots complete without OOM;
2. runtime batch geometry remains frozen;
3. MaxRL finite-G advantage identity and ledger checks pass;
4. vLLM reports no memory-capacity failure or pathological preemption;
5. steady-state timing is compared against the checkpointing-enabled baseline
   using engineering timing only, not reward or behavioral outcomes.

If 0.20 still OOMs, do not silently alter other scientific settings. Record
the failure and make another pre-outcome amendment.
