# A100 no-checkpointing speed experiment rejected; revert to baseline

Date: 2026-09-05

## Status

The attempted A100 speed optimization is rejected as an engineering path before
any full replication run is launched.

This is not a scientific outcome.

## Attempts

Baseline single-A100 engineering configuration:

```text
gradient_checkpointing = True
vllm_gpu_memory_utilization = 0.30
```

completed the earlier 20-step pilot and used about 70 GB GPU memory.

Attempt 1:

```text
gradient_checkpointing = False
vllm_gpu_memory_utilization = 0.30
```

failed before completing step 1 on the first backward pass with a 9.27 GiB
allocation request and less than 1 GiB free.

Attempt 2:

```text
gradient_checkpointing = False
vllm_gpu_memory_utilization = 0.20
```

also failed during backward. With the default allocator it failed before
completing step 1. With `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
it completed one step (reported 17.31 s/it) but failed on the next backward
pass, again on a 9.27 GiB allocation request. At failure approximately 7.26 GiB
was free and only ~0.84 GiB was reserved but unallocated.

## Decision

Do not continue reducing the vLLM memory fraction merely to force
checkpointing off.

Reason:

1. the no-checkpointing path is still not robustly memory-safe at 0.20;
2. further reducing the colocated vLLM budget risks generation-side capacity /
   preemption and may reduce throughput;
3. the one completed no-checkpointing step did not show an obvious speed
   advantage over the checkpointing-enabled pilot;
4. the submission deadline makes further memory/throughput tuning lower value
   than launching the matched suite on the known-good configuration.

The active A100 replication execution contract therefore reverts to:

```text
gradient_checkpointing = True
vllm_gpu_memory_utilization = 0.30
world_size = 1
per_device_train_batch_size = 8
gradient_accumulation_steps = 4
global_optimizer_batch_size = 32
generation_batch_size = 32
steps_per_generation = 4
num_generations = 16
unique prompts per generation batch = 2
```

The prior no-checkpointing and vLLM-0.20 amendments remain in the repository as
provenance of rejected engineering attempts and are no longer active.

No full A100 replication outcome was produced under the rejected settings.
