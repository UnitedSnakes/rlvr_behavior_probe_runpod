# A100 replication speed amendment: disable gradient checkpointing

Date: 2026-09-05

## Status

This is a pre-outcome engineering amendment to the active six-run single-A100
replication plan. It is written before any full A100 replication outcome is
produced or inspected.

The currently running 20-step checkpointing-enabled pilots are engineering
baselines only and remain disposable.

## Change

For the single-A100 replication suite only:

```text
gradient_checkpointing = False
```

The original A40 canonical runs and their configuration remain unchanged.

## Rationale

Each single-A100 pilot used approximately 70 GB of an 80 GB A100 while
`gradient_checkpointing=True`. The A100 suite is now a complete matched
three-seed rerun (GRPO and MaxRL for seeds 42, 43, and 44), so its internal
execution contract may use an A100-specific implementation setting when the
scientific semantics remain unchanged across all six runs.

Disabling gradient checkpointing trades additional activation memory for less
backward recomputation. It is adopted only if a fresh 20-step GRPO and MaxRL
engineering pilot fits in memory and passes the existing structural gates.

## Frozen A100 suite after this amendment

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
vllm_gpu_memory_utilization = 0.30
```

All reward, objective, sampling, completion-length, optimizer, scheduler,
dataset, pi0, importance-sampling, and finite-G settings remain unchanged.

## Acceptance gate

Before launching any full A100 replication run:

1. full CPU tests pass on the approved execution commit;
2. A100 runtime acceptance and FA2 probe pass;
3. rerun a 20-step GRPO seed43 pilot with checkpointing disabled;
4. rerun a 20-step MaxRL seed43 pilot with checkpointing disabled;
5. neither pilot OOMs;
6. runtime batch geometry remains exactly frozen;
7. MaxRL finite-G advantage identity and ledger checks pass;
8. compare steady-state timing against the checkpointing-enabled engineering
   baseline without using reward or behavioral outcomes to choose the setting.

If disabling checkpointing OOMs or fails a structural gate, do not change any
other scientific setting silently. Add another pre-outcome amendment before
trying a different memory allocation.

## Interpretation

Gradient checkpointing is treated as an execution/memory implementation choice,
not as part of the scientific objective. All six A100 runs must use the same
setting. The original 2xA40 seed42 result remains an external discovery /
robustness reference and is not claimed to be execution-identical to the A100
suite.
