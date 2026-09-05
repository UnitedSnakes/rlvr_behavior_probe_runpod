# Single-A100 GRPO/MaxRL 20-step replication pilot gate PASS

Date: 2026-09-05

## Status

The active single-A100 replication execution contract has passed the 20-step
engineering gate for both GRPO and practical MaxRL before any full replication
run is launched.

These pilot outputs are engineering evidence only and have
`scientific_use=false`.

Active execution contract:

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

The corrected pi0 lineage is:

```text
f89fc90226a67a6a3c7374f9c13abadfcecda88f397ab812fa4130f1f425605b
```

## GRPO seed43 engineering pilot

Observed training summary:

```text
20/20 steps complete
train_runtime = 315.2 s
mean progress timing = 15.76 s/it
runtime batch geometry = PASS
ledger files = 1
ledger rows = 640
generation steps = 20
prompt groups = 40
rank set = {0}
```

The manual fail-closed GRPO ledger structure check passed:

```text
status = PASS
files = 1
rows = 640
steps = 20
groups = 40
ranks = {0: 640}
```

## MaxRL seed43 engineering pilot

Observed training summary:

```text
20/20 steps complete
train_runtime = 317.7 s
mean progress timing = 15.89 s/it
runtime batch geometry = PASS
```

The single-A100 MaxRL acceptance profile reported:

```json
{
  "aggregate_token_is_ess_fraction": 0.9979681011405994,
  "group_size": 16,
  "groups": 40,
  "max_advantage_error": 4.7683715642676816e-08,
  "rank_files": 1,
  "rows": 640,
  "status": "PASS",
  "steps": 20
}
```

Therefore the practical MaxRL finite-G advantage identity and token-level IS
diagnostics are structurally healthy on the active single-A100 path.

## Rejected speed experiment

The attempted no-checkpointing path remains rejected. The active configuration
is the known-good checkpointed / vLLM-0.30 baseline.

## Test-suite note

After adding the single-A100 MaxRL acceptance profile, one existing A40 unit
test failed only because its expected exception-message substring changed.
The fail-closed behavior itself was correct. The message compatibility was
restored in commit:

```text
8a62690ed89ca67fbe9af21febbba80f463c3d7c
fix: preserve rank-count acceptance message
```

A fresh full pytest run is required on this commit before launching full
replication.

## Authorization

If the full pytest suite passes on the approved execution commit, authorize the
six full single-A100 runs:

```text
GPU0: GRPO  seed42
GPU1: MaxRL seed42
GPU2: GRPO  seed43
GPU3: MaxRL seed43
GPU4: GRPO  seed44
GPU5: MaxRL seed44
```

No further throughput tuning is authorized before these runs.
