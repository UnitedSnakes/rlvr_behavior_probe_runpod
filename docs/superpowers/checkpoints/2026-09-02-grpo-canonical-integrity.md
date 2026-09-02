# Controlled Qwen3 — Canonical GRPO structural integrity

## Status

Completed 2026-09-02. **STRUCTURAL INTEGRITY PASS.**

This checkpoint records only provenance, completeness, and numerical-integrity checks. No substantive canonical outcome curve had been inspected when this checkpoint was written.

## Provenance

```text
mode: canonical
scientific_use: true
pi0_lineage_id: f89fc90226a67a6a3c7374f9c13abadfcecda88f397ab812fa4130f1f425605b
```

The run starts from the untouched corrected canonical `pi_0` lineage.

Runtime batch semantics:

```text
world_size:                         2
per_device_train_batch_size:        4
gradient_accumulation_steps:        4
global_optimizer_batch_size:       32
generation_batch_size:              32
num_generations / G:                16
unique_prompts_per_generation:       2
TRL steps_per_generation:            4
```

## Canonical trajectory completeness

```text
max optimizer steps: 3736
ledger files:           2
ledger rows:       119552
ledger steps:         3736 (generation_global_step 0..3735)
ranks:                   [0, 1]
prompt groups:          7472
rows per generation step: 32.0
```

The ledger therefore contains exactly `3736 * 32 = 119552` rollout rows and `3736 * 2 = 7472` G=16 prompt groups, with no missing generation step.

## Policy snapshots

All frozen 5% policy snapshots are present and carry the same `pi0_lineage_id`:

```text
  5% -> step  187
 10% -> step  374
 15% -> step  560
 20% -> step  747
 25% -> step  934
 30% -> step 1121
 35% -> step 1308
 40% -> step 1494
 45% -> step 1681
 50% -> step 1868
 55% -> step 2055
 60% -> step 2242
 65% -> step 2428
 70% -> step 2615
 75% -> step 2802
 80% -> step 2989
 85% -> step 3176
 90% -> step 3362
 95% -> step 3549
100% -> step 3736
```

The final snapshot is saved after optimizer step 3736, while the final generation ledger index is 3735; this is the expected callback/index convention rather than a missing step.

## Numerical integrity

A recursive check over all numeric ledger fields found:

```text
non-finite numeric fields: 0
```

## Decision

**Canonical structural integrity: PASS.**

The run is admissible for the preregistered canonical analyses. This checkpoint does not claim any scientific result about reward dynamics, termination, correctness, difficulty allocation, or probability movement.
