# Controlled Qwen3 RLVR — Long-Context SFT and Compute Topology Amendment

## Status

This amendment is authoritative where it differs from the earlier 2026-08-27 SFT amendment and checkpoint notes.

The replacement SFT length cutoff has now been frozen at **16,384 formatted tokens** from the corrected extended live audit described below.

## Evidence forcing the amendment

The corrected Qwen3 tokenizer audit over the pinned OpenR1 source produced 64,968 verified, contamination-screened candidates before length filtering, with:

- p50 = 4,974 tokens
- p75 = 8,096 tokens
- p90 = 11,843.6 tokens
- p95 = 14,128.65 tokens
- p99 = 16,876 tokens
- p99.5 = 17,496.165 tokens
- p99.9 = 18,946.089 tokens
- maximum = 22,319 tokens
- fraction >2,048 = 0.8967337766
- fraction >4,096 = 0.6033431843
- fraction >8,192 = 0.2441355744
- fraction >12,288 = 0.0889053072
- fraction >16,384 = 0.0163003325
- fraction >32,768 = 0.0

Therefore the previous 2,048-token SFT cutoff is rejected: it removes nearly 90% of otherwise eligible verified traces and creates severe short-trace selection pressure before RL.

## Frozen length-selection rule

Canonical SFT uses:

```text
max formatted tokens = 16,384
SFTConfig max_length = 16,384
```

The data-selection cutoff and trainer max length must remain identical and are both validated as canonical invariants.

The alternative 12,288-token cutoff is rejected for the first canonical run because it would remove about 8.89% of the contamination-clean verified distribution, whereas 16,384 removes only about 1.63%. The latter preserves almost the entire empirical reasoning-trace distribution while still cutting the extreme tail.

The longest observed formatted candidate is 22,319 tokens and no candidate exceeds 32,768 tokens. Qwen/Qwen3-0.6B-Base has a 32,768-token native context window, so the frozen 16,384 cutoff remains comfortably inside the model context limit.

The pinned audit predicts approximately 63,909 eligible verified candidates after the 16,384 filter, far above the required 10,512 train+validation examples.

No canonical SFT may run from an audit-only or stale data materialization. The canonical SFT runner must verify the hashed data bundle and confirm that its recorded `max_formatted_tokens` equals the configured 16,384 cutoff before loading the model.

## SFT scientific batch invariant

The scientific invariant is the global optimizer batch size, not a single-device layout:

- `global_batch_size = 64`
- canonical SFT world size = 2 A40 GPUs
- ordinary data-parallel replication, not context parallelism
- canonical hardware layout: `per_device_train_batch_size = 1`, `gradient_accumulation_steps = 32`, `world_size = 2`

This preserves `1 × 2 × 32 = 64` while allowing long-context sequences to fit comfortably. A one-GPU engineering smoke may use the same per-device/accumulation values even though its effective smoke batch is not scientific evidence.

Context parallelism is deliberately excluded because the canonical SFT retains BFD packing, and TRL documents context parallelism as incompatible with packing.

## Distributed save semantics

Canonical two-GPU SFT must be filesystem-safe:

1. all ranks train;
2. synchronize after training;
3. world process zero alone writes the run manifest and freezes `pi_0`;
4. synchronize after the save;
5. all ranks may verify/read the completed `pi_0` manifest, but nonzero ranks must not create or mutate canonical output files.

The recorded run manifest must include world size, per-device batch size, gradient accumulation steps, and computed global batch size.

## GRPO topology

The first GRPO pilot and first canonical GRPO remain single-A40 runs. The existing `G=16`, generation batch 32, and effective optimizer batch 32 semantics remain unchanged.

Multi-GPU GRPO is not part of this amendment because it would require a separate remapping and validation of generation/update semantics and colocated vLLM behavior.

## Rollout and evaluation topology

Large independent sampling jobs may use up to four A40 GPUs as deterministic shards. Each GPU runs an independent sample-ID/seed range, and outputs are merged deterministically. This is preferred over distributed training when the workload is embarrassingly parallel.

## Constraints

- Do not change pinned model/dataset revisions, contamination semantics, source-index identity, SFT subset size, validation size, epochs, learning rate, or optimizer in this amendment.
- Do not change the GRPO recipe.
- Do not silently disable packing to enable context parallelism.
- Do not run canonical SFT from any data bundle whose audit cutoff differs from 16,384.
- Do not silently change `flash_attention_2`; runtime compatibility is a separate explicit acceptance gate.
