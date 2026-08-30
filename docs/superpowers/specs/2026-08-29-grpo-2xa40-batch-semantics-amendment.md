# Controlled Qwen3 GRPO — 2×A40 Batch-Semantics Amendment

## Status

Adopted after the post-1024-cap correction exposed a single-A40 OOM at `max_completion_length=2048`. This amendment supersedes only the GRPO compute/batch section of the earlier design amendments. The 2048 completion-cap amendment remains in force.

## Live evidence motivating the change

The first one-A40 2048-token pilot failed before completing step 1 with CUDA OOM while computing old-policy token log-probabilities. PyTorch attempted an additional ~9.28 GiB allocation with only ~6.63 GiB free on the 44.42 GiB A40. This makes the post-amendment 2048 recipe infeasible on one A40 without another engineering change.

An earlier two-A40 1024-token pilot did run, but it accidentally kept `per_device_train_batch_size=8`. That run is not canonical batch evidence because it changed the training semantics:

```text
world_size = 2
per_device_train_batch_size = 8
gradient_accumulation_steps = 4
=> effective optimizer batch = 64

generation_batch_size = 32
per-step global batch = 16
=> steps_per_generation = 2
```

The intended single-A40 design had:

```text
world_size = 1
per_device_train_batch_size = 8
gradient_accumulation_steps = 4
=> effective optimizer batch = 32

generation_batch_size = 32
per-step global batch = 8
=> steps_per_generation = 4

num_generations = 16
=> 2 unique prompts per generation batch
```

The accidental two-A40 pilot's final `epoch=0.01071` after 20 optimizer steps is consistent with consuming about twice the intended unique-prompt progress. It is retained as engineering evidence only.

## TRL 1.12 semantics audit

The runtime is pinned to TRL 1.12.0. Its source defines the relevant semantics as follows:

- `num_generations` must divide the effective training batch `num_processes * per_device_train_batch_size * gradient_accumulation_steps`.
- An explicit `generation_batch_size` is global. TRL requires it to be divisible by `per_device_train_batch_size * num_processes` and derives `steps_per_generation` from that quotient.
- The GRPO sampler repeats each prompt so complete generation groups are distributed across processes; rewards and advantages are normalized across the full prompt group.
- In colocated vLLM mode with `vllm_tensor_parallel_size=1`, each training process/GPU holds its own vLLM copy and generates for its local prompt slice.

Audited upstream source:

```text
huggingface/trl tag v1.12.0
trl/trainer/grpo_config.py
trl/trainer/grpo_trainer.py
trl/generation/vllm_generation.py
```

## Exact 2×A40 remap

The canonical post-amendment mapping is:

```text
canonical_world_size = 2
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
global_optimizer_batch_size = 32

generation_batch_size = 32
num_generations = 16
steps_per_generation = 4
unique prompts per generation batch = 2

max_prompt_tokens = 512
max_completion_length = 2048
vllm_max_model_length = 2560
```

This preserves the intended single-device scientific semantics exactly at the batch/group level:

```text
old intended: 8 × 1 × 4 = 32
new canonical: 4 × 2 × 4 = 32
```

and:

```text
old intended steps_per_generation = 32 / (8 × 1) = 4
new canonical steps_per_generation = 32 / (4 × 2) = 4
```

`G=16` and `generation_batch_size=32` still yield exactly two independent prompt groups per generation batch.

The only intended change is how the same global work is partitioned across two A40 devices so the 2048-token forward/log-prob computation fits in memory.

## Runtime guard

The controlled GRPO runner must fail closed unless all of these hold:

```text
runtime world_size == 2
per_device_train_batch_size == 4
gradient_accumulation_steps == 4
global optimizer batch == 32
generation_batch_size == 32
steps_per_generation == 4
num_generations == 16
unique prompts per generation batch == 2
```

The YAML should record `canonical_world_size: 2` and `global_optimizer_batch_size: 32` as semantic provenance fields. They are controlled-run metadata and are not forwarded as unsupported constructor arguments to `trl.GRPOConfig`.

The runtime guard should derive `WORLD_SIZE` from the launched distributed environment. Running the canonical/pilot GRPO entry point with ordinary `python` or with a world size other than two must fail before model training begins.

## Pilot revalidation gate

After code/CI verification, rerun a fresh disposable 20-step pilot from untouched `pi_0` with:

```bash
torchrun --nproc_per_node=2 -m controlled_run.train_grpo ... --mode pilot --pilot-steps 20
```

Acceptance requires:

1. no CUDA OOM;
2. manifest records the exact runtime batch mapping above;
3. 20-step final epoch progress is approximately the intended single-device rate (`~0.00535`, allowing ordinary Trainer rounding/logging differences);
4. `completions/clipped_ratio` is inspected under the new 2048 cap;
5. `frac_reward_zero_std`, reward, entropy, grad norm, policy clipping, sampling-logp difference, and importance-sampling ratios remain numerically sane;
6. pilot weights remain disposable and canonical GRPO later restarts from untouched `pi_0`.

If per-device batch 4 still OOMs, stop and make a second documented compute amendment. Do not silently change to per-device 2 / accumulation 8, vLLM memory utilization, sleep mode, tensor parallelism, or another GPU class without auditing how that change affects the frozen semantics.

## Evaluation consistency

The 2048 completion cap is now part of the canonical sampling policy. Therefore p0 and checkpoint evaluation intended for direct `p0 -> pt` comparison must use the post-amendment 2048 cap. The existing 30-question K=16 preliminary p0 measured at 1024 remains diagnostic only and must not be mixed into canonical post-amendment probability estimates.
