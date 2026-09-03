# MaxRL 20-step GPU pilot acceptance checker — complete

Date: 2026-09-03

## Status

The fail-closed CPU checker for the frozen disposable 20-step practical MaxRL-15 GPU pilot is implemented and has completed fresh RED -> GREEN cycles, including a post-implementation review correction to the distributed rank-layout assumption.

This checker is an engineering/runtime validation instrument. It does not inspect or gate on reward improvement, difficulty-bin left shift, `DeltaC`, or any other scientific outcome.

The scientific MaxRL hypothesis hierarchy remains frozen in:

```text
docs/superpowers/specs/2026-09-03-maxrl-objective-intervention-amendment.md
```

The practical estimator semantics remain frozen in:

```text
docs/superpowers/specs/2026-09-03-maxrl-practical-estimator-implementation-amendment.md
```

The broader pre-GPU implementation checkpoint is:

```text
docs/superpowers/checkpoints/2026-09-03-maxrl-implementation-ready-for-gpu-pilot.md
```

## Checker

Module:

```text
controlled_run/maxrl_pilot_acceptance.py
```

CLI:

```bash
python -m controlled_run.maxrl_pilot_acceptance \
  controlled_run_outputs/maxrl_pilot_20
```

A clean pilot prints a JSON report with:

```text
status = PASS
steps = 20
rows = 640
groups = 40
rank_files = 2
group_size = 16
max_advantage_error <= 1e-6
aggregate_token_is_ess_fraction finite and in (0, 1]
```

Any violated structural/provenance/advantage/IS invariant raises `ValueError` and the command exits nonzero.

## Frozen checks

### Manifest and lineage

The checker requires:

```text
mode = pilot
scientific_use = false
pilot_steps = 20
pi0_lineage_id = f89fc90226a67a6a3c7374f9c13abadfcecda88f397ab812fa4130f1f425605b
```

It requires the exact frozen runtime geometry:

```text
world_size = 2
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
global_optimizer_batch_size = 32
generation_batch_size = 32
steps_per_generation = 4
num_generations = 16
unique_prompts_per_generation_batch = 2
```

It also requires `vllm_importance_sampling_mode = token_truncate`.

### Objective provenance

The checker requires:

```text
objective_family = MaxRL
objective_intervention = replace_group_advantages_only
advantage_estimator = practical_maxrl
rollouts_per_prompt = 16
effective_maxrl_order = 15
all_failure_behavior = zero_group_gradient
maxrl_denominator_epsilon = 0.0
grouping_semantics = trl_global_reward_order_grouped_by_num_generations
```

and exact trainer composition:

```text
trl.GRPOTrainer
PracticalMaxRLTrainer
SignalLedgerGRPOTrainer
```

### Ledger structure

For exactly 20 steps, the checker requires:

```text
2 JSONL rank files
640 total rollout rows
steps exactly 0..19
32 rows per step
16 rows per rank per step
320 rows per rank
40 (generation_global_step, dataset_index) groups
16 rows per group
```

There is deliberately **no** requirement that every prompt group contain an `8/8` split across ranks. The current wrapper forms global contiguous `G`-sized groups before taking the rank-local contiguous slice. Under the frozen global generation batch `32`, `G=16`, and world size `2`, a prompt group can therefore be rank-local. The engineering invariant is the per-step rank load (`16` rows per rank), not a fabricated within-group rank ratio.

### Exact practical MaxRL-15 identities

For each global `G=16` prompt group, the checker reconstructs:

```text
K = sum(canonical_reward)
```

and verifies every row reports the same `group_successes = K`.

It then checks:

```text
K = 0 or K = 16:
  advantage = 0

0 < K < 16 and reward = 1:
  advantage = (16 - K) / K

0 < K < 16 and reward = 0:
  advantage = -1
```

Any error above `1e-6` is a hard failure.

### Token-level vLLM IS diagnostics

The checker requires positive finite `actual_is_ratio_count`, finite `actual_is_ratio_sum`, and positive finite `actual_is_ratio_sq_sum` on every rollout row.

It reconstructs the aggregate token-level effective sample-size fraction as:

```text
ESS/N = (sum rho)^2 / [N * sum rho^2]
```

and requires it to be finite and in `(0, 1]` up to numerical tolerance.

No minimum close-to-one threshold is imposed in the 20-step engineering checker. A scientific/mechanistic comparison to canonical GRPO is intentionally deferred.

## TDD record

### Structural checker

RED commit:

```text
6a6ac15a7a12685c101b307066b053ae8ce432b5
```

Fresh RED result:

```text
collection error: ModuleNotFoundError: controlled_run.maxrl_pilot_acceptance
```

The only missing object was the not-yet-implemented checker module.

GREEN commit:

```text
c49342b83d2b09cb85b9813dcfed0b50568b217a
```

Fresh GitHub Actions workflow:

```text
33799153646
```

Result: success.

### CLI

RED commit:

```text
2b4f295e3380de34b8fa09bf6e37bc842bb7e355
```

Fresh RED result:

```text
1 failed, 250 passed, 14 warnings
```

The sole failure was:

```text
AttributeError: module 'controlled_run.maxrl_pilot_acceptance' has no attribute 'main'
```

GREEN commit:

```text
401fef72451fe452149b2e1912a1497453a7625a
```

Fresh GitHub Actions verification:

```text
workflow: 33799611733
job:      100795564191
result:   251 passed, 14 warnings in 7.19s
```

### Post-implementation review: distributed rank-layout correction

A final review compared the acceptance checker against the existing distributed MaxRL wrapper test and the signal-ledger gather/slice implementation. It found one important checker bug: the first checker version assumed that every `G=16` prompt group must contain exactly eight rows from each rank.

That assumption was not part of the trainer contract. The wrapper gathers global rewards, constructs contiguous global `G` groups, and then takes a contiguous rank-local slice. The existing distributed trainer test already encodes this ordering. A correct pilot could therefore have been rejected by the old checker.

RED commit:

```text
3de9beff370d2888b990132d1487a3650235615f
```

Fresh RED result:

```text
4 failed, 248 passed, 14 warnings
```

The primary valid-layout test failed exactly on the obsolete message:

```text
MaxRL group (...) must contain 8 rows per rank
```

GREEN commit:

```text
4e48754f3832ddfafa49621c3e20bfc5da77653f
```

The corrected checker now requires the actual geometry-derived rank invariant:

```text
16 rollout rows per rank per generation step
```

and does not impose a within-group rank split.

Fresh GitHub Actions verification:

```text
workflow: 33800504256
job:      100798582485
result:   252 passed, 14 warnings in 10.73s
```

This is the current final CPU/code gate before the first real MaxRL GPU pilot.

## Next live command

On a fresh 2-GPU A40 RunPod checkout at `4e48754f3832ddfafa49621c3e20bfc5da77653f` or a later verified documentation-only descendant, first verify that the `--pi0-dir` points to the untouched corrected canonical pre-RL policy.

Then run a new disposable output directory:

```bash
rm -rf controlled_run_outputs/maxrl_pilot_20
set -o pipefail

torchrun --nproc_per_node=2 \
  -m controlled_run.train_maxrl \
  --pi0-dir controlled_run_outputs/sft/pi_0 \
  --output-dir controlled_run_outputs/maxrl_pilot_20 \
  --mode pilot \
  --pilot-steps 20 \
  2>&1 | tee controlled_run_outputs/maxrl_pilot_20.log
```

Immediately after training completes, run:

```bash
python -m controlled_run.maxrl_pilot_acceptance \
  controlled_run_outputs/maxrl_pilot_20
```

Do not proceed to the 150-step shakedown if the checker does not print `status: PASS`.

The pilot output remains disposable and must not be promoted into the canonical MaxRL lineage.
