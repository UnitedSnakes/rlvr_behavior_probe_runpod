# Practical MaxRL-15 implementation — pre-GPU pilot checkpoint

Date: 2026-09-03

## Status

The practical MaxRL estimator, trainer integration, entrypoint, provenance path, distributed group slicing, signal-ledger semantics, and TRL completion-table advantage logging have completed the CPU/TDD implementation gate.

**Decision:** the branch is ready for a disposable 20-step GPU engineering pilot after the RunPod checkout is updated to this checkpoint commit or later.

This is **not** a scientific MaxRL result and does **not** authorize interpreting behavioral outcomes. No MaxRL GPU outcome has been observed at the time this checkpoint is written.

The scientific hypothesis hierarchy remains frozen in:

- `docs/superpowers/specs/2026-09-03-maxrl-objective-intervention-amendment.md`

The exact estimator semantics remain frozen in:

- `docs/superpowers/specs/2026-09-03-maxrl-practical-estimator-implementation-amendment.md`

## Frozen estimator

The matched run keeps `G = N = 16` rollouts per prompt. Under the paper's dropped-baseline practical estimator, this corresponds to **practical MaxRL-15**, not MaxRL-16.

For a binary-reward group, define

```text
K = sum_i r_i
p_hat = K / 16
```

The implemented advantage is

```text
K = 0:  A_i = 0
K > 0:  A_i = (r_i - p_hat) / p_hat
```

Hence for `0 < K < 16`:

```text
successful rollout: A = (16 - K) / K
failed rollout:     A = -1
```

For `K = 16`, all advantages are zero. The scientific denominator epsilon is exactly `0.0`.

The practical estimator changes only the group advantage. The matched outer training stack remains the canonical GRPO stack, including `loss_type=dapo`, PPO clipping, token-level vLLM importance correction with `token_truncate`, and the same canonical sampling/configuration file.

## Implemented trainer composition

The implemented composition is deliberately:

```text
TRL GRPOTrainer
  -> PracticalMaxRLTrainer
  -> SignalLedgerGRPOTrainer
```

The same `MaxRLRewardBatchRecorder` instance is shared by:

1. the canonical terminated-and-correct reward function;
2. the MaxRL wrapper, which peeks without consuming;
3. the existing signal-ledger wrapper, which consumes and records the batch.

The MaxRL wrapper gathers the canonical binary rewards across processes, uses the same global contiguous `G`-group geometry as TRL 1.12, computes global practical-MaxRL advantages, and then slices the appropriate rank-local tensor.

The existing `controlled_run/signal_ledger.py` canonical GRPO implementation is not modified to contain a MaxRL branch.

## Instrumentation semantics

After the final instrumentation fix, the following three objects use the same practical-MaxRL advantage tensor:

```text
policy loss input
signal_ledger advantage field
TRL completion-table _logs["advantages"] latest global batch
```

This matters because TRL computes and appends its ordinary GRPO advantage before the wrapper is able to replace the returned tensor. The MaxRL wrapper now replaces exactly the latest global advantage-log batch after computing MaxRL advantages.

Other TRL diagnostics that describe the underlying reward groups can still retain their ordinary GRPO/reward-group semantics. In particular, metrics such as reward standard deviation or `frac_reward_zero_std` are diagnostics of reward-group structure; they must not be relabeled as MaxRL objective quantities merely because the policy-loss advantage has been replaced.

## Training entrypoint and provenance

The MaxRL entrypoint is:

```text
controlled_run/train_maxrl.py
```

It deliberately reuses:

```text
controlled_run/configs/grpo_qwen3_0_6b.yaml
```

There is no separately tuned MaxRL YAML. This prevents an accidental hyperparameter intervention from being conflated with the objective intervention.

The run writes:

```text
maxrl_run_manifest.json
```

with objective provenance including at least:

```text
objective_family: MaxRL
objective_intervention: replace_group_advantages_only
advantage_estimator: practical_maxrl
rollouts_per_prompt: 16
effective_maxrl_order: 15
all_failure_behavior: zero_group_gradient
maxrl_denominator_epsilon: 0.0
trainer_composition:
  - trl.GRPOTrainer
  - PracticalMaxRLTrainer
  - SignalLedgerGRPOTrainer
grouping_semantics: trl_global_reward_order_grouped_by_num_generations
```

The default output directory is `controlled_run_outputs/maxrl`.

## TDD / CPU verification record

The implementation was built in staged RED -> GREEN cycles rather than by first writing the production path.

### Practical estimator / wrapper

The pure estimator and composed trainer wrapper test, among other cases:

- `K=0` all zero;
- `K=16` all zero;
- `K=1` gives success `15`, failures `-1` for `G=16`;
- active mixed-group advantages sum to zero;
- non-binary/non-finite inputs fail closed;
- distributed global grouping is performed before rank slicing;
- the outer existing ledger records the post-replacement MaxRL advantage.

Trainer-composition GREEN commit:

```text
6b0543f2a50e546018a91d53a87f852b4ff319fc
```

### Shared controlled-training core

RED commit:

```text
77042b2291d087ef63bce206e8e1b9fe673af436
```

RED result:

```text
1 failed, 240 passed, 14 warnings
```

The sole expected failure was the missing `_run_controlled_grpo` hook.

GREEN commit:

```text
1080b935734f6e891c9785bf550c5181e04a1410
```

Fresh full-suite result:

```text
241 passed, 14 warnings
```

### MaxRL training entrypoint

RED commit:

```text
0deae8e0fc8405e6917c29713cc29c9dfdc0ea42
```

The RED run failed during collection only because `controlled_run.train_maxrl` did not yet exist.

GREEN commit:

```text
83553621777852eff0e23fc205c1d340bfe047e2
```

Fresh full-suite result:

```text
244 passed, 14 warnings
```

### TRL advantage-log semantic alignment

RED commit:

```text
b41dec8f5e82b8cd0fcf1a00e83ae8cb10fc0479
```

Fresh RED result:

```text
1 failed, 244 passed, 14 warnings
```

The sole failure showed the intended instrumentation mismatch: the loss/ledger used MaxRL while the fake TRL global completion-table log still contained the base GRPO advantages.

GREEN commit:

```text
160937a8ba697890b22863b789a1737d6201da16
```

Fresh GitHub Actions workflow/job:

```text
workflow: 33793353461
job:      100775015860
result:   245 passed, 14 warnings in 10.16s
```

This establishes the current CPU/code contract only.

## Frozen 20-step GPU pilot purpose

The first GPU run is an **engineering pilot**, not a mechanism-result run. Its job is to answer:

```text
Does the real 2-GPU TRL/vLLM path execute the frozen MaxRL estimator,
record the expected group identities, and remain numerically stable?
```

The 20-step pilot must not be used to tune the scientific estimator or to decide whether H2 or H3 is preferred.

A noisy 20-step difficulty-allocation pattern is not itself a pass/fail scientific result.

## Frozen 20-step launch geometry

Use the exact corrected canonical `pi0`, not any pilot/shakedown checkpoint and not the final GRPO model.

Expected runtime geometry remains:

```text
world_size = 2
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
global optimizer batch = 32
generation batch = 32
G = 16
unique prompts per generation batch = 2
steps_per_generation = 4
pilot max steps = 20
mode = pilot
scientific_use = false
```

The process must be launched through the repository's two-process distributed CUDA path so that `WORLD_SIZE=2`; a plain single-process Python invocation is not an acceptable substitute for this pilot.

The module entrypoint to invoke under that two-process launcher is:

```text
controlled_run.train_maxrl
```

with:

```text
--mode pilot
--pilot-steps 20
--pi0-dir <untouched corrected canonical pi0 directory>
--output-dir <new disposable MaxRL pilot directory>
```

Do not reuse an existing output directory with scientific artifacts.

## Frozen structural acceptance criteria for the 20-step pilot

The pilot passes the **engineering** gate only if all of the following hold.

### A. Manifest / lineage

`maxrl_run_manifest.json` must exist and report:

```text
mode = pilot
scientific_use = false
pilot_steps = 20
world_size = 2
num_generations = 16
generation_batch_size = 32
global_optimizer_batch_size = 32
steps_per_generation = 4
unique_prompts_per_generation_batch = 2
advantage_estimator = practical_maxrl
rollouts_per_prompt = 16
effective_maxrl_order = 15
all_failure_behavior = zero_group_gradient
maxrl_denominator_epsilon = 0.0
```

The `pi0_lineage_id` must equal the corrected canonical lineage:

```text
f89fc90226a67a6a3c7374f9c13abadfcecda88f397ab812fa4130f1f425605b
```

### B. Ledger structure

For an uninterrupted 20-step pilot, the expected total is:

```text
2 rank files
20 generation_global_step values: 0..19
32 rollout rows per step across both ranks
640 total rollout rows
2 prompt groups per step
40 total (generation_global_step, dataset_index) groups
16 rows per prompt group
```

Each row must contain finite/binary reward metadata and the existing signal-ledger schema.

### C. Exact MaxRL group identities

For every reconstructed `(generation_global_step, dataset_index)` group, define `K` from the 16 `canonical_reward` values.

The recorded `group_successes` must equal that `K` on every row.

The recorded advantage must satisfy, up to a small floating-point tolerance:

```text
K = 0:      every advantage = 0
K = 16:     every advantage = 0
0 < K < 16:
  reward = 1 -> advantage = (16 - K) / K
  reward = 0 -> advantage = -1
```

Every active mixed group must have advantage sum approximately zero.

Any systematic violation of these identities is a hard STOP, even if reward rises.

### D. Numerical/runtime sanity

The run must complete without:

```text
NaN/Inf loss
NaN/Inf gradient norm
CUDA/vLLM synchronization failure
rank desynchronization
group reconstruction failure
reward-recorder consumption failure
```

Token-level vLLM importance-sampling diagnostics must remain finite/non-degenerate. The pilot is not required to reproduce the canonical GRPO `ESS/N` numerically, but a collapse resembling the historical sequence-level-IS failure is a STOP.

### E. Diagnostic semantic consistency

For inspected pilot batches, the MaxRL advantages recorded by the signal ledger must agree with the practical-MaxRL identities above. The TRL completion-table advantage log is expected to use the same MaxRL values after the wrapper's latest-batch replacement.

Underlying reward-group metrics such as `frac_reward_zero_std` may still have ordinary reward-group semantics and are not expected to equal an invented MaxRL-specific metric.

## Explicit non-criteria for the 20-step pilot

The following are **not** required for engineering PASS:

- reward must increase over only 20 steps;
- completion length must already fall monotonically;
- low-`p0` signal allocation must already show a statistically or visually decisive left shift;
- `DeltaC` must move in any direction;
- H2 or H3 must appear supported.

Those are too outcome-sensitive/noisy for this disposable engineering gate.

## Next gate if 20-step pilot passes

Run a disposable 150-step MaxRL shakedown from a fresh untouched copy of the same corrected canonical `pi0`.

The shakedown should check both:

1. the same structural/finite-value identities as the 20-step pilot;
2. whether the realized signal-allocation intervention begins to behave qualitatively as intended before a full canonical trajectory is spent.

The H1 mechanism check remains conceptually prior to behavioral interpretation: if practical MaxRL-15 does not materially alter realized signal allocation relative to canonical GRPO, stop and inspect estimator/stack interactions before interpreting H2/H3.

Only after the shakedown is clean should a matched canonical MaxRL seed42 run be launched from an untouched corrected `pi0`.

## Scientific boundary

Current status is therefore:

```text
CPU/TDD implementation gate: PASS
20-step real GPU pilot:       NOT YET RUN
150-step MaxRL shakedown:     NOT YET RUN
canonical MaxRL seed42:       NOT YET RUN
MaxRL scientific outcome:     NONE
```

Do not convert the CPU pass into a scientific claim. Do not use pilot artifacts as canonical trajectory data.
