# Controlled Qwen3 RLVR — Truncation-Policy Handoff

Checkpoint date: 2026-08-30, afternoon. Supersedes
`2026-08-30-thinking-transition-handoff.md`.

Active branch: `controlled-qwen3-rlvr-task6`.

## Executive status

The thinking-transition investigation is **closed**. It was the wrong target.
A 256-prompt `p0` measurement showed the corrected `pi_0` reasons fine and
produces a good `p0` distribution; what it fails at is terminating.

The blocking issue is now truncation policy, and the fix is an objective
amendment rather than another SFT run. Canonical GRPO has not started.

Authoritative for the change:
`docs/superpowers/specs/2026-08-30-grpo-truncation-policy-amendment.md`.

## What closed the thinking-tag line

The forced-`<think>` intervention completed. Appending `<think>\n` to the
generation prompt normalized the first generated token completely — 128/128
became ordinary English openers (`Okay`, `Alright`) instead of the previous
multilingual fragments — and changed nothing downstream:

```text
                  stop        correct     closed think
unforced 8192   84/128       87/128       2/128
forced   8192   84/128       81/128       1/128
```

Identical stop rate. Entering the think phase does not produce the exit.
`<think>` and `</think>` are not on the causal path to anything measured
here, and the 256-prompt run confirmed it at scale (1.05% / 1.22% presence
against 53.88% sample accuracy).

Do not reopen this.

## The p0 measurement

`diagnose_p0_signal_budget.py`, corrected `pi_0`, GSM8K train `[:256]`,
`K=16`, cap 2048, GRPO-matched sampling, two 128-prompt shards on 1 x A40
each, ~13 min per shard.

```text
clipped_ratio                 2094/4096 = 51.12%
sample accuracy                           53.88%
interior p0                    220/256  = 85.94%
stopped length p50/p75/p90/p99  979 / 1353 / 1703 / 2015
```

Clip rate by `p0` bin — the finding:

```text
bin             n   mean p0    clip%
0              16     0.000   77.74%
(0, 1/4]       51     0.141   75.61%
(1/4, 1/2]     57     0.402   62.06%
(1/2, 3/4]     53     0.684   47.41%
(3/4, 1)       59     0.874   27.12%
1              20     1.000   16.88%
```

Truncation is graded by difficulty, about 4.6x from hardest to easiest bin.
With `mask_truncated_completions=true` this deletes rollouts in inverse
proportion to `p0` — exactly the axis the GRPO vs MaxRL comparison measures.

Raw outputs:

```text
controlled_run_outputs/sft_corrected/pi_0/p0_signal_budget/summary_shard{0,1}of2.json
controlled_run_outputs/sft_corrected/pi_0/p0_signal_budget/rollouts_shard{0,1}of2.jsonl
```

## The decision

`analyze_truncation_policy.py` computed the counterfactual offline from those
rollouts. Live-group rate: 56.25% under current masking, 87.50% under
"no masking, reward 0 on truncation", 85.94% under "no masking, reward as
scored". The middle option was adopted. Full evidence, cost (15.41% of
rollouts lose reward 1), residual (14 of 51 prompts in the `(0, 1/4]` bin
still die), and scientific consequences are in the amendment.

The `pi_0` reasoning behavior is good and is **not** being rebuilt. An SFT
data-recipe change remains the fallback if the shakedown below fails.

## Code state

Applied on top of `31420eb`:

- `controlled_run/rewards.py` — added `is_truncated_completion`,
  `make_gsm8k_terminated_binary_reward`, `resolve_terminal_token_ids`. No
  torch dependency, so the CPU lane still runs them. `gsm8k_binary_reward` is
  unchanged and still used by `sample_p0.py`.
- `controlled_run/train_grpo.py` — `reward_funcs` now built from the
  tokenizer's terminal ids.
- `controlled_run/config.py`, `configs/grpo_qwen3_0_6b.yaml` — `reward`
  renamed, `mask_truncated_completions: false`.
- `tests/` — 9 new reward tests; two hardcoded `is True` assertions updated
  in `test_controlled_run_config.py` and `test_controlled_run_grpo.py`.

Note on implementation: TRL 1.12 passes `completion_ids` to synchronous
custom reward functions (`grpo_trainer.py::_calculate_rewards`), so no
`GRPOTrainer` subclass is needed. The truncation predicate reuses TRL's own
rule, `ids[-1] not in [eos_token_id, pad_token_id]`.

`test_controlled_run_grpo.py` was edited but has not been run against the
live checkout. Run the full suite before the pilot.

## Next steps, in order

1. Full `python -m pytest -q` green on the live checkout.
2. 20-step disposable engineering pilot from untouched `pi_0`, 2 x A40.
   Check only: reward variance nonzero, loss and gradients finite,
   checkpoints reload, memory safe.
3. 150-step disposable shakedown from untouched `pi_0`. Acceptance is
   trend-based: `clipped_ratio` falls materially, `completions/mean_length`
   falls, reward mean rises, `frac_reward_zero_std` stays low.
4. **Gate.** If `clipped_ratio` does not fall, stop and switch to the SFT
   data-recipe path (filter OpenR1 to short traces). Do not raise the cap and
   do not add a format reward.
5. If the shakedown passes, freeze corrected `pi_0` as canonical and upload
   it to a canonical HF repo. It is currently only in the diagnostic repo
   `HKReporter/rlvr-behavior-probe-pi0-corrected-diagnostic-2026-08-30`.
6. K=1024 deep `p0` bank on the fixed 30-question subset, cap 2048,
   GRPO sampling, sharded across A40s.
7. Reachability-depth curve from that bank, compared against the Qwen2.5
   pilot curve in `analyses/reachability_depth.csv`.
8. Canonical one-epoch GRPO from untouched `pi_0`, 2 x A40.

## Open items not in the code

- `analyses/` (difficulty bins, reachability depth) exists only on this
  branch. It is not on `main`, which is what the public repo link shows.
- The corrected `pi_0` is not yet frozen as canonical. Naming still says
  `-corrected-diagnostic-`.
- The frozen SFT bundle is named `...canonical-sft-2026-08-28`. If step 4
  fails and the data recipe changes, that name becomes misleading.
