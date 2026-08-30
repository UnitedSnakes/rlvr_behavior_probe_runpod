# Controlled Qwen3 GRPO — Truncation-Policy and Reward Amendment

## Status

Adopted 2026-08-30. This amendment supersedes only the truncation-handling and
reward-definition parts of the frozen GRPO recipe. The 2048 completion cap
remains in force and is **not** raised by this amendment.

This document is the mandated design revisit required by
`2026-08-29-grpo-completion-cap-amendment.md`, which predeclared:

> If 2048 still produces a material clipped fraction, stop and revisit the
> scientific design rather than silently increasing the cap again; the prior
> amendment authorized only the single 1024 -> 2048 correction.

A 256-prompt measurement on the corrected `pi_0` shows a 51.12% clipped
fraction. The tripwire has fired. The response taken here is a change to the
truncation policy and reward definition, not another cap increase.

## Live evidence motivating the change

Measurement: `diagnose_p0_signal_budget.py`, corrected `pi_0`, GSM8K train
`[:256]` at revision `740312add88f781978c0658806c59bc2815b9866`, run as two
128-prompt shards on 1 x A40 each. Sampling matched the frozen GRPO recipe
exactly (`K=16`, `max_completion_length=2048`, `temperature=0.8`,
`top_p=0.95`, `top_k=0`, `repetition_penalty=1.0`, `seed=42`).

Merged over 256 prompts / 4096 rollouts:

```text
clipped_ratio:                2094/4096 = 51.12%
sample accuracy:                          53.88%
interior p0 (neither 0 nor 16):  220/256 = 85.94%
naturally stopped length p50/p75/p90/p99: 979 / 1353 / 1703 / 2015
```

Group state under the current `mask_truncated_completions=true` rule:

```text
live                144/256 = 56.25%
dead_all_correct     82/256 = 32.03%
dead_all_wrong       28/256 = 10.94%
dead_all_masked       2/256 =  0.78%
```

### Truncation is graded by difficulty

The decisive observation is not the aggregate clipped fraction but its
dependence on the pre-RL success probability:

```text
bin             n   mean p0    clip%    live%
0              16     0.000   77.74%    0.00%
(0, 1/4]       51     0.141   75.61%   68.63%
(1/4, 1/2]     57     0.402   62.06%   84.21%
(1/2, 3/4]     53     0.684   47.41%   62.26%
(3/4, 1)       59     0.874   27.12%   47.46%
1              20     1.000   16.88%    0.00%
```

Clip rate falls monotonically with `p0`, by a factor of about 4.6 from the
hardest to the easiest bin. The causal direction is straightforward: when the
policy cannot solve a problem it enters extended self-restart behavior, runs
long, and is truncated.

Under `mask_truncated_completions=true` this means the masking rule deletes
rollouts at a rate inversely proportional to `p0`. The deleted mass is
concentrated in exactly the low-`p0` regime that the planned GRPO-vs-MaxRL
objective comparison is designed to measure. Left unchanged, the experiment
would measure the truncation filter rather than the objective.

The rule additionally destroys within-group variance at high `p0`: for a
prompt with 14/16 correct rollouts, the two incorrect rollouts are the ones
most likely to be truncated, leaving an all-correct survivor set and zero
advantage. Of the 82 `dead_all_correct` groups, only 20 have `p0 = 1`; the
remaining 62 have genuine raw reward variance that masking removes.

### Offline counterfactual

`analyze_truncation_policy.py`, computed from the saved rollouts without
regeneration. Three truncation policies:

```text
A  mask_truncated_completions=true, reward = correctness      (current)
B  no masking, reward = terminated AND correct
C  no masking, reward = correctness as scored
```

```text
                                   live groups
A  mask truncated (current)      144/256 = 56.25%
B  no mask, reward 0 on truncate 224/256 = 87.50%
C  no mask, reward as scored     220/256 = 85.94%
```

Live rate by `p0` bin:

```text
bin             n       A%       B%       C%
0              16    0.00%    0.00%    0.00%
(0, 1/4]       51   68.63%   72.55%  100.00%
(1/4, 1/2]     57   84.21%  100.00%  100.00%
(1/2, 3/4]     53   62.26%  100.00%  100.00%
(3/4, 1)       59   47.46%  100.00%  100.00%
1              20    0.00%   90.00%    0.00%
```

Policy B removes the difficulty gradient entirely for every bin above 1/4 and
restores the high-`p0` variance that masking destroyed.

## Hypotheses ruled out by this measurement

1. **Missing Qwen3 thinking-control grammar explains the problem.** Ruled out.
   Free generation contains `<think>` in 43/4096 = 1.05% of rollouts and
   `</think>` in 50/4096 = 1.22%, yet sample accuracy is 53.88% and 85.94% of
   prompts have interior `p0`. The thinking grammar is effectively absent and
   affects none of the quantities this experiment measures. The
   thinking-transition investigation recorded in
   `checkpoints/2026-08-30-thinking-transition-handoff.md` is closed and must
   not be reopened as a blocker for GRPO.

2. **A longer horizon fixes truncation.** Ruled out by
   `horizon_curve_2048_4096_8192.json`: of 55 trajectories clipped at 2048,
   only 9 stopped naturally by 4096 and 11 by 8192. Naturally stopped
   completions have p90 = 1703 tokens, well inside the cap. The population is
   bimodal — either the policy converges within roughly 1700 tokens, or it
   enters a non-terminating self-restart mode. Extending the budget does not
   convert the second mode into the first.

## Amendment

For canonical GRPO, the GRPO engineering pilot, and any `p0` or checkpoint
sampling intended for comparison with canonical GRPO policies:

```text
mask_truncated_completions: false
reward: binary_terminated_final_answer_correctness
```

The reward becomes:

```text
r(x, z) = 1  if z terminated within max_completion_length
             and the extracted final numeric answer is correct
          0  otherwise
```

`Terminated` means generation ended because the policy emitted the assistant
terminal token `<|endoftext|>` (`151643`), that is, the sampler finish reason
is not `length`. Termination must be determined from the finish reason or
token ids, never from the decoded text: vLLM omits special tokens from text
output by default, and this already produced one misclassification in the
8192-clipped analysis.

All other scientific GRPO hyperparameters remain unchanged, including
`max_completion_length=2048`, `vllm_max_model_length=2560`, `G=16`,
`beta=0`, `epsilon=0.2`, `loss_type=dapo`, `scale_rewards=group`,
`learning_rate=1e-6`, `seed=42`, and the 2 x A40 batch semantics frozen in
`2026-08-29-grpo-2xa40-batch-semantics-amendment.md`.

This amendment does not change the SFT data bundle, the SFT recipe, or the
`pi_0` lineage.

### Why not policy C

Policy C requires only a configuration flag and reaches 85.94% live groups,
but it assigns reward 1 to trajectories that emit a correct answer and then
continue past the budget. It therefore applies no pressure toward
termination. Combined with `loss_type=dapo`, which deliberately omits
response-length normalization, and a starting clipped fraction above 50%,
C carries a material risk of length escalation during training. C is
acceptable only as a temporary engineering fallback and is not the canonical
policy.

### Cost of the change

631/4096 = 15.41% of rollouts are currently scored correct despite being
truncated. Under this amendment those rollouts receive reward 0. This is the
explicit, quantified price of the change and is accepted deliberately.

### Quantified residual

Policy B leaves 32/256 = 12.5% dead groups:

```text
16  p0 = 0            no correct rollout at K=16; genuine, not an artifact
 2  p0 = 1, no truncation   genuine saturation
14  (0, 1/4] bin      every correct rollout was truncated
```

The third category is 27.45% of the `(0, 1/4]` bin and is the only remaining
difficulty-graded loss of signal. It is recorded here rather than removed,
and it is the quantified justification for a future SFT data-recipe change if
one becomes necessary.

### Effect on the p0 distribution

Assigning reward 0 to truncated rollouts lowers effective `p0`. The bin
structure survives and the distribution moves toward the low and middle range
that the objective comparison targets:

```text
bin            raw n   policy B n
0                 16           30
(0, 1/4]          51           84
(1/4, 1/2]        57           46
(1/2, 3/4]        53           71
(3/4, 1)          59           23
1                 20            2

interior under B: 224/256 = 87.50%
```

Saturated prompts drop from 20 to 2, which increases rather than decreases the
usable evidence for the planned analysis.

## Scientific consequences

1. **The canonical objective is no longer pure final-answer correctness.** It
   is correctness within the generation budget. Every claim derived from the
   controlled run must state this. The alternative — keeping a
   correctness-only reward while silently deleting difficulty-correlated
   rollouts through masking — is the less honest of the two options, not the
   more conservative one.

2. **The Qwen2.5 pilot and the controlled Qwen3 run no longer optimize the
   same objective.** The pilot remains valid as external contrast and
   motivating evidence. Any side-by-side comparison must note the objective
   difference explicitly.

3. **Early GRPO will contain a termination-learning phase.** With over half of
   initial rollouts receiving reward 0 for truncation, early gradient will be
   dominated by learning to stop rather than learning to solve. The
   `p0`-to-learning-signal analysis must either anchor to checkpoints after
   this phase or measure the two effects separately. The 5% checkpoint density
   is sufficient to locate the transition.

## Required revalidation

Before canonical GRPO:

1. Update `GRPO_INVARIANTS`, `controlled_run/configs/grpo_qwen3_0_6b.yaml`,
   and the affected invariant tests. TRL 1.12 does not pass truncation status
   to custom reward functions, so the reward must be threaded through a
   `GRPOTrainer` subclass rather than inferred inside
   `gsm8k_binary_reward`.
2. Run the full controlled CPU test suite and observe a green
   `python -m pytest -q` on the live checkout.
3. Run a 20-step disposable engineering pilot from the untouched `pi_0` on the
   canonical 2 x A40 topology. Check only that reward has nonzero variance,
   loss and gradients remain finite, checkpoints reload, and memory is safe.
4. Run a 150-step disposable shakedown from the untouched `pi_0`. Acceptance
   is trend-based, not threshold-based:

```text
completions/clipped_ratio        must fall materially
completions/mean_length          must fall
reward mean                      must rise
frac_reward_zero_std             must stay low
```

5. If `clipped_ratio` does not fall materially across the shakedown, stop. The
   termination pressure is not effective, and the correct response is an SFT
   data-recipe change, not a further objective change.
6. Discard pilot and shakedown outputs as scientific data. Restart canonical
   GRPO from the untouched `pi_0`.

## Constraints

- Do not raise `max_completion_length` above 2048 under this amendment.
- Do not add a separate format reward, a think-tag requirement, or any other
  structural condition. The only condition added is termination.
- Do not change any SFT artifact, the frozen data bundle, or the `pi_0`
  lineage.
- Do not mix `p0` or evaluation results collected under
  `mask_truncated_completions=true` with post-amendment estimates. The
  256-prompt measurement above is the pre-amendment diagnostic that motivated
  this change and is retained as such.
- Do not treat a green test suite as evidence that the objective change works.
  Only the 150-step shakedown trend is that evidence.
