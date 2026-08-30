# Controlled Qwen3 GRPO — Completion-Cap Amendment

## Status

Adopted after the live 20-step engineering pilot on 2026-08-29. This amendment exercises the contingency already specified in `2026-08-27-qwen3-controlled-rlvr-design-amendment.md`: if a material fraction of trajectories hits the 1024-token completion cap, raise the cap to 2048 as a documented engineering correction before canonical GRPO and restart from untouched `pi_0`.

## Pilot evidence

The disposable pilot used the exact frozen `pi_0` lineage and the canonical sampling/reward settings, but it was mistakenly launched with two A40s while the frozen compute policy specifies one A40 unless batch semantics are explicitly remapped. Therefore the pilot is not evidence about canonical optimizer-batch dynamics. It remains valid engineering evidence about response truncation because generation used the same model, prompt construction, decoding distribution, and 1024-token completion cap.

Observed diagnostics over the 20-step run included:

```text
completions/mean_length:       1005 -> 1016
completions/max_length:        1024 -> 1024
completions/clipped_ratio:     0.9531 -> 0.9750
max terminated length:         611.5 -> 475.9
reward mean:                   0.4562 -> 0.4344
frac_reward_zero_std:          0.05 -> 0.10
sampling logp diff mean:       0.01226 -> 0.008839
importance sampling mean:      0.9533 -> 0.9751
```

The 95–97.5% clipped ratio is unambiguously a material truncation fraction. Because `mask_truncated_completions=true`, most sampled trajectories are excluded from the policy loss, severely reducing useful learning signal and making the 1024 boundary an engineering confound.

## Amendment

For all canonical post-amendment GRPO and matching p0 / checkpoint sampling:

```text
max_prompt_tokens: 512
max_completion_length: 2048
vllm_max_model_length: 2560
mask_truncated_completions: true
```

All other scientific GRPO hyperparameters remain unchanged.

This is not performance tuning. It is the predeclared truncation correction required to prevent the hard generation boundary from dominating which trajectories contribute to the loss.

## Required revalidation

Before canonical GRPO:

1. Update the frozen GRPO config and invariant tests to `2048 / 2560`.
2. Run the full controlled CPU test suite.
3. Restart a fresh **one-A40** 20-step disposable pilot from the untouched canonical `pi_0`.
4. Inspect the same diagnostics, especially `completions/clipped_ratio` and terminated-length statistics.
5. If 2048 still produces a material clipped fraction, stop and revisit the scientific design rather than silently increasing the cap again; the prior amendment authorized only the single 1024 -> 2048 correction.
6. Any p0/evaluation result intended for comparison with the canonical GRPO policy must use the post-amendment 2048-token sampling cap. The earlier 30-question `K=16` preliminary p0 collected at the 1024 cap is retained only as a pre-amendment diagnostic and must not be mixed with canonical post-amendment p0 estimates.

## Compute semantics

The canonical GRPO batch semantics remain the single-device recipe:

```text
1 x A40
num_generations = 16
generation_batch_size = 32
per_device_train_batch_size = 8
gradient_accumulation_steps = 4
effective optimizer batch = 32 examples
```

The two-A40 20-step run that triggered this amendment is disposable engineering evidence only and its weights/results do not enter the canonical lineage.
