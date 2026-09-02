# Controlled Qwen3 GRPO — Top-p / Importance-Sampling Amendment

## Status

Adopted 2026-09-01. This amendment changes only the GRPO rollout sampling
parameter `top_p` from `0.95` to `1.0` for the controlled Qwen3 run. Canonical
GRPO remains **blocked** until the revalidation steps in this document pass.

The amendment is motivated by the preregistered signal-ledger GPU pilot in
`2026-09-01-signal-allocation-analysis-prereg.md`. It does not change the
reward, completion cap, batch semantics, optimizer, GRPO loss, reward scaling,
or `pi_0` lineage.

## Triggering evidence

A disposable 20-step pilot from the untouched canonical `pi_0` used the frozen
pre-amendment recipe:

```text
temperature: 0.8
top_p: 0.95
top_k: 0
repetition_penalty: 1.0
vllm_importance_sampling_correction: true
vllm_importance_sampling_mode: sequence_mask
vllm_importance_sampling_cap: 3.0
loss_type: dapo
G: 16
generation_batch_size: 32
world_size: 2
```

The signal ledger recorded 640/640 rollouts across 20 generation batches, 40
prompt groups, and two ranks without missing raw or post-mask IS quantities.

### Sequence-level IS degeneracy

The raw sequence importance log-ratio was strongly determined by completion
length:

```text
log rho ~ completion_length
slope/token:  -0.0184845
slope/1000:  -18.4845
R^2:           0.81903
correlation:  -0.90500
```

The post-mask sequence weights were highly concentrated:

```text
mean rho:      0.0033637
median rho:    6.57e-12
max rho:       0.60837
ESS:           6.18 / 640
ESS/N:         0.00965
rho < 1e-6:    75.47%
rho < 1e-3:    95.00%
upper-cap masked: 0 / 640
```

No rollout hit the upper cap. The pathology is therefore not caused by the
`clip_max=3` rejection rule; it is continuous lower-tail down-weighting.

### Mechanism: truncated behavior distribution vs full-policy log-probability

TRL v1.12 configures colocated vLLM with
`logprobs_mode="processed_logprobs"`. vLLM therefore reports sampled-token
log-probabilities after temperature/logit processing and top-p truncation,
including renormalization over the surviving nucleus. The trainer-side
recomputation, however, takes a full-vocabulary log-softmax and does not replay
the top-p mask.

For a sampled token that survives nucleus truncation, let `S_t` be the full
policy probability mass retained by the nucleus at position `t`. Then

```text
log q_vLLM(a_t) = log p_train(a_t) - log S_t
Delta_t = log p_train(a_t) - log q_vLLM(a_t) = log S_t <= 0
```

and sequence-level IS becomes

```text
rho_sequence = exp(sum_t Delta_t) = product_t S_t.
```

Thus a small systematic per-token mismatch compounds geometrically with
response length.

This mechanism is already documented upstream in Hugging Face TRL issue #6789,
"GRPO: vLLM importance-sampling ratio is biased when top_p/top_k/min_p truncate
sampling" (opened 2026-08-18). It is not claimed here as a novel discovery of
this project.

### Our pilot is consistent with that mechanism

The existing ledger stores the sequence sum rather than every token-level
`Delta_t`. The sequence-average discrepancy

```text
sequence_mean_logprob_diff = raw_log_rho / completion_length
```

was therefore used as an offline diagnostic. For `top_p=0.95`, the theoretical
pure-top-p interval is

```text
[log(0.95), 0] = [-0.0512933, 0].
```

Observed over all 640 rollouts:

```text
mean:     -0.0165095
std:       0.0037209
min:      -0.0276801
q05:      -0.0219994
median:   -0.0167720
q95:      -0.0102030
max:      -0.0023332
inside [log(.95), 0]: 640/640 = 100%
below log(.95):         0/640 =   0%
positive:                0/640 =   0%
```

Because this is a sequence-average statistic, it does not prove that every
individual token lies in the interval. Combined with the exact TRL/vLLM
log-probability semantics, however, the observed one-sided distribution is
consistent with the known top-p truncation mismatch and inconsistent with a
purely symmetric numerical-noise explanation.

## Difficulty association is mediated by response length in this pilot

The initial marginal table suggested lower-success `k/G` groups had more
negative `log rho`. The preregistered follow-up must not interpret that table
as an independent difficulty effect because response length already explains
81.9% of the variance.

A rollout-level joint regression was fit with one-way cluster-robust standard
errors at the generated prompt-group level `(generation_global_step,
dataset_index)`:

```text
log rho ~ completion_length + k/G
n rollouts:       640
n clusters:        40
R^2:            0.819030

length coefficient:       -0.0184808
cluster SE:                 0.0005039
t-like:                   -36.675

k/G coefficient:            0.0213822
cluster SE:                  1.6098575
t-like:                      0.0133
```

Conditional on completion length, the realized `k/G` association is
numerically negligible in this pilot. The earlier marginal gradient over
`k/G` is therefore treated as a response-length shadow, not evidence of a
direct difficulty-dependent IS effect.

This does **not** establish that difficulty can never affect backend mismatch;
`k/G` is itself a noisy within-batch realization. The later confirmatory
allocation analysis will use independently estimated train-side `p_0` as the
difficulty covariate.

## Amendment

For canonical GRPO, canonical-comparable train-side `p_0` sampling, and all
post-amendment engineering pilots:

```text
top_p: 1.0
```

The following remain unchanged:

```text
temperature: 0.8
top_k: 0
repetition_penalty: 1.0
vllm_importance_sampling_correction: true
vllm_importance_sampling_mode: sequence_mask
vllm_importance_sampling_cap: 3.0
mask_truncated_completions: false
reward: binary_terminated_final_answer_correctness
max_completion_length: 2048
loss_type: dapo
scale_rewards: group
learning_rate: 1e-6
seed: 42
```

### Why `top_p=1.0` is the primary correction

The scientific object is the learning dynamics of the policy `pi`, not a
separately truncated nucleus behavior policy `q`. With `top_p=0.95`, rollouts
come from `q != pi`; the current TRL/vLLM stack then compares a processed
nucleus log-probability against a full-policy training log-probability.

Setting `top_p=1.0` removes the nucleus truncation itself and returns the rollout
behavior distribution to the full temperature-scaled policy that the trainer
recomputes. It also matches the TRL default for `top_p`.

The purpose is not to tune IS statistics after seeing the pilot. It is to
remove an identified mismatch between the sampled behavior distribution and
the policy distribution that the experiment intends to study.

### Why not make `token_truncate` canonical now

TRL v1.12 also offers token-level IS modes. Under the observed scale, a
per-token top-p ratio is near one and the configured upper cap of `3.0` would
rarely or never activate. With DAPO global token normalization, much of a
nearly common token-level scaling can be absorbed by Adam's scale invariance.

More importantly, switching from sequence-level to token-level ratios does not
replay the missing top-p support restriction on the trainer side and is not an
exact correction for trajectory/state-visitation mismatch. It is therefore a
reasonable engineering comparator, not the preferred canonical definition.

### Why not disable IS

`vllm_importance_sampling_correction=false` would remove the pathological
weighting but would also discard correction for genuine residual vLLM-versus-
training-model log-probability discrepancies. The pilot was designed to
measure those discrepancies rather than assume they are absent. IS therefore
remains enabled while the known top-p confound is removed.

## Claim boundary

The pilot directly establishes that the **sequence weights** are strongly
length-determined and highly degenerate. It does not yet establish the same
magnitude of distortion in parameter updates or behavior-probability movement,
because advantages, DAPO token normalization, PPO clipping, optimizer state,
and shared-parameter effects intervene downstream.

Allowed wording before canonical outcome data:

> Under `top_p=0.95`, the resulting sequence IS weights are strongly
> length-determined (`R^2=0.82`) and highly degenerate (`ESS/N≈0.01`), a
> pipeline effect that can distort the intended difficulty allocation.

Do not state that the observed IS weights have already been proven to cause a
specific `p_0 -> Delta p` distortion.

## Required revalidation before canonical GRPO

1. Update the frozen GRPO invariant and YAML config to `top_p=1.0`, with an
   invariant regression test that rejects a return to `0.95` without another
   amendment.
2. Run the complete CPU suite and require zero failures.
3. From the untouched canonical `pi_0`, run a disposable short 2 x A40 pilot
   with all settings identical except `top_p=1.0`.
4. Recompute the signal-ledger diagnostics. The pilot is accepted only if the
   specific top-p pathology is removed: the sequence-average discrepancy is no
   longer confined to the old `[log(.95), 0]` band, the sequence-weight ESS is
   no longer collapsed by deterministic length compounding, and the strong
   old `log rho ~ completion_length` relationship is materially reduced.
   No arbitrary numerical cutoff is introduced after observing the first
   pilot.
5. Inspect residual raw `log rho` before deciding whether a token-level IS
   comparator is still scientifically useful. If substantial residual
   structure remains after `top_p=1.0`, stop and diagnose it rather than
   silently switching IS mode.
6. Only after this revalidation may canonical GRPO be launched.

## Consequences for previously collected probability banks

Any train-side `p_0` bank intended to stratify canonical GRPO allocation must
match the amended rollout distribution and the canonical terminated-and-correct
reward. A bank collected with `top_p=0.95` is not objective-and-sampling matched
for confirmatory train allocation analysis and must not be substituted for the
post-amendment bank.

The fixed GSM8K TEST deep bank remains useful as an earlier reachability and
transfer reference, subject to its already-recorded correctness-only reward
mismatch. It is not a direct train-allocation baseline.
