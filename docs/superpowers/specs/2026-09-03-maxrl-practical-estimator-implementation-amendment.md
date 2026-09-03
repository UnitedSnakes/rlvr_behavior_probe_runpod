# MaxRL practical-estimator implementation amendment

Date: 2026-09-03

## Status

This is an implementation-specific amendment written **before any MaxRL training outcome is observed**. It refines `2026-09-03-maxrl-objective-intervention-amendment.md` after checking the current MaxRL paper and official implementation.

This document freezes the estimator semantics that must be implemented and tested before any MaxRL GPU run. It does not alter the already-completed canonical GRPO run.

## Source of truth

Primary theoretical source:

- Tajwar et al., *Maximum Likelihood Reinforcement Learning*, arXiv:2602.02710v3, Sections 4.1--4.3 and Appendix D, especially Theorem 2 / Algorithm 1 / Theorem 5.

Reference implementation checked:

- `tajwarfahim/maxrl`
- source commit inspected: `7197bbb46a2ecd866da52f6b401ff20a34fe9390`
- dedicated `AdvantageEstimator.MAXRL` computes group-mean reward normalization rather than GRPO reward-std normalization.
- the public Qwen3 experiment uses 16 rollouts per prompt and the dedicated `maxrl` advantage estimator.

The paper, not a convenience approximation, determines the scientific semantics below.

## Correction: practical N-rollout MaxRL is effectively order T = N - 1

The unbaselined estimator in Eq. (9), for `N` rollouts and `K` successes,

```text
(1/K) * sum_i r_i S_i   if K > 0
0                       if K = 0
```

is unbiased for the `T = N` truncated MaxRL gradient (Theorem 2).

Algorithm 1 uses a variance-reduced centered estimator and, in the all-failure case `K = 0`, drops both the success term and the unconditional-score baseline. The paper's Theorem 5 shows that this practical dropped-baseline estimator is unbiased for **order `T = N - 1`**, not `T = N`.

Therefore, under the controlled experiment's frozen `G = N = 16` rollouts per prompt, the practical estimator implemented here is:

```text
practical MaxRL-15
```

Do not call it MaxRL-16 in analysis, manifests, or paper-facing text. The matched intervention changes no rollout-count hyperparameter.

Its population weight function is

```text
w_15(p) = sum_{t=1}^{15} (1-p)^(t-1)
        = [1 - (1-p)^15] / p
```

for `p > 0`, with the continuous limit `w_15(0) = 15`.

## Frozen per-group advantage

For a binary-reward group of size `G = 16`, let

```text
K = sum_i r_i
p_hat = K / G
```

The practical MaxRL advantage is frozen as:

```text
A_i = 0                              if K = 0
A_i = (r_i - p_hat) / p_hat          if K > 0
```

Equivalently, for `0 < K < G`:

```text
success rollout: A_success = (G - K) / K
failure rollout: A_failure = -1
```

and for `K = G`, all advantages are zero.

For `K > 0`, the group-average score-weighted update is

```text
(1/G) * sum_i A_i S_i
= (1/K) * sum_{i: r_i=1} S_i - (1/G) * sum_i S_i.
```

The `K = 0` branch deliberately sets the entire group to zero, matching Algorithm 1's dropped-baseline convention and yielding the Theorem-5 `T=N-1` objective.

## No epsilon in the scientific estimator

The paper's Algorithm 1 has an explicit `p_hat > 0` branch, so division by zero is avoidable without numerical smoothing. The controlled implementation will therefore use the exact branch above and **will not add epsilon to `p_hat` when `K > 0`**.

The public reference code uses a small epsilon in the denominator for engineering convenience. With `G=16`, any positive empirical mean is at least `1/16`, so epsilon is unnecessary here and would introduce a small avoidable deviation from the exact estimator proven in Theorem 5.

This deliberate difference from the public code must remain documented.

## Matched outer training stack

Only the group advantage normalization is an objective intervention. The following canonical GRPO settings remain unchanged:

```text
exact corrected pi0 lineage
G = 16
generation batch = 32
global optimizer batch = 32
world size = 2
one training epoch
temperature = 0.8
top_p = 1.0
binary terminated-and-correct reward
learning rate = 1e-6
num_iterations = 1
PPO clip epsilon = 0.2
beta = 0
loss_type = dapo
vLLM IS correction = token_truncate
vLLM IS cap = 3.0
seed/data seed = 42 for the first matched pair
snapshot schedule = 5% increments
```

TRL may internally compute its ordinary GRPO group-standardized advantage first. In MaxRL mode, the wrapper must replace that tensor with the frozen practical-MaxRL advantage **before the tensor is consumed by the policy loss**. The canonical GRPO path must remain byte-for-byte behaviorally unchanged when MaxRL mode is not requested.

## DAPO / token-normalization boundary

The MaxRL theorem concerns the score-function estimator before the surrounding LLM training stack's token-level loss aggregation, clipping, and off-policy correction.

The controlled experiment intentionally keeps the same DAPO token-normalized policy loss and the same vLLM token-level importance-sampling correction used in the canonical GRPO run. The official MaxRL Qwen3 implementation also uses token-mean loss aggregation, so preserving token-level aggregation is not an ad hoc change introduced only for this comparison.

Nevertheless, the resulting realized optimizer update must **not** be described as a naked unbiased estimator of the sequence-level `J_MaxRL^(15)` theorem. DAPO token normalization, PPO clipping, finite completion lengths, and vLLM IS correction can transform realized gradient mass.

This is why the objective-comparison protocol treats the realized signal ledger as a primary mechanism measurement rather than assuming nominal weighting survives the stack unchanged.

## Implementation boundary

Prefer a minimal wrapper around the existing TRL path rather than copying TRL's generation/loss implementation.

The implementation must provide a pure CPU-testable helper for practical MaxRL group advantages and use it to overwrite the generated batch's `advantages` tensor in MaxRL mode after global reward groups are known and before the policy loss consumes the batch.

The existing signal ledger must record the **post-replacement MaxRL advantage**, not the temporary GRPO advantage produced by the base trainer.

The run manifest must record at minimum:

```text
advantage_estimator: practical_maxrl
rollouts_per_prompt: 16
effective_maxrl_order: 15
all_failure_behavior: zero_group_gradient
maxrl_denominator_epsilon: 0
```

No new MaxRL temperature, scale, or tuning hyperparameter is introduced.

## CPU acceptance tests before GPU

Tests are written first under TDD. At minimum they must demonstrate:

1. `K=0`: all advantages exactly zero.
2. `K=G`: all advantages exactly zero.
3. `G=16, K=1`: success advantage `15`, failures `-1`.
4. `G=16, K=4`: successes `3`, failures `-1`.
5. every active mixed group has zero sum of advantages up to numerical precision.
6. invalid non-binary / non-finite reward inputs fail closed.
7. grouping/shape mismatches fail closed.
8. GRPO mode retains the base trainer's original advantage tensor unchanged.
9. MaxRL mode replaces the tensor before the ledger records it.
10. manifest/provenance records practical order `15` and the exact estimator semantics.

No GPU run is authorized until the focused RED->GREEN cycle and the full CPU suite pass freshly.

## GPU gates after CPU verification

After code acceptance:

1. 20-step engineering pilot from an untouched copy of the exact corrected `pi0`.
2. Verify ledger reconstruction, `K -> advantage` identities, finite values, and expected qualitative low-`p0` signal reweighting.
3. 150-step shakedown if the 20-step pilot is structurally clean.
4. Only then launch the matched canonical seed42 MaxRL trajectory from untouched `pi0`.

Pilot and shakedown outputs remain scientifically disposable and never become part of the canonical lineage.

## Interpretation gate

The previously frozen H1 remains the first scientific gate: practical MaxRL-15 must materially change **realized** signal allocation in the expected lower-`p0` direction relative to canonical GRPO before downstream behavioral differences are interpreted as an objective-level test of transfer.

If H1 fails, inspect estimator/implementation/normalization first. Do not rescue the run by redefining the behavioral target post hoc.
