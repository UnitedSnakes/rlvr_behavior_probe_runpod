# Controlled Qwen3 — 150-step disposable GRPO shakedown

## Status

Completed 2026-09-02. **PASS.**

This run used the untouched canonical `pi_0` and the fully frozen post-amendment GRPO recipe, including `top_p=1.0`, canonical terminated-and-correct reward, `mask_truncated_completions=false`, and token-level vLLM importance correction with `token_truncate`.

This shakedown is disposable engineering/scientific validation data and must not be mixed with canonical trajectory analysis.

## Frozen acceptance criteria

The truncation-policy amendment preregistered four trend-based requirements:

```text
completions/clipped_ratio   must fall materially
completions/mean_length     must fall
reward mean                 must rise
frac_reward_zero_std        must stay low
```

## Structure

```text
rollout ledger rows: 4800
steps:               150 (0..149)
ranks:               2
groups:              300
```

Each 30-step window contains 60 G=16 prompt groups / 960 rollouts.

## 30-step windows

```text
steps      clipped      mean length    reward       zero-std groups
0-29        53.54%         1589.9       33.65%         10.00%
30-59       56.35%         1618.2       32.08%         18.33%
60-89       50.10%         1531.7       39.69%         20.00%
90-119      45.94%         1487.8       44.27%          5.00%
120-149     46.46%         1481.0       41.77%          8.33%
```

Early-to-late comparison:

```text
clipped ratio:  53.54% -> 46.46%   delta -7.08 percentage points
mean length:    1589.9 -> 1481.0   delta -108.9 tokens
reward mean:    33.65% -> 41.77%   delta +8.12 percentage points
zero-std:       10.00% ->  8.33%   delta -1.67 percentage points
```

Descriptive per-step OLS slopes over the full 150 steps:

```text
clipped ratio:  -0.083109 per 100 steps
mean length:    -115.248 tokens per 100 steps
reward mean:    +0.093482 per 100 steps
zero-std:       -0.052180 per 100 steps
```

The first two windows are noisy and briefly move in the wrong direction, but the later sustained movement is aligned across the three primary quantities. The final 60 steps remain well below the initial clipped ratio and mean length and well above the initial reward rate.

## Zero-std interpretation

The mid-run zero-std windows reach 18.33% and 20.00%, but this is not a persistent collapse. Across all 300 groups the zero-std count is approximately 37/300 = 12.33%, and the final two windows fall to 5.00% and 8.33%.

This is compatible with the finite-G dead-group rate suggested by the canonical K=32 train baseline: empirical K16 live rates were 86.72% and 85.55%, corresponding to descriptive dead rates of roughly 13.3% and 14.5%. Therefore the shakedown does not show a new zero-variance failure mode.

## Importance-sampling sanity

```text
active tokens:                7,400,207
actual token ESS/N:           0.9979601
sequence upper masks:         0
token-at-upper-cap fraction:  0.0
```

The token-level vLLM correction remains non-degenerate throughout training and no sequence-level masking reappears.

## Gate decision

All four frozen shakedown requirements are satisfied:

1. completion clipping falls materially;
2. mean completion length falls;
3. canonical reward rises;
4. zero-std groups remain at a finite-G baseline scale and do not trend upward or collapse the run.

**150-step shakedown gate: PASS.**

The pre-canonical scientific blockers recorded in the truncation-policy, top-p/IS, token-diagnostic, and K=32 rebaseline checkpoints are now released.

Canonical GRPO may now be launched, but it must start from the untouched canonical `pi_0`, not from this shakedown or any prior pilot checkpoint. The shakedown output remains disposable and must not be used as canonical scientific trajectory data.
