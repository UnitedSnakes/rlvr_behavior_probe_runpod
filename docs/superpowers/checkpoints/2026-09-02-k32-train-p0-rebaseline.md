# Controlled Qwen3 — Canonical train K=32 p0 rebaseline

## Status

Completed 2026-09-02. **PASS.**

This checkpoint replaces the old `top_p=0.95` train-side truncation/signal-budget numbers for all forward-looking controlled-run analyses. The older numbers remain historical motivation only.

Canonical GRPO is **not yet released**. The remaining pre-canonical scientific gate is the 150-step disposable shakedown required by `2026-08-30-grpo-truncation-policy-amendment.md`.

## Frozen collection semantics

Policy: untouched corrected canonical `pi_0`.

Dataset: GSM8K train `[:256]`, revision `740312add88f781978c0658806c59bc2815b9866`.

Sampling / reward:

```text
temperature: 0.8
top_p: 1.0
top_k: 0
repetition_penalty: 1.0
max_completion_length: 2048
reward: binary_terminated_final_answer_correctness
mask_truncated_completions: false
```

Probability-estimation bank:

```text
K = 32 per prompt
half A = independent vLLM call, n=16, seed = seed*100000 + dataset_index
half B = independent vLLM call, n=16, seed = seed*100000 + dataset_index + 50000
```

Canonical GRPO group size remains **G=16**. K=32 improves estimation of `p0`; it does not redefine any group-level GRPO quantity.

Primary model-implied live probability:

```text
P(live | p) = 1 - p^16 - (1-p)^16
```

## Structural verification

```text
prompts:               256
unique dataset indices:256
index range:           0..255
duplicates:            0
missing indices:       0
rollouts:              8192
```

Both GPU shards exited 0. Each prompt contains exactly two G=16 halves, and `p0 = (p0_A + p0_B)/2` exactly.

## Exact aggregate

```text
canonical terminated-and-correct success: 34.9121%
correctness-only:                         50.4517%
termination:                              45.6177%
cap / nontermination:                     54.3823%

P(correct | terminated):                  76.5320%
P(terminated | correct):                  69.1991%
correct but nonterminated, all rollouts:  15.5396%
fraction of correct rollouts lost to
  nontermination:                         30.8009%

mean train p0:                             0.349121
median train p0:                           0.3125
p0 = 0 prompts:                           20 / 256
p0 = 1 prompts:                            0 / 256
interior K32 p0 prompts:                 236 / 256 = 92.19%
```

The 2048-cap / termination issue therefore remains a first-order part of the canonical objective after switching to `top_p=1.0`: almost one third of correctness-only successes do not terminate within the budget and receive canonical reward 0.

This is direct baseline evidence for a possible early termination-acquisition phase. It does not by itself establish temporal ordering; that must be tested in the 150-step shakedown and later checkpoint trajectories.

## Exact frozen-bin table

Bins use the preregistered K=32 terminated-and-correct `p0` estimate.

```text
bin          n    mean p0    cap%   term%  correct%  reward%  model-live%  liveA%  liveB%
0           20    0.0000    82.34   17.66     4.69     0.00       0.00       0.00    0.00
(0,.25]     97    0.1205    72.42   27.58    25.23    12.05      77.42      86.60   82.47
(.25,.5]    51    0.3719    52.88   47.12    57.78    37.19      99.85     100.00  100.00
(.5,.75]    65    0.6101    34.09   65.91    81.39    61.01      99.86     100.00  100.00
(.75,1)     23    0.8288    14.67   85.33    92.93    82.88      93.44      95.65  100.00
```

The completion-cap gradient remains extremely strong under the final sampling policy: cap rate falls from 82.34% in the observed `p0=0` bin to 14.67% in `(0.75,1)`.

Do not directly interpret the numerical difference from the historical `top_p=0.95` table as an effect of changing top-p: both the operational `p0` distribution and other bank semantics changed. The valid conclusion is that the **difficulty-correlated nontermination mechanism robustly persists** under the final canonical sampling/reward semantics.

## Cross-fit A/B agreement

```text
mean p0_A:                 0.354248
mean p0_B:                 0.343994
mean A-B:                  0.010254
MAE(A,B):                  0.103516
RMSE(A,B):                 0.138438
corr(A,B):                 0.874866
exact same K16 estimate:   23.05%
|A-B| <= 1/16:             53.91%
|A-B| <= 2/16:             77.73%

empirical live A:          86.72%
empirical live B:          85.55%
live A/B agreement:        87.89%
```

The A/B mean difference is about 1.0 percentage point and the two independent K16 estimates correlate 0.875. Their pointwise disagreement is large enough to justify the preregistered cross-fit design rather than treating K16 `p0` as noise-free.

## Cross-fit cap x difficulty

The decisive robustness check uses one independent K16 half to define difficulty and the other half to measure cap/reward outcomes, then swaps them.

```text
A defines bin; B measures outcome
bin          n    cap%   reward%
0           33    80.11    3.41
(0,.25]     89    69.17   14.33
(.25,.5]    57    48.57   42.65
(.5,.75]    56    35.94   59.38
(.75,1)     20    19.38   79.06
1            1    18.75   81.25

B defines bin; A measures outcome
bin          n    cap%   reward%
0           37    76.86    5.07
(0,.25]     86    70.42   17.08
(.25,.5]    59    47.78   43.11
(.5,.75]    55    32.61   61.70
(.75,1)     19    18.09   77.63
```

The cap gradient is monotone in both cross-fit directions through the populated interior bins. Reward rises correspondingly. Therefore the observed difficulty-to-nontermination relation is not an artifact of using the same K32 sampling noise on both the x-axis and outcome.

The single `p0_A=1` prompt is retained as descriptive finite-K noise and must not be interpreted as evidence for a true support-boundary population.

## G=16 live exposure

```text
mean plug-in model P(live | p0_K32): 0.829780
empirical live A:                    0.867188
empirical live B:                    0.855469
observed K32 boundary p0=0 or 1:     0.078125
```

The empirical live rates being a few percentage points above the plug-in model mean is not treated as a contradiction. `f(p)=1-p^16-(1-p)^16` is concave; substituting a noisy finite-K estimate `p_hat` therefore creates downward plug-in bias by Jensen's inequality, especially near the observed `p_hat=0` boundary. Empirical A/B live states remain descriptive checks; the preregistered model-implied G=16 quantity is primary for difficulty-allocation analysis.

## Scientific conclusions released by this bank

1. The train-side baseline is now matched to final `top_p=1.0` sampling and canonical terminated-and-correct reward.
2. Difficulty-correlated nontermination is robust and survives independent cross-fitting.
3. The completion cap is not being used as a masking filter in canonical training; nontermination instead enters explicitly as reward 0.
4. Termination acquisition can be a large early learning component: 30.8% of correctness-only successes are currently lost to nontermination.
5. The usable train `p0` distribution is broad and mostly interior: 92.19% of prompts have K32 `p0` strictly between 0 and 1.
6. K=32 is an estimation bank only; all GRPO live-group theory remains G=16.

## Gate decision

**Train-side K=32 rebaseline: PASS.**

Do not launch canonical GRPO yet. Next run the required **150-step disposable shakedown from untouched `pi_0`** under the fully frozen recipe.

Trend-based acceptance remains exactly as frozen in the truncation amendment:

```text
completions/clipped_ratio   must fall materially
completions/mean_length     must fall
reward mean                 must rise
frac_reward_zero_std        must stay low
```

If clipped ratio does not fall materially, stop and revisit the SFT data recipe. Do not make another objective or cap change ad hoc.
