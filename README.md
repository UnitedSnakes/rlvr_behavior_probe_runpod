# RLVR Signal Allocation and Behavioral Change

RLVR produces training signal on sampled responses, but behavioral improvement does not have to stay on the same problems that produced that signal. This project asks **how tightly problem-level RL signal predicts where correctness actually improves**.

Cross-problem generalization itself is not surprising. The question here is more specific: if we track where the RL signal is generated — or deliberately move that signal around — does behavioral improvement move with it?

I look at this in two ways on Qwen3-0.6B + GSM8K:

1. **Exposure timing:** does a problem improve differently before vs. after it contributes its own RL training group?
2. **Objective intervention:** if MaxRL strongly changes which problems receive advantage mass, do correctness gains shift in the same way?

Both diagnostics point to the same mismatch: **where the scalar RL signal is concentrated and where behavior improves are not tightly coupled at the problem level in these runs.**

## 1. Own exposure does not mark where improvement begins

I tracked a fixed panel of 256 GSM8K training problems through one epoch of GRPO. Each problem is used for training once, so I know exactly when it contributes its own training group.

For problems with low but nonzero pre-RL success rate (`p0`), correctness is already substantially higher before that happens:

| training point | already exposed | not yet exposed |
|---:|---:|---:|
| 25% | +6.25 pp | +10.02 pp |
| 45% | +7.42 pp | +12.80 pp |
| 65% | +10.90 pp | +12.80 pp |

A September 6 question-level resampling check keeps the pre-exposure gain positive at all three checkpoints. The exposed-vs-unexposed difference itself crosses zero, so I do **not** claim that unseen problems improve more, or that direct exposure has no effect.

The useful result is narrower: **a substantial fraction of the behavioral improvement is already present before a problem contributes any RL training group of its own.** There is no stable own-exposure advantage in these checkpoints.

## 2. MaxRL moves the signal much more than it moves correctness

Exposure timing is observational, so I also changed the objective. GRPO and MaxRL start from the same SFT model and use the same outer training setup; MaxRL changes how sampled groups are weighted.

That intervention clearly changes where the realized signal goes. By the end of training, cumulative absolute advantage under MaxRL relative to GRPO is:

| `p0` bin | MaxRL / GRPO advantage mass |
|---:|---:|
| 0 | 2.205x |
| (0, .25] | 1.720x |
| (.25, .5] | 0.988x |
| (.5, .75] | 0.592x |
| (.75, 1) | 0.463x |

So MaxRL puts much more realized advantage on problems the starting model rarely solves, and much less on easier problems.

But correctness does not show a matching persistent reallocation. At the endpoint, MaxRL minus GRPO correctness change in the same bins is:

| `p0` bin | difference in correctness change |
|---:|---:|
| 0 | -4.71 pp |
| (0, .25] | -0.44 pp |
| (.25, .5] | -1.91 pp |
| (.5, .75] | -0.62 pp |
| (.75, 1) | -0.16 pp |

Across the full trajectory, the shift in advantage mass toward low-`p0` problems is persistent; the corresponding correctness differences are not. A centered follow-up analysis is more mixed, so I do **not** claim that the objective can never change the allocation of behavioral gains.

The narrower point is the one I care about: **a large change in where RLVR places scalar training signal does not produce a comparably clean change in where correctness improves.**

## Scoring robustness

One possible confound is answer extraction. The model changes how often and how cleanly it finishes answers during training, while the original GSM8K scorer can fall back to the last number in a response.

On September 8 I rescored the frozen outputs with that fallback disabled. The main pattern remains:

- whole-panel GRPO correctness change: **+7.29 pp → +8.20 pp** under the stricter scorer;
- pre-exposure gains in the low-nonzero-`p0` bin: **+10.57, +13.53, +14.14 pp** at 25%, 45%, and 65%.

So that particular extraction artifact does not explain the pre-exposure improvement.

## The open question

The experiments leave a gap between **where optimization pressure is applied** and **where behavior changes**. I do not yet know what mediates that gap.

Possible candidates include shared reasoning patterns, shared internal representations, interference between problems, stopping behavior, and answer-format changes. The next step is to ask whether we can predict which problems benefit from training signal generated elsewhere, and what distinguishes responses that improve from those that do not.

## Setup

- Qwen3-0.6B
- common SFT warm start on 10k OpenR1-Math examples
- GSM8K train `[:256]` fixed panel
- K=32 pre-RL bank for `p0`, split for cross-fitting
- separate K=16 snapshot evaluation bank
- matched GRPO / MaxRL, seed 42
- one training epoch, 3736 optimizer steps

Main analysis code and outputs:

- `analyses/exposure_split_adjusted.py`
- `analyses/strict_extractor_robustness.py`
- `analyses/maxrl_objective_comparison.py`
- `analyses/canonical_exposure_split_adjusted/`
- `analyses/canonical_maxrl_grpo_objective_comparison/`
- `analyses/strict_extractor_robustness/`

The current GRPO/MaxRL comparison is one matched training seed, and exposure order is not randomized. I treat these as diagnostics of the signal-to-behavior relationship, not as a randomized causal estimate.

For the full internal experiment history, implementation notes, and agent handoff material, see the `research-workbench` branch.
