# RLVR Behavioral Probe

This project asks one question: **when RLVR makes a model better, how local is that learning?** Does a problem mainly improve after that same problem has been used for RL training, or can training on other problems make it better first?

The current controlled runs use Qwen3-0.6B on GSM8K. `p0` is the pre-RL success rate of a problem. GRPO and MaxRL start from the same SFT model and use the same outer training setup.

## Main result: improvement before a problem's own RL exposure

I tracked a fixed panel of 256 GSM8K training problems through one epoch of GRPO. Because each problem is used for training once, I know exactly when it first contributes its own training group.

For problems with low but nonzero `p0`, correctness already improves substantially **before** that happens:

| training point | already exposed | not yet exposed |
|---:|---:|---:|
| 25% | +6.25 pp | +10.02 pp |
| 45% | +7.42 pp | +12.80 pp |
| 65% | +10.90 pp | +12.80 pp |

A September 6 question-level resampling check keeps the pre-exposure gain positive at all three checkpoints. The exposed-vs-unexposed difference itself is not stable enough to claim that unseen problems improve more, or that direct exposure has zero effect.

The narrower conclusion is the important one: **a problem can improve substantially before it has contributed any training group of its own.**

## GRPO vs. MaxRL: moving the signal does not move correctness in lockstep

I then ran a matched MaxRL comparison. MaxRL changes where the realized RL signal is concentrated: by the end of training, relative to GRPO, much more cumulative absolute advantage falls on low-`p0` problems and much less on high-`p0` problems.

| `p0` bin | MaxRL / GRPO advantage mass |
|---:|---:|
| 0 | 2.205x |
| (0, .25] | 1.720x |
| (.25, .5] | 0.988x |
| (.5, .75] | 0.592x |
| (.75, 1) | 0.463x |

But correctness does not show a matching persistent shift toward those low-`p0` bins. At the endpoint, MaxRL minus GRPO correctness change is:

| `p0` bin | difference in correctness change |
|---:|---:|
| 0 | -4.71 pp |
| (0, .25] | -0.44 pp |
| (.25, .5] | -1.91 pp |
| (.5, .75] | -0.62 pp |
| (.75, 1) | -0.16 pp |

Across the full trajectory, the advantage-mass shift is stable; the binwise correctness differences are not. A centered follow-up analysis gives a more mixed picture, so I do **not** claim that behavior never reallocates. The result is simply that **where the scalar RL signal lands and where correctness improves are not tightly coupled at the problem level in these runs.**

## Scoring check

One possible artifact was that the model's stopping/formatting behavior changes during training and the GSM8K scorer can fall back to the last number in a response.

On September 8 I rescored the frozen outputs with that fallback disabled. The main result remains:

- whole-panel GRPO correctness change: **+7.29 pp → +8.20 pp** under the stricter scorer;
- pre-exposure gains in the low-nonzero-`p0` bin: **+10.57, +13.53, +14.14 pp** at 25%, 45%, and 65%.

So that particular extraction artifact does not explain the pre-exposure improvement.

## What I do not know yet

The experiments argue against a simple story where improvement is mostly tied to a problem's own RL update. They do **not** tell me what transfers across problems.

The next question is: **what is the unit of transfer?** Shared reasoning patterns, internal representations, stopping behavior, answer format, or some mixture of these could all matter. I am especially interested in whether we can predict which problems benefit from training signal generated elsewhere.

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

The current GRPO/MaxRL comparison is one matched training seed, and exposure order is not randomized. I treat these as diagnostics of the simple problem-local story, not as a randomized causal estimate.

For the full internal experiment history, implementation notes, and agent handoff material, see the `research-workbench` branch.
