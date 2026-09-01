# Controlled Qwen3 RLVR — Signal-Allocation Analysis Preregistration

Date: 2026-09-01. Written before the signal-ledger GPU pilot.

This note fixes the primary analysis definitions before observing the new
per-rollout importance-sampling / signal-ledger measurements. It does not amend
any canonical training hyperparameter.

## Panels

- **Train allocation panel:** the same GSM8K train `[:256]` natural-order panel
  used for the pre-RL signal-budget measurement. These prompts are eligible to
  receive direct GRPO rollout/gradient signal.
- **Held-out transfer panel:** the fixed 30-question GSM8K test subset used by
  the earlier deep evaluations. These prompts never enter GRPO training and are
  interpreted only as transfer/generalization measurements.

The two panels must not be conflated: `p0_test -> delta p_test` is not a direct
measurement of objective signal allocation.

## Primary training-side outcomes

For each generated training group / completion, record the quantities needed to
separate nominal objective weighting from realized pipeline weighting:

- dataset index and generation global step;
- raw correctness, termination, canonical terminated-and-correct reward;
- completion length;
- group successes `k`, group size `G`, TRL group reward standard deviation,
  and realized advantage;
- raw sequence `log rho` between training-model and vLLM sampling log-probs;
- post-`sequence_mask` importance-sampling ratio and its log when finite;
- whether the raw ratio exceeded the configured upper cap and was therefore
  zeroed by the mask rule.

The primary IS diagnostic is the distribution of raw `log rho` as a function of
completion length and realized group difficulty `k/G`, together with the
importance-weight effective sample size

`ESS = (sum rho)^2 / sum rho^2`.

No numerical pass/fail threshold for these quantities is declared here. The GPU
pilot is a diagnostic gate: a systematic difficulty- or length-dependent
reweighting that materially changes the intended signal allocation requires a
written design amendment before canonical GRPO.

## Temporal reward decomposition

Because the canonical reward is

`R = 1[terminated] * 1[correct]`,

checkpoint analysis will report both

- `P_t(T)`; and
- `P_t(C | T)`

in addition to `P_t(R=1) = P_t(T) P_t(C|T)`.

This tests whether early training is primarily a termination transition, a
correctness transition, or both.

## p0 estimation and regression-to-the-mean control

When `p0` is used both to stratify prompts and to define a later probability
change, the same Monte Carlo noise must not appear with opposite signs on the
x-axis and in `delta p`.

Primary analyses therefore use independent baseline samples via cross-fitting.
For a K=32 pre-RL bank, one 16-sample half defines the `p0` stratum while the
other 16-sample half supplies the baseline probability used in `delta p`; the
roles are then swapped and the two estimates are averaged.

Primary `p0` bins are fixed as:

- `0`;
- `(0, 1/4]`;
- `(1/4, 1/2]`;
- `(1/2, 3/4]`;
- `(3/4, 1)`;
- `1`.

## Confirmatory vs exploratory

Confirmatory analyses for the controlled run are:

1. train-side `p0` / realized group state versus realized signal exposure;
2. the raw-`log rho` length and `k/G` diagnostics plus ESS;
3. the `P(T)` and `P(C|T)` temporal decomposition;
4. separately reported train movement and held-out transfer curves under the
   fixed `p0` bins and cross-fitting rule.

Exploratory analyses include detailed trajectory topology, lag/lead relations,
frozen-pool GRPO-vs-MaxRL calibration, adaptive-sampling interventions, and any
representation-level explanation of residuals.
