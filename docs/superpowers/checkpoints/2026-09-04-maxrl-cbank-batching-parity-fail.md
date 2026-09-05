# Canonical MaxRL C-bank batching parity failure

Date: 2026-09-04

## Status

A proposed evaluation-runtime optimization batched multiple independently
seeded questions into one vLLM `LLM.generate` call. Before using that
implementation for the canonical MaxRL fixed-panel result, it was compared
against the original sequential evaluator on the same MaxRL `pi_005` policy
and the same C-bank question seeds.

Observed exact sample-path parity check:

```text
old sequential questions = 72
new batched questions     = 128
common questions          = 72
rollouts checked          = 1152
token-path mismatches     = 1147
exact matches             = 5
```

Decision:

```text
batched evaluator for primary GRPO-vs-MaxRL comparison: REJECTED
canonical GRPO evaluator: sequential
canonical MaxRL evaluator: sequential
```

## Interpretation

This mismatch does not by itself show that batching changes the marginal
sampling distribution or introduces bias. Long stochastic autoregressive
trajectories can diverge after batch-shape-dependent numerical differences even
when explicit request seeds are preserved.

It does show that the same nominal C-bank seed does not preserve the same
realized sample path across evaluator batch shapes. Mixing sequential GRPO
evaluation with batched MaxRL evaluation would therefore break the intended
common-random-number pairing of the objective comparison.

The batched implementation also showed no compelling wall-clock benefit on
this workload: generation remained decode-bound with the A40 near full
utilization.

## Canonical decision

The canonical MaxRL fixed-panel evaluation was restarted from empty canonical
output directories using the sequential evaluator implementation from:

```text
1c26b1f0f3c5f6ea1187fd00318587388a891272
```

Its generation structure matches the canonical GRPO evaluator:

```text
for each question:
    one LLM.generate call
    n = 16
    one question-specific C-bank seed
```

The partial sequential and batched `pi_005` outputs are parity-diagnostic
artifacts only. They do not enter H2/H3 analysis.

The earlier batching runtime amendment is retained as historical provenance of
the proposed optimization; this checkpoint records its rejection for the
canonical paired result.
