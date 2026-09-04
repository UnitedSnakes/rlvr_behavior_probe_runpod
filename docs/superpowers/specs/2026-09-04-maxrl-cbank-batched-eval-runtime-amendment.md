# MaxRL C-bank batched-evaluation runtime amendment

Date: 2026-09-04

## Status

This runtime amendment is written after the first canonical MaxRL C-bank
evaluation attempt began, but before any fixed-panel behavioral outcome was
analyzed or used to judge H2/H3.

The initial evaluator submitted one question per `LLM.generate` call. That
preserved the intended sampling semantics but underutilized vLLM continuous
batching and projected several hours of avoidable evaluation time.

The partial outputs from that slow attempt are disposable and must be deleted
rather than mixed with the restarted evaluation.

## Change

Snapshot evaluation now submits multiple independently seeded questions in one
vLLM call:

```text
default request_batch_size = 32
```

Each question still receives its own `SamplingParams` object with the exact
same frozen values:

```text
n = 16
temperature = 0.8
top_p = 1.0
top_k = 0
repetition_penalty = 1.0
max_tokens = 2048
seed = 42*100000 + dataset_index + 75000
```

Only request scheduling changes.

vLLM 0.27.1 explicitly supports a sequence of prompts paired one-to-one with a
sequence of `SamplingParams` and automatically batches those requests subject
to memory constraints.

## Scientific boundary

This is an evaluation-runtime optimization, not a scientific estimator or
sampling-policy amendment.

The following remain unchanged:

```text
canonical MaxRL policy snapshots
train256 panel
K=16 C-bank
per-question C-bank seeds
generation distribution parameters
token-level termination semantics
R/T/C scoring
A/B cross-fit p0 bins
H2/H3 comparison rule
```

No partial slow-evaluation output may enter the canonical analysis. Restart all
20 MaxRL snapshots under one evaluator implementation after the code/test gate
passes.
