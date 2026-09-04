# Practical MaxRL-15 — real 2×A40 20-step GPU pilot PASS

Date: 2026-09-03

## Status

The disposable real 2×A40 practical MaxRL-15 engineering pilot has passed the
pre-frozen structural acceptance gate.

This is an engineering/runtime result only. It is not a scientific MaxRL
outcome and must not be used to support or reject H1/H2/H3.

Current execution state:

```text
CPU/TDD implementation gate: PASS
2×A40 NCCL host preflight:   PASS
20-step real GPU pilot:      PASS
150-step MaxRL shakedown:    NEXT
canonical MaxRL seed42:      NOT YET RUN
MaxRL scientific outcome:    NONE
```

The accepted code commit is:

```text
116741fa59f470e8d8410d22bddcf1897d3b40a9
fix: support TRL deque advantage logs in MaxRL
```

## Host qualification

The replacement RunPod host passed the repository's real two-rank NCCL smoke on
the default transport path.

Observed preflight facts:

```text
world_size = 2
GPU 0 = NVIDIA A40
GPU 1 = NVIDIA A40
torch = 2.13.0+cu130
torch CUDA = 13.0
NCCL = 2.29.7
rank-local inputs: 1 and 2
all-reduce result on both ranks: 3
status = PASS
```

No `NCCL_P2P_DISABLE=1` or comparable transport workaround was used.

## Pilot attempt 1 — integration failure before optimizer step 1

The first real MaxRL pilot attempt reached real generation/scoring but failed
before the first optimizer step.

Failure:

```text
RuntimeError:
Practical MaxRL expects TRL _logs['advantages'] to be a list
```

Inspection of the installed TRL 1.12 runtime showed:

```text
"advantages": deque(maxlen=args.generation_batch_size)
```

and later:

```text
self._logs["advantages"].extend(all_process_advantages.tolist())
```

The earlier CPU test double had modeled this diagnostic buffer as a Python
`list`. Therefore the failure was a real integration-contract mismatch in
instrumentation alignment, not evidence of an error in the practical MaxRL
estimator, NCCL, rank slicing, or vLLM generation.

Attempt 1 scientific status:

```text
optimizer steps completed = 0
scientific outcome = NONE
```

## Regression fix

The production wrapper now supports both the historical/list test shape and the
real TRL `collections.deque` shape while preserving the invariant that the
latest global completion-table advantage batch matches the actual MaxRL
advantages used by the loss and signal ledger.

The regression test was changed to model the real TRL deque container.

The RunPod bootstrap unit tests were also isolated from host-level optional
environment variables:

```text
RLVR_EXPECT_COMMIT
RLVR_RUN_2XA40_PREFLIGHT
```

so a real pod template cannot accidentally alter fake-bootstrap unit-test
semantics.

## Accepted 20-step rerun

The disposable pilot was rerun from the untouched corrected canonical pi0 and
completed all 20 steps.

The frozen acceptance checker reported:

```json
{
  "aggregate_token_is_ess_fraction": 0.9979146076241244,
  "group_size": 16,
  "groups": 40,
  "max_advantage_error": 1.5894571969710114e-07,
  "rank_files": 2,
  "rows": 640,
  "status": "PASS",
  "steps": 20
}
```

Therefore the real stack satisfied the engineering invariants:

```text
2 rank files
20 generation_global_step values
640 rollout rows
40 prompt groups
16 rows per prompt group
practical MaxRL-15 advantage identities within tolerance
finite/non-degenerate token-level importance sampling
aggregate token IS ESS/N ≈ 0.997915
```

The pilot artifacts remain disposable and must not enter the canonical MaxRL
trajectory.

## Interpretation boundary

The 20-step pilot does not establish that:

- reward improves;
- realized signal allocation has already shifted toward lower p0;
- correctness movement changes;
- H2 or H3 is favored.

Those are outside the engineering gate.

The only supported conclusion is:

```text
The real 2×A40 TRL/vLLM path can execute the frozen practical MaxRL-15
advantage intervention with the expected group structure, ledger identities,
and numerically healthy token-level IS diagnostics.
```

## Next gate

Run a disposable 150-step MaxRL shakedown from a fresh untouched copy of the
same corrected canonical pi0.

The shakedown should retain all structural checks from the 20-step pilot and
then inspect whether the realized training-signal allocation begins to shift in
the intended lower-p0 direction before spending a full canonical trajectory.

Do not resume the 20-step pilot checkpoint and do not promote its artifacts into
the canonical lineage.

If the 150-step shakedown does not materially alter realized signal allocation
relative to canonical GRPO, stop mechanistic interpretation and inspect the
objective/outer-stack interaction before launching canonical MaxRL.
