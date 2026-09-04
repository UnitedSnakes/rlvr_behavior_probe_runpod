# Practical MaxRL-15 — 150-step shakedown structural PASS

Date: 2026-09-03

## Status

The disposable real 2xA40 150-step practical MaxRL-15 shakedown completed and
passed the structural/numerical verifier.

Observed checker output:

```json
{
  "status": "PASS",
  "steps": 150,
  "rows": 4800,
  "groups": 300,
  "rank_files": 2,
  "max_advantage_error": 1.5894571969710114e-07,
  "aggregate_token_is_ess_fraction": 0.997971078180227
}
```

This establishes:

```text
150 complete generation steps
4800 ledger rollout rows
300 G=16 prompt groups
2 rank files
practical MaxRL advantage identities within 1e-6
finite, non-degenerate token-level IS
aggregate token IS ESS/N ~= 0.997971
```

This checkpoint records structural acceptance only. At the time the
pre-H1-outcome analysis addendum was written, the p0-binned shakedown signal
allocation had not been inspected.

Current gate state:

```text
CPU/TDD implementation:          PASS
2xA40 default NCCL preflight:    PASS
20-step real GPU pilot:          PASS
150-step structural shakedown:   PASS
matched first-150 H1 analysis:   NEXT
canonical MaxRL seed42:          NOT YET AUTHORIZED
MaxRL behavioral outcome:        NONE
```

The shakedown remains disposable. Do not resume it into the canonical MaxRL
trajectory.
