# Canonical practical MaxRL-15 — structural integrity PASS

Date: 2026-09-04

## Status

The full canonical seed42 practical MaxRL-15 training trajectory completed and
passed the post-training structural/numerical acceptance gate.

This checkpoint is written before inspecting the canonical MaxRL fixed-panel
K=16 C-bank behavioral outcomes.

Decision:

```text
canonical MaxRL training structural integrity: PASS
fixed-panel behavioral deblind: NOT YET PERFORMED
H2/H3 outcome: NOT YET OBSERVED
```

## Training provenance

```text
training execution commit:
981475795538eee391c7e86aa022ee609b539770

objective:
practical MaxRL-15

mode:
canonical

scientific_use:
true

pi0 lineage:
f89fc90226a67a6a3c7374f9c13abadfcecda88f397ab812fa4130f1f425605b
```

The later acceptance/evaluation support code is not the training execution
commit and must not replace this provenance.

## Structural acceptance result

Observed checker output:

```json
{
  "aggregate_token_is_ess_fraction": 0.9980541312524671,
  "execution_commit": "981475795538eee391c7e86aa022ee609b539770",
  "group_size": 16,
  "groups": 7472,
  "max_advantage_error": 1.5894571969710114e-07,
  "mode": "canonical",
  "nonfinite_numeric_fields": 0,
  "rank_files": 2,
  "rows": 119552,
  "scientific_use": true,
  "snapshot_final_step": 3736,
  "snapshots": 20,
  "status": "PASS",
  "steps": 3736
}
```

Thus the complete trajectory contains:

```text
3736 generation steps, indices 0..3735
119552 rollout ledger rows
7472 G=16 prompt groups
2 rank ledger files
20 frozen policy snapshots
final snapshot after optimizer step 3736
max practical-MaxRL advantage identity error ~= 1.59e-7
aggregate actual token-IS ESS/N ~= 0.998054
non-finite numeric ledger fields = 0
```

## Interpretation boundary

This establishes canonical trajectory integrity only.

It does not establish:

- final fixed-panel DeltaC;
- whether MaxRL changes behavioral allocation relative to GRPO;
- H2 or H3;
- whether final reward movement is driven by correctness versus termination.

Do not substitute training reward logs for the fixed K=16 C-bank panel.

## Next gate

Use the pre-outcome protocol frozen in:

```text
docs/superpowers/specs/2026-09-04-maxrl-canonical-fixed-panel-preoutcome-addendum.md
```

Evaluate all 20 MaxRL policy snapshots on the same train256 K=16 C-bank used for
canonical GRPO, then run the same K=32 A/B cross-fit movement analysis and
ledger-signal join.

Only after the complete evaluation/analysis pipeline passes should the
GRPO-versus-MaxRL H2/H3 comparison be deblinded.
