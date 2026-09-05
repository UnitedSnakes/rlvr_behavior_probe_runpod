# Canonical practical MaxRL-15 seed42 backup bundle

Date: 2026-09-05

This bundle is the off-pod backup record for the completed canonical practical
MaxRL-15 seed42 intervention and its fixed-panel H2/H3 analysis.

The scientific post-outcome checkpoint is:

```text
docs/superpowers/checkpoints/2026-09-05-maxrl-h2-h3-postoutcome-gate.md
```

The frozen result is:

```text
H1 mechanism gate: SUPPORTED
H2 primary behavioral prediction: SUPPORTED
H3 alternative as the primary explanation: NOT SUPPORTED
H4 diagnostic stop: NOT ACTIVE
```

## Large-artifact boundary

Git remains the source of truth for code, frozen scientific documents, and
lightweight derived tables/figures.

Private Hugging Face storage is the off-pod source of truth for the large raw
artifacts:

```text
model/checkpoint repo:
HKReporter/rlvr-behavior-probe-maxrl-canonical-seed42-2026-09-05

analysis/raw-evaluation dataset repo:
HKReporter/rlvr-behavior-probe-maxrl-analysis-seed42-2026-09-05
```

Both repos must remain private unless the user explicitly changes that policy.

## Required backup set

Model/checkpoint repo:

```text
controlled_run_outputs/maxrl_canonical_seed42/
```

This includes the canonical manifests, snapshot schedule, 20 policy snapshots,
and the two full signal-ledger rank shards.

Analysis/raw-evaluation repo:

```text
controlled_run_outputs/maxrl_snapshot_eval_train256_k16_cbank/
controlled_run_outputs/maxrl_snapshot_crossfit/
controlled_run_outputs/maxrl_ledger_crossfit_signal/
controlled_run_outputs/maxrl_grpo_objective_comparison/
```

Also preserve, when present:

```text
controlled_run_outputs/_slow_evaluator_partial_for_parity_check/
controlled_run_outputs/_batched_parity_pi005/
controlled_run_outputs/_parity_fail_batched_pi005/
controlled_run_outputs/*maxrl*.log
```

The partial evaluator outputs are diagnostic provenance only and must never be
mixed into the canonical H2/H3 analysis.

## Provenance constants

```text
training execution commit:
981475795538eee391c7e86aa022ee609b539770

sequential snapshot evaluator implementation:
1c26b1f0f3c5f6ea1187fd00318587388a891272

pi0 lineage:
f89fc90226a67a6a3c7374f9c13abadfcecda88f397ab812fa4130f1f425605b
```

Later analysis/packaging commits do not replace the training execution commit.

## Packaging command

From the authenticated RunPod/repository root:

```bash
python -m controlled_run.backup_maxrl_seed42 \
  --prepare-git-results
```

This computes SHA256 for the full backup set and copies the lightweight derived
outputs into tracked `analyses/canonical_maxrl_*` directories. It does not
upload unless `--upload` is supplied.

After inspecting the plan and ensuring Hugging Face authentication is active:

```bash
python -m controlled_run.backup_maxrl_seed42 \
  --prepare-git-results \
  --upload
```

A successful remote backup must end with:

```text
MAXRL REMOTE BACKUP: VERIFIED
```

and write:

```text
hf_bundles/2026-09-05-canonical-maxrl-seed42/backup_manifest.json
hf_bundles/2026-09-05-canonical-maxrl-seed42/upload_record.json
```

The local manifest records SHA256 and byte size for every mapped artifact.
Remote verification checks that every expected path exists, checks remote sizes
where metadata is available, and checks Hugging Face LFS SHA256 where exposed
by the API.

## Lightweight Git result set

After regenerating the MaxRL ledger plots with `--objective-label MaxRL`, copy
the derived outputs into Git through the backup utility. The intended tracked
directories are:

```text
analyses/canonical_maxrl_snapshot_crossfit/
analyses/canonical_maxrl_ledger_crossfit_signal/
analyses/canonical_maxrl_grpo_objective_comparison/
```

These are derived and reproducible. The raw C-bank JSONL, signal ledger, and
policy snapshots remain outside Git.
