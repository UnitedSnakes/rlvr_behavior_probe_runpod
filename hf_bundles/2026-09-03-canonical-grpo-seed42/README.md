# Canonical GRPO seed42 analysis bundle

Date: 2026-09-03

This directory is the lightweight packaging record for the canonical Qwen3-0.6B GRPO seed42 analysis after the exposed-vs-unexposed outcome deblind.

It does **not** duplicate the 24+ GB canonical checkpoint repository inside Git. The model/checkpoint lineage remains on Hugging Face; this bundle describes the lightweight analysis artifacts and provenance that should be uploaded alongside it or to a dedicated private Dataset repository.

## Canonical upstream artifacts

### Corrected pre-RL policy

```text
HF repo: HKReporter/rlvr-behavior-probe-pi0-corrected-canonical-2026-08-30
pi0_lineage_id: f89fc90226a67a6a3c7374f9c13abadfcecda88f397ab812fa4130f1f425605b
```

### Canonical GRPO trajectory

```text
HF repo: HKReporter/rlvr-behavior-probe-grpo-canonical-seed42-2026-09-02
training seed: 42
optimizer steps: 3736
G: 16
world size: 2
analysis implementation commit: 386f300e36562ad78063fcfd4b5ed4137325fd9d
```

The GRPO repo should remain the source of truth for large checkpoints, ledger shards, and snapshot model artifacts. This analysis bundle is for compact derived tables, provenance, and paper-facing diagnostics.

## Scientific result frozen with this bundle

Primary post-outcome result:

```text
no stable own-exposure advantage
```

On the fixed 256-question train panel, substantial correctness gains occur among questions that have not yet been directly sampled for training. At the frozen 25/45/65 cutoffs there are zero `own_exposure_candidate` cells among the 15 adjusted symmetric cutoff x p0-bin cells under the pre-outcome descriptive classification rule.

The complete interpretation and causal caveats are in:

```text
docs/superpowers/checkpoints/2026-09-03-exposure-split-postoutcome-and-paper-claim-freeze.md
```

Do not summarize this bundle as a randomized causal estimate of transfer.

## Lightweight files to include in the HF analysis bundle

The following local outputs are the minimum preferred upload set after regenerating/verifying them from the canonical data.

### Fixed-panel baseline and movement

```text
analyses/canonical_snapshot_crossfit/crossfit_trajectory.csv
analyses/canonical_snapshot_crossfit/per_question_crossfit.csv
```

### Canonical ledger signal allocation

```text
analyses/canonical_ledger_crossfit_signal/signal_trajectory_directional.csv
analyses/canonical_ledger_crossfit_signal/signal_trajectory_symmetric.csv
analyses/canonical_ledger_crossfit_signal/signal_movement_join.csv
analyses/canonical_ledger_crossfit_signal/ledger_integrity.json
analyses/canonical_ledger_crossfit_signal/cumulative_abs_advantage_per_panel_question.png
analyses/canonical_ledger_crossfit_signal/active_group_fraction.png
analyses/canonical_ledger_crossfit_signal/exploratory_dapo_is_abs_mass_per_panel_question.png
```

### Exposure timing / balance

```text
analyses/canonical_exposure_split_transfer/balance_question_rows.csv
```

If the exposure-split analysis directory contains additional raw/symmetric summary CSVs produced by the current script version, include them as well, but do not substitute them for the frozen covariate-adjusted outputs below.

### Frozen covariate-adjusted exposure analysis

```text
analyses/canonical_exposure_split_adjusted/adjustment_input_rows.csv
analyses/canonical_exposure_split_adjusted/adjusted_directional.csv
analyses/canonical_exposure_split_adjusted/adjusted_symmetric.csv
analyses/canonical_exposure_split_adjusted/adjusted_skipped_cells.csv
```

### Scientific provenance docs

Include verbatim copies of:

```text
docs/superpowers/specs/2026-09-01-signal-allocation-analysis-prereg.md
docs/superpowers/checkpoints/2026-09-02-postrun-preoutcome-analysis-addendum.md
docs/superpowers/checkpoints/2026-09-03-exposure-split-preoutcome-decision.md
docs/superpowers/checkpoints/2026-09-03-cutoff-balance-observed-and-adjustment.md
docs/superpowers/checkpoints/2026-09-03-exposure-split-postoutcome-and-paper-claim-freeze.md
docs/superpowers/specs/2026-09-03-maxrl-objective-intervention-amendment.md
```

The upload should preserve these filenames and dates. Do not rewrite the pre-outcome files during packaging.

## Regeneration commands

From the repository root with the project environment active:

```bash
python -m analyses.ledger_crossfit_signal_allocation
python -m analyses.exposure_split_adjusted
```

The exposure adjustment command must end with:

```text
CANONICAL COVARIATE-ADJUSTED EXPOSURE SPLIT: COMPLETE
```

The ledger command must end with:

```text
CANONICAL LEDGER CROSS-FIT SIGNAL ANALYSIS: PASS
```

These markers establish successful script completion only. They do not substitute for checking the input lineage or for scientific interpretation.

## Suggested remote layout

If a dedicated private Hugging Face Dataset repo is used, a clean layout is:

```text
canonical-seed42-analysis-2026-09-03/
  manifest.json
  provenance/
    2026-09-01-signal-allocation-analysis-prereg.md
    2026-09-02-postrun-preoutcome-analysis-addendum.md
    2026-09-03-exposure-split-preoutcome-decision.md
    2026-09-03-cutoff-balance-observed-and-adjustment.md
    2026-09-03-exposure-split-postoutcome-and-paper-claim-freeze.md
    2026-09-03-maxrl-objective-intervention-amendment.md
  snapshot_crossfit/
    crossfit_trajectory.csv
    per_question_crossfit.csv
  ledger_signal/
    ...
  exposure_split/
    balance_question_rows.csv
    adjustment_input_rows.csv
    adjusted_directional.csv
    adjusted_symmetric.csv
    adjusted_skipped_cells.csv
```

A dedicated repo name is not frozen here. Do not create a public repo merely for convenience; preserve the current privacy policy unless the user explicitly changes it.

## Upload boundary

The current ChatGPT environment has GitHub write access but no authenticated Hugging Face write connector/token. Therefore this repository record can prepare and verify the bundle specification, but it cannot truthfully claim that the private HF upload itself has occurred.

When uploading from the user's authenticated machine/RunPod, copy only regenerated files that exist and record SHA256 hashes for every uploaded lightweight artifact. The final HF-side `manifest.json` should match the Git version in this directory, with an additional hash table if desired.

## Verification before calling the HF bundle complete

On the authenticated machine:

1. regenerate the canonical analysis outputs from the exact canonical inputs;
2. verify the two completion markers above;
3. verify the canonical `pi0_lineage_id` and the canonical GRPO HF repo identity;
4. compute SHA256 for every lightweight file being uploaded;
5. upload to the intended private HF Dataset repo/path;
6. list/download the remote files and compare hashes;
7. only then call the HF copy complete.

The Git package being present is not evidence that the HF copy exists.
