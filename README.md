# RLVR Behavioral Probe

Controlled experiments on how reinforcement-learning post-training reallocates training signal and changes model behavior.

The project began with a simple question: if a behavior has pre-RL success probability `p0`, does RL mainly amplify already reachable successes or expand what the model can reach? The current controlled result makes the question more specific:

> **Does behavioral improvement occur on the same questions that directly generate the training signal?**

For the canonical Qwen3-0.6B GRPO run, the answer is not stably yes.

## Current result — 2026-09-03

Canonical seed42 GRPO was trained from an exactly frozen corrected pre-RL policy. A fixed 256-question GSM8K-train panel was evaluated throughout training. At three analysis cutoffs, each panel question was classified by whether it had already been sampled for training.

The split definition, 25/45/65 cutoffs, interpretation scale, measured-covariate balance audit, and covariate-adjusted OLS estimator were fixed before reading the exposed-vs-unexposed outcomes.

Primary result:

```text
no stable own-exposure advantage
```

For low-`p0` questions `(0,.25]`, adjusted symmetric correctness movement was:

| training cutoff | already exposed | not yet exposed | unexposed - exposed |
|---:|---:|---:|---:|
| 25% | +6.25 pp | +10.02 pp | +3.77 pp |
| 45% | +7.42 pp | +12.80 pp | +5.38 pp |
| 65% | +10.90 pp | +12.80 pp | +1.90 pp |

Across all 15 adjusted symmetric cutoff × `p0`-bin cells, the pre-frozen descriptive labels are:

```text
transfer_compatible:      8
unexposed_higher:         3
mixed_or_uncertain:       2
not_classifiable:         2
own_exposure_candidate:   0
```

This is evidence against a **strong prompt-local account** in which questions should systematically improve more after they themselves generate direct training signal. It is not a randomized causal estimate: exposure order can still contain unmeasured structure, and shared-parameter interference is inherent. The safe interpretation is that the run shows substantial behavioral change before own exposure and no stable own-exposure advantage, consistent with substantial cross-question transfer.

Full provenance and causal caveats:

- `docs/superpowers/checkpoints/2026-09-03-exposure-split-postoutcome-and-paper-claim-freeze.md`

## Why this changes the project

The current scientific chain is:

```text
objective
  -> realized training-signal allocation
  -> parameter update
  -> shared-parameter transfer / interference
  -> behavioral-change allocation
```

So these are different objects:

```text
where training signal is allocated
!=
where behavioral improvement appears
```

That distinction is now the main motivation for the planned GRPO-versus-MaxRL objective intervention. The next question is not merely whether MaxRL changes nominal difficulty weighting, but whether a verified change in **realized signal allocation** actually changes the **allocation of correctness improvement**.

The scientific hypothesis hierarchy is frozen in:

- `docs/superpowers/specs/2026-09-03-maxrl-objective-intervention-amendment.md`

The implementation semantics are frozen separately in:

- `docs/superpowers/specs/2026-09-03-maxrl-practical-estimator-implementation-amendment.md`

For the frozen `G=16` comparison, the paper's dropped-baseline practical estimator corresponds to **order `T=15` (practical MaxRL-15)**. This changes only the group advantage estimator; the matched canonical DAPO/token-IS outer stack remains fixed. No MaxRL GPU outcome exists yet.

## Canonical lineage

### Corrected pre-RL policy

```text
HF repo:
HKReporter/rlvr-behavior-probe-pi0-corrected-canonical-2026-08-30

pi0_lineage_id:
f89fc90226a67a6a3c7374f9c13abadfcecda88f397ab812fa4130f1f425605b
```

### Canonical GRPO seed42

```text
HF repo:
HKReporter/rlvr-behavior-probe-grpo-canonical-seed42-2026-09-02

analysis implementation commit:
386f300e36562ad78063fcfd4b5ed4137325fd9d
```

Canonical training geometry:

```text
model: Qwen3-0.6B
mode: canonical
scientific_use: true
world_size: 2
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
global optimizer batch: 32
generation batch: 32
G / num_generations: 16
unique prompts per generation: 2
TRL steps_per_generation: 4
optimizer steps: 3736
prompt groups: 7472
ledger rows: 119552
```

The canonical ledger has two rank files, steps `0..3735`, 32 rollout rows per generation step, and exactly 7472 `G=16` prompt groups.

Structural-integrity record:

- `docs/superpowers/checkpoints/2026-09-02-grpo-canonical-integrity.md`

## Fixed-panel measurement

The train allocation panel is GSM8K train `[:256]`.

Baseline probability is estimated with a K=32 pre-RL bank split into independent A/B halves. Primary movement analyses cross-fit the baseline:

```text
A half defines the p0 bin -> B half supplies the baseline outcome
B half defines the p0 bin -> A half supplies the baseline outcome
```

Snapshot outcomes use a separate K=16 C-bank. The frozen bins are:

```text
0
(0,.25]
(.25,.5]
(.5,.75]
(.75,1)
1
```

The canonical reward is separated into:

```text
R = terminated and correct
T = termination
C = correctness independent of termination
```

From `pi0` to the final snapshot:

```text
DeltaR = +18.12 pp
DeltaT = +31.63 pp
DeltaC =  +7.29 pp
```

Termination acquisition dominates the global reward movement, while correctness still improves nontrivially. This is why the own-exposure analysis is stated in terms of `DeltaC`, not reward alone.

## Realized signal allocation

The canonical signal ledger reconstructs each `(generation_global_step, dataset_index)` prompt group and measures realized training-signal allocation across the frozen `p0` bins.

The current result is that the realized signal-allocation shape and the correctness-movement shape are not interchangeable. Prompt-local movement reflects parameter updates produced by many other training questions as well as any own exposure.

Canonical token-level importance sampling is well behaved (`ESS/N` approximately 0.998 across bins). Earlier sequence-level-IS collapse and truncation-mask distortions are retained as implementation/instrumentation history; they must not be presented as the canonical token-level phenomenon.

## Reproduce the current analyses

Use Python module invocation from the repository root:

```bash
python -m analyses.ledger_crossfit_signal_allocation
python -m analyses.exposure_split_adjusted
```

Expected completion markers:

```text
CANONICAL LEDGER CROSS-FIT SIGNAL ANALYSIS: PASS
CANONICAL COVARIATE-ADJUSTED EXPOSURE SPLIT: COMPLETE
```

Key derived outputs:

```text
analyses/canonical_ledger_crossfit_signal/
analyses/canonical_snapshot_crossfit/
analyses/canonical_exposure_split_transfer/
analyses/canonical_exposure_split_adjusted/
```

The adjusted exposure analysis writes:

```text
adjustment_input_rows.csv
adjusted_directional.csv
adjusted_symmetric.csv
adjusted_skipped_cells.csv
```

A successful script marker is a code/data-pipeline check, not a scientific conclusion by itself.

## Local development

Create the platform-neutral development environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r controlled_run/requirements-dev.txt
python -m pytest -q
```

Canonical SFT data are selected with the frozen formatted-token cutoff of 16,384. Verify the materialized bundle with:

```bash
python -m controlled_run.data_bundle
```

Canonical A40 runtime acceptance remains fail-closed:

```bash
python -m controlled_run.runtime_acceptance \
  --attention-backend flash_attention_2
```

Do not silently fall back from the canonical attention/runtime path.

## Historical Apple-Silicon development runtime

This path is retained only for reproducibility of the earlier local-development workflow. It is not the canonical measurement backend.

The vLLM-Metal development environment requires **macOS 15+**, native **arm64**, and **Python 3.12**. The official installer creates `~/.venv-vllm-metal`:

```bash
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
uv pip install --python ~/.venv-vllm-metal/bin/python -r requirements-macos-vllm.txt
```

The historical compatibility smoke used the exact public SFT revision `checkpoint-8-of-10`; it must not silently substitute an MLX-community conversion or another checkpoint.

**Metal results are for development, smoke tests, and small exploratory runs. CUDA vLLM remains the canonical measurement backend.**

## Historical RunPod image/bootstrap workflow

This block preserves the earlier image/bootstrap contract and is not the active branch selector for the present `codex/signal-ledger` work. The historical template used:

```text
HF_TOKEN={{ RUNPOD_SECRET_huggingface_token }}
GITHUB_DEPLOY_KEY_B64={{ RUNPOD_SECRET_github_rlvr_deploy_key_b64 }}
RLVR_REPO=git@github.com:UnitedSnakes/rlvr_behavior_probe_runpod.git
RLVR_BRANCH=difficulty-bin-analysis
RLVR_REPO_DIR=/workspace/rlvr_behavior_probe_runpod
```

The bootstrap command is `rlvr-bootstrap`; failures were logged to `/workspace/rlvr-bootstrap.log` without killing the pod. Historical result backups targeted the pre-existing private Dataset repo:

```text
HKReporter/rlvr-behavior-probe-results
```

The image acceptance path also retained a non-canonical top-k/JIT smoke test with:

```text
--top-k 20
--repetition-penalty 1.1
```

These values are historical infrastructure tests, not current canonical science settings.

## Authoritative research records

Start with the newest checkpoint rather than reading the whole history.

Current post-outcome handoff:

- `docs/superpowers/checkpoints/2026-09-03-exposure-split-postoutcome-and-paper-claim-freeze.md`

Important pre-outcome provenance:

- `docs/superpowers/specs/2026-09-01-signal-allocation-analysis-prereg.md`
- `docs/superpowers/checkpoints/2026-09-02-postrun-preoutcome-analysis-addendum.md`
- `docs/superpowers/checkpoints/2026-09-03-exposure-split-preoutcome-decision.md`
- `docs/superpowers/checkpoints/2026-09-03-cutoff-balance-observed-and-adjustment.md`

Next objective-intervention records:

- `docs/superpowers/specs/2026-09-03-maxrl-objective-intervention-amendment.md`
- `docs/superpowers/specs/2026-09-03-maxrl-practical-estimator-implementation-amendment.md`

Large-artifact / Hugging Face packaging map:

- `hf_bundles/2026-09-03-canonical-grpo-seed42/README.md`
- `hf_bundles/2026-09-03-canonical-grpo-seed42/manifest.json`

## Hugging Face artifact policy

Large checkpoints, raw rollout/ledger artifacts, and canonical model snapshots live in private Hugging Face repos under `HKReporter/`. Git stores code, configs, manifests, lightweight tables, and scientific provenance documents.

The 2026-09-03 HF bundle manifest records what lightweight analysis files should accompany the canonical seed42 run. The current ChatGPT session has GitHub write access but no authenticated Hugging Face write connector/token, so the manifest explicitly records that the HF-side analysis upload has **not** been performed from this environment.

Do not call an HF backup complete until the remote files have been listed/downloaded and their hashes checked against the local artifacts.

## Historical Qwen2.5 pilot

Before the controlled Qwen3 run, this project used public Qwen2.5 SFT/PPO checkpoints on a fixed 30-problem GSM8K subset. With 8 rollouts per problem and a 2048-token budget:

- sample accuracy moved from **55.4% SFT to 72.9% final RLVR**;
- the apparent RL advantage shrank from **+17.5 pp at pass@1** to **+3.3 pp at pass@8**;
- roughly **89–97% of positive gains** came from questions solved at least once by the shallow SFT sample;
- deeper SFT sampling removed the apparent "RL-only" successes in that small panel.

Those observations motivated the controlled lineage, larger pre-RL bank, cross-fitting, fixed-panel trajectory, and signal ledger. They are preliminary/historical evidence and are **not** the canonical training recipe or the current headline result.

## Project discipline

- Never rewrite a preregistration after seeing an outcome; add a dated checkpoint/amendment instead.
- Never infer truncation or special-token behavior from decoded text when token ids/finish reasons are available.
- Never treat green tests as scientific validation.
- Never merge pilot/shakedown outputs into the canonical lineage.
- Never describe the current exposure split as randomized.
- Prefer "no stable own-exposure advantage" / "evidence against a strong prompt-local account" over causal claims that the data do not support.
