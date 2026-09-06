# RLVR Behavioral Probe

Controlled experiments on how reinforcement-learning post-training reallocates training signal and changes model behavior.

The project began with a simple question: if a behavior has pre-RL success probability `p0`, does RL mainly amplify already reachable successes or expand what the model can reach? The current controlled result makes the question more specific:

> **Does behavioral improvement occur on the same questions that directly generate the training signal?**

For the canonical Qwen3-0.6B GRPO run, the answer is not stably yes.

## Current scientific result — 2026-09-05

Canonical seed42 Qwen3-0.6B GRPO and practical MaxRL-15 now provide two
complementary diagnostics of the same paper-level question:

> **Question-level behavioral improvement does not stably track local training
> signal allocation.**

### 1. GRPO own-exposure analysis

A fixed 256-question GSM8K-train panel was evaluated throughout canonical GRPO
training. At frozen 25/45/65 cutoffs, each panel question was classified by
whether its unique own training exposure had already occurred.

The split definition, cutoffs, interpretation scale, measured-covariate balance
audit, and covariate-adjusted estimator were fixed before reading the
exposed-vs-unexposed outcomes.

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

Across all 15 adjusted symmetric cutoff x `p0`-bin cells:

```text
transfer_compatible:      7
unexposed_higher:         3
mixed_or_uncertain:       2
not_classifiable:         3
own_exposure_candidate:   0
```

This is evidence against a **strong prompt-local account** in which questions
should systematically improve more after they themselves generate direct
training signal. It is not a randomized causal estimate. The safe
interpretation is substantial behavioral change before own exposure and no
stable own-exposure advantage, consistent with substantial cross-question
transfer.

### 2. MaxRL objective intervention

The follow-up intervention asked a stronger question: if the objective
materially changes **realized** signal allocation, does the allocation of
correctness improvement move with it?

Practical MaxRL-15 changes only the group advantage estimator while preserving
the matched canonical GRPO outer stack.

The canonical MaxRL seed42 run completed all 3736 optimizer steps, passed the
frozen structural gate, and was evaluated with the same sequential train-256
K=16 C-bank protocol as canonical GRPO.

Frozen result:

```text
H1 mechanism gate: SUPPORTED
H2 primary behavioral prediction: SUPPORTED
H3 alternative as the primary explanation: NOT SUPPORTED
H4 diagnostic stop: NOT ACTIVE
```

At 100%, MaxRL/GRPO cumulative `|A|` ratios across increasing frozen `p0`
bins are:

```text
0          2.205x
(0,.25]    1.720x
(.25,.5]   0.988x
(.5,.75]   0.592x
(.75,1)    0.463x
```

The corresponding MaxRL-minus-GRPO `DeltaC` contrasts are:

```text
0         -4.712 pp
(0,.25]   -0.437 pp
(.25,.5]  -1.908 pp
(.5,.75]  -0.620 pp
(.75,1)   -0.156 pp
```

The full 5%-through-100% trajectory shows a persistent left shift in realized
scalar advantage mass under MaxRL, while the same-bin absolute correctness
contrast fluctuates rather than showing a matching persistent advantage.

A post-outcome whole-panel-centered diagnostic further shows that evidence about
**relative** behavioral reallocation is mixed. The paper-safe conclusion is:

> Changing the objective materially reallocates realized scalar advantage mass,
> but a persistent matching absolute correctness advantage is not observed in
> the same bins; relative behavioral reallocation is less conclusive.

This does not prove that shared representations are the unique cause or that
objective allocation can never affect behavioral allocation.

The current paper-facing result hierarchy is therefore:

1. **main result:** no stable own-exposure advantage under canonical GRPO;
2. **complementary objective intervention:** MaxRL moves realized scalar
   advantage mass without a persistent matching same-bin absolute `DeltaC`
   advantage; relative behavioral reallocation is mixed;
3. implementation/pipeline diagnostics are supporting instrumentation evidence,
   not the headline.

Single-seed limitation:

> All behavioral and signal-allocation comparisons reported here are from a
> single matched training seed. We therefore do not estimate between-seed
> variability in either bin-level behavioral changes or objective-induced
> signal reallocation.

Current paper-facing correction/sensitivity handoff:

- `docs/superpowers/checkpoints/2026-09-06-attrib-deadline-postoutcome-sensitivity.md`

Earlier writing handoff:

- `docs/superpowers/checkpoints/2026-09-05-attrib-writing-handoff.md`

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

### Canonical MaxRL seed42

```text
training execution commit:
981475795538eee391c7e86aa022ee609b539770

sequential evaluator implementation:
1c26b1f0f3c5f6ea1187fd00318587388a891272

model/checkpoint HF repo:
HKReporter/rlvr-behavior-probe-maxrl-canonical-seed42-2026-09-05

analysis/raw-evaluation HF repo:
HKReporter/rlvr-behavior-probe-maxrl-analysis-seed42-2026-09-05
```

Structural acceptance:

```text
optimizer steps: 3736
prompt groups: 7472
ledger rows: 119552
rank files: 2
policy snapshots: 20
aggregate token IS ESS/N: 0.9980541312524671
status: PASS
```

The off-pod MaxRL backup is remotely verified: 313/313 expected files are
present and size-verified; 99 Hugging Face LFS objects were SHA256-verified.

Backup record:

- `hf_bundles/2026-09-05-canonical-maxrl-seed42/upload_record.json`

## Fixed-panel measurement

The train allocation panel is GSM8K train `[:256]`.

Baseline probability is estimated with a K=32 pre-RL bank split into
independent A/B halves. Primary movement analyses cross-fit the baseline:

```text
A half defines the p0 bin -> B half supplies the baseline outcome
B half defines the p0 bin -> A half supplies the baseline outcome
```

Snapshot outcomes use a separate K=16 C-bank under the sequential
one-question/request evaluator. Frozen bins are:

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

Shared pi0 aggregate:

```text
R = 0.349121
T = 0.456177
C = 0.504517
```

Canonical GRPO at 100%:

```text
R = 0.530273
T = 0.772461
C = 0.577393

DeltaR = +18.12 pp
DeltaT = +31.63 pp
DeltaC =  +7.29 pp
```

Canonical MaxRL at 100%:

```text
R = 0.515381
T = 0.747314
C = 0.563965

DeltaR = +16.63 pp
DeltaT = +29.11 pp
DeltaC =  +5.94 pp
```

The largest aggregate marginal change under both objectives is termination,
while correctness improves more modestly. These marginal changes are not an
additive causal decomposition of reward. Keep `DeltaC`, `DeltaT`, and
`DeltaR` separate.

## Realized signal allocation

The signal ledger reconstructs each
`(generation_global_step, dataset_index)` prompt group and measures realized
training-signal allocation across the frozen `p0` bins.

Canonical token-level importance sampling is well behaved (ESS/N approximately
0.998). Historical sequence-level-IS collapse and truncation-mask distortions
are retained as instrumentation history and must not be presented as the
canonical token-level phenomenon.

The MaxRL intervention verifies that the signal-allocation shape can be
strongly changed without a matching persistent change in the binwise
correctness-improvement shape.

## Reproduce the current analyses

Use Python module invocation from the repository root.

Core GRPO analyses:

```bash
python -m analyses.ledger_crossfit_signal_allocation
python -m analyses.exposure_split_adjusted
```

Paper-facing tracked GRPO outputs:

```text
analyses/canonical_snapshot_crossfit/
analyses/canonical_ledger_crossfit_signal/
analyses/canonical_exposure_split_transfer/
analyses/canonical_exposure_split_adjusted/
```

Paper-facing tracked MaxRL outputs:

```text
analyses/canonical_maxrl_snapshot_crossfit/
analyses/canonical_maxrl_ledger_crossfit_signal/
analyses/canonical_maxrl_grpo_objective_comparison/
```

Especially useful compact files:

```text
analyses/canonical_exposure_split_adjusted/adjusted_symmetric.csv
analyses/canonical_maxrl_grpo_objective_comparison/objective_comparison.csv
analyses/canonical_maxrl_grpo_objective_comparison/summary.json
analyses/canonical_snapshot_crossfit/aggregate_sanity.csv
analyses/canonical_maxrl_snapshot_crossfit/aggregate_sanity.csv
```

A successful analysis-script marker is a code/data-pipeline check, not a
scientific conclusion by itself.

## Execution lanes and artifact boundary

### M5 Pro development lane

The ordinary development environment is platform-neutral Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r controlled_run/requirements-dev.txt
python -m pytest -q
```

The M5 Pro lane is the default home for unit tests, deterministic data preparation,
analysis, figures/tables, provenance verification, and synthetic MaxRL estimator
tests. CUDA, NCCL, FlashAttention, and canonical TRL/vLLM training do not belong
in the ordinary M5 development environment.

Canonical SFT data are selected with the frozen formatted-token cutoff of 16,384.
Verify a materialized bundle with:

```bash
python -m controlled_run.data_bundle
```

### Artifact boundary

Git stores source code, configs, scientific specs/checkpoints, manifests/hashes,
lightweight derived CSV/JSON tables, and paper-facing figures.

Raw/generated experiment artifacts belong in local storage or private Hugging
Face repositories, including model checkpoints, raw rollout JSONL, raw
signal-ledger JSONL, snapshot-evaluation raw banks, large logs, and large
intermediate files.

Prefer writing raw computational outputs below:

```text
controlled_run_outputs/
```

which is intentionally Git-ignored. Do not use `git add .` as an
experiment-backup mechanism; explicitly stage the small analysis/provenance
files intended for Git.

Canonical remote reference inputs are registered separately from host
infrastructure in:

```text
controlled_run/artifact_registry.json
```

The registry records exact Hugging Face repositories and file paths used by
controlled analyses. Provisioning is explicit and fail-closed rather than part
of RunPod bootstrap. Before the first download of a newly registered bundle,
resolve the remote immutable revision:

```bash
python -m controlled_run.prepare_analysis_inputs \
  --bundle canonical_grpo_seed42_h1_reference \
  --resolve-only
```

Pin that SHA as `expected_revision_sha` in the registry and commit it before
provisioning:

```bash
python -m controlled_run.prepare_analysis_inputs \
  --bundle canonical_grpo_seed42_h1_reference
```

Provisioned files and their SHA256 hashes are recorded under
`controlled_run_outputs/reference_inputs/<bundle>/provision_manifest.json`.

### Optional Apple-Silicon vLLM-Metal runtime

The vLLM-Metal environment remains separate from the ordinary M5 development
environment. It requires **macOS 15+**, native **arm64**, and **Python 3.12**.
The official installer creates `~/.venv-vllm-metal`:

```bash
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
uv pip install --python ~/.venv-vllm-metal/bin/python -r requirements-macos-vllm.txt
```

The historical compatibility smoke used the exact public SFT revision
`checkpoint-8-of-10`; it must not silently substitute an MLX-community
conversion or another checkpoint.

**Metal results are for development, smoke tests, and small exploratory runs.
CUDA vLLM remains the canonical measurement backend.**

### Controlled 2×A40 RunPod workflow

This remains the preferred topology for any future matched seed43/44
replication. No new GPU run is part of the current ATTRIB-writing handoff.

The controlled CUDA lane uses the repository-owned image:

```text
ghcr.io/unitedsnakes/rlvr-vllm
```

For a new image revision, first use the immutable `sha-*` image produced by
GitHub Actions. Promote a tested image to the stable `0.27.1` tag only after
fresh-pod bootstrap and distributed preflight pass.

Recommended active template:

```text
GPU: 2 × NVIDIA A40
Container disk: enough local space for the intended trajectory/snapshots (80 GB recommended)
Network volume: optional; not part of scientific identity
HTTP port: 8888
TCP port: 22
```

RunPod Secrets / environment:

```text
HF_TOKEN={{ RUNPOD_SECRET_huggingface_token }}
GITHUB_DEPLOY_KEY_B64={{ RUNPOD_SECRET_github_rlvr_deploy_key_b64 }}

RLVR_REPO=git@github.com:UnitedSnakes/rlvr_behavior_probe_runpod.git
RLVR_BRANCH=codex/signal-ledger
RLVR_REPO_DIR=/workspace/rlvr_behavior_probe_runpod

RLVR_EXPECT_COMMIT=<approved exact execution commit SHA>
RLVR_RUN_2XA40_PREFLIGHT=1
```

Use the existing Container start command:

```bash
/bin/bash -lc '/start.sh & start_pid=$!; rlvr-bootstrap > /workspace/rlvr-bootstrap.log 2>&1; bootstrap_status=$?; if [ "$bootstrap_status" -ne 0 ]; then printf "[rlvr-bootstrap] startup bootstrap failed with exit %s; pod remains available; rerun rlvr-bootstrap manually\n" "$bootstrap_status" >> /workspace/rlvr-bootstrap.log; fi; wait "$start_pid"'
```

`/start.sh` remains the long-lived RunPod SSH/Jupyter service. Bootstrap
failure is intentionally non-fatal to the pod so the instance remains
inspectable.

For the active controlled lane, bootstrap performs:

```text
GitHub/deploy-key setup
→ branch synchronization
→ exact commit gate when RLVR_EXPECT_COMMIT is set
→ static A40/runtime acceptance
→ fail-closed 2-rank NCCL all-reduce preflight
```

The distributed preflight is equivalent to:

```bash
TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
TORCH_NCCL_DESYNC_DEBUG=1 \
torchrun --nproc_per_node=2 \
  -m controlled_run.distributed_preflight \
  --collective-timeout-seconds 30 \
  --output-json /workspace/rlvr-2xa40-preflight.json
```

It requires a real NCCL all-reduce on the default transport path. Rank-local
values 1 and 2 must both observe sum 3. A pod that fails or hangs in the
default NCCL path is rejected for controlled training.

Do not make `NCCL_P2P_DISABLE=1`, `NCCL_CUMEM_ENABLE=0`, or another
transport workaround part of the canonical runtime merely to make a bad host
pass. Such settings are diagnostic unless a later written infrastructure
decision explicitly changes the canonical transport policy.

Canonical A40 static acceptance can also be run manually:

```bash
python -m controlled_run.runtime_acceptance \
  --attention-backend flash_attention_2
```

An optional real Qwen3 BF16 FlashAttention2 forward/backward probe is:

```bash
python -m controlled_run.runtime_acceptance \
  --attention-backend flash_attention_2 \
  --probe-model
```

For the corrected canonical pre-RL policy, the current Hugging Face download
layout places the actual model one level below the download root. The training
argument must point to the directory that directly contains
`pi0_manifest.json`:

```text
controlled_run_outputs/sft/pi_0/pi_0
```

The required canonical lineage remains:

```text
f89fc90226a67a6a3c7374f9c13abadfcecda88f397ab812fa4130f1f425605b
```

Do not infer the model directory from the outer download folder; verify the
manifest explicitly.

### Historical RunPod image/bootstrap workflow

The earlier template used:

```text
HF_TOKEN={{ RUNPOD_SECRET_huggingface_token }}
GITHUB_DEPLOY_KEY_B64={{ RUNPOD_SECRET_github_rlvr_deploy_key_b64 }}
RLVR_REPO=git@github.com:UnitedSnakes/rlvr_behavior_probe_runpod.git
RLVR_BRANCH=difficulty-bin-analysis
RLVR_REPO_DIR=/workspace/rlvr_behavior_probe_runpod
```

The bootstrap command is `rlvr-bootstrap`; failures were logged to
`/workspace/rlvr-bootstrap.log` without killing the pod. Historical result
backups targeted:

```text
HKReporter/rlvr-behavior-probe-results
```

The old image smoke also used:

```text
--top-k 20
--repetition-penalty 1.1
```

These historical values are retained for reproducibility only and are not
current canonical science settings.

## Authoritative research records

Start with the current correction/sensitivity handoff rather than reading the whole history:

- `docs/superpowers/checkpoints/2026-09-06-attrib-deadline-postoutcome-sensitivity.md`
- `docs/superpowers/checkpoints/2026-09-05-attrib-writing-handoff.md`

Paper-facing post-outcome checkpoints:

- `docs/superpowers/checkpoints/2026-09-03-exposure-split-postoutcome-and-paper-claim-freeze.md`
- `docs/superpowers/checkpoints/2026-09-05-maxrl-h2-h3-postoutcome-gate.md`

Important pre-outcome provenance:

- `docs/superpowers/specs/2026-09-01-signal-allocation-analysis-prereg.md`
- `docs/superpowers/checkpoints/2026-09-02-postrun-preoutcome-analysis-addendum.md`
- `docs/superpowers/checkpoints/2026-09-03-exposure-split-preoutcome-decision.md`
- `docs/superpowers/checkpoints/2026-09-03-cutoff-balance-observed-and-adjustment.md`
- `docs/superpowers/specs/2026-09-03-maxrl-objective-intervention-amendment.md`
- `docs/superpowers/specs/2026-09-04-maxrl-canonical-fixed-panel-preoutcome-addendum.md`

Relevant engineering/evaluator records:

- `docs/superpowers/checkpoints/2026-09-04-maxrl-canonical-structural-pass.md`
- `docs/superpowers/checkpoints/2026-09-04-maxrl-cbank-batching-parity-fail.md`
- `docs/superpowers/checkpoints/2026-09-05-a100-full-replication-aborted-before-scientific-use.md`

Large-artifact / Hugging Face packaging maps:

- `hf_bundles/2026-09-03-canonical-grpo-seed42/README.md`
- `hf_bundles/2026-09-03-canonical-grpo-seed42/manifest.json`
- `hf_bundles/2026-09-05-canonical-maxrl-seed42/README.md`
- `hf_bundles/2026-09-05-canonical-maxrl-seed42/manifest.json`
- `hf_bundles/2026-09-05-canonical-maxrl-seed42/upload_record.json`

## Hugging Face artifact policy

Large checkpoints, raw rollout/ledger artifacts, canonical model snapshots,
and raw C-bank evaluations live in private Hugging Face repos under
`HKReporter/`. Git stores code, configs, manifests, lightweight tables/figures,
and scientific provenance documents.

The canonical MaxRL seed42 backup is remotely verified:

```text
model repo commit:
e67069c666ea372ce4fc4f0dc14617f35a1fce0f

analysis repo commit:
88b7f3244de0c61488ff53dab63d29e2f8669642

313/313 files present
313 size-verified
99 LFS SHA256-verified
```

The GRPO canonical model repository remains the source of truth for its large
training artifacts. The 2026-09-03 GRPO analysis-bundle manifest still records
its own upload status separately; do not infer that status from the verified
MaxRL backup.

Do not call any future HF backup complete until remote presence and available
hash/size checks have been verified.

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
