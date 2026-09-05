# RLVR Behavioral Probe — Project Instructions

Studying RLVR as behavior-probability dynamics, not benchmark optimization.

The current seed42 evidence separates local training signal from local behavioral
change in two ways. Canonical GRPO shows that **question-level behavioral
improvement is not stably localized to direct own exposure**. The matched
practical MaxRL-15 intervention then materially reallocates realized signal
toward lower-p0 questions without a corresponding stable relocation of
question-level correctness improvement. The immediate task is to write this
two-part result clearly and conservatively, not to open a new experiment.

## Read first

Start every session by reading the newest file in `docs/superpowers/checkpoints/`.
The authoritative writing handoff is now:

- `docs/superpowers/checkpoints/2026-09-05-attrib-writing-handoff.md`

For the two paper-facing scientific results, also read:

- `docs/superpowers/checkpoints/2026-09-03-exposure-split-postoutcome-and-paper-claim-freeze.md`
- `docs/superpowers/checkpoints/2026-09-05-maxrl-h2-h3-postoutcome-gate.md`

For canonical MaxRL evaluation provenance, read only when needed:

- `docs/superpowers/checkpoints/2026-09-04-maxrl-canonical-structural-pass.md`
- `docs/superpowers/specs/2026-09-04-maxrl-canonical-fixed-panel-preoutcome-addendum.md`
- `docs/superpowers/checkpoints/2026-09-04-maxrl-cbank-batching-parity-fail.md`

The A100 replication detour is closed:

- `docs/superpowers/checkpoints/2026-09-05-a100-full-replication-aborted-before-scientific-use.md`

Any partial A100 full-run artifacts are engineering/provenance only and must not
enter scientific claims. The preferred future replication plan, if the project
continues, is matched A40 seed43/44 GRPO+MaxRL under the original 2xA40
execution contract.

**Current priority:** write the ATTRIB paper from the already accepted seed42
evidence. Do not start new GPU experiments during drafting merely because
hardware is available.

Do not read the entire historical docs tree at startup. Open older
checkpoints/specs only when the current handoff points to them or the task
requires provenance.

## Current canonical result

Canonical seed42 GRPO starts from the corrected untouched `pi0`:

```text
pi0_lineage_id: f89fc90226a67a6a3c7374f9c13abadfcecda88f397ab812fa4130f1f425605b
GRPO HF repo: HKReporter/rlvr-behavior-probe-grpo-canonical-seed42-2026-09-02
pi0 HF repo: HKReporter/rlvr-behavior-probe-pi0-corrected-canonical-2026-08-30
analysis implementation commit: 386f300e36562ad78063fcfd4b5ed4137325fd9d
```

The fixed 256-question train panel was evaluated throughout training. At frozen 25/45/65 cutoffs, questions were split by whether their unique own training exposure had already occurred. The split definition, interpretation scale, balance audit, and covariate-adjusted estimator were frozen before reading the exposed-vs-unexposed outcomes.

Primary conclusion:

```text
no stable own-exposure advantage
```

Substantial correctness improvement occurs among questions not yet directly sampled for training. Across 15 adjusted symmetric cutoff x p0-bin cells there are 8 `transfer_compatible`, 3 `unexposed_higher`, 2 `mixed_or_uncertain`, 2 `not_classifiable`, and 0 `own_exposure_candidate` cells under the pre-frozen descriptive rule.

This is evidence against a **strong prompt-local account** in which directly sampled questions should systematically improve more. It is not a randomized causal estimate and does not prove that cross-question transfer is the unique or causally dominant channel.

## Current scientific chain

Keep these quantities separate:

```text
objective
  -> realized training-signal allocation
  -> parameter update
  -> shared-parameter transfer / interference
  -> behavioral-change allocation
```

Do not equate local signal allocation with eventual local behavioral movement.

Paper limitation:

> All behavioral and signal-allocation comparisons reported here are from a
> single matched training seed. We therefore do not estimate between-seed
> variability in either bin-level behavioral changes or objective-induced
> signal reallocation.

The paper-facing claim hierarchy is frozen in the latest checkpoint. In particular:

1. main result: no stable own-exposure advantage;
2. supporting result: realized signal allocation and correctness movement have different allocation shapes;
3. implementation distortions are supporting instrumentation evidence, not the headline;
4. `DeltaT`, `DeltaC`, and `DeltaR` must remain separated.

## MaxRL status

Practical MaxRL-15 is complete as a canonical matched seed42 objective
intervention.

Frozen implementation semantics:

```text
G = N = 16
effective MaxRL order = 15
K = sum_i r_i
K = 0: A_i = 0
K > 0: A_i = (r_i - K/16) / (K/16)
epsilon = 0
```

Canonical training structural acceptance:

```text
steps = 3736
rows = 119552
groups = 7472
rank_files = 2
snapshots = 20
max_advantage_error = 1.5894571969710114e-07
aggregate_token_is_ess_fraction = 0.9980541312524671
nonfinite_numeric_fields = 0
status = PASS
```

The canonical fixed train-256 K=16 C-bank evaluation completed under the same
sequential evaluator structure used for canonical GRPO.

Frozen scientific judgment:

```text
H1 mechanism gate: SUPPORTED
H2 primary behavioral prediction: SUPPORTED
H3 alternative as the primary explanation: NOT SUPPORTED
H4 diagnostic stop: NOT ACTIVE
```

At the 100% endpoint, MaxRL/GRPO cumulative signal ratios across increasing
frozen p0 bins are:

```text
0          2.205x
(0,.25]    1.720x
(.25,.5]   0.988x
(.5,.75]   0.592x
(.75,1)    0.463x
```

The corresponding MaxRL-minus-GRPO DeltaC contrasts are:

```text
0         -4.712 pp
(0,.25]   -0.437 pp
(.25,.5]  -1.908 pp
(.5,.75]  -0.620 pp
(.75,1)   -0.156 pp
```

The full 5%-through-100% trajectory shows the same qualitative separation:
signal allocation is persistently left-shifted, while
`DeltaC_MaxRL - DeltaC_GRPO` fluctuates rather than moving persistently in the
corresponding direction.

Paper-safe conclusion:

> Changing the objective materially reallocates realized training signal, but
> question-level correctness improvement does not correspondingly and stably
> reallocate.

Do not claim that representations uniquely cause this effect, that transfer is
the unique causal channel, or that objective allocation can never affect
behavioral allocation. The supported interpretation is that shared-parameter
transfer/interference substantially mediates the mapping from local signal to
local behavioral change.

Keep `DeltaC`, `DeltaT`, and `DeltaR` separate. Canonical MaxRL aggregate
movement is approximately:

```text
DeltaR = +16.63 pp
DeltaT = +29.11 pp
DeltaC =  +5.95 pp
```

The original GRPO exposed-vs-unexposed finding remains the main paper result;
MaxRL is the second independent intervention.

Canonical MaxRL large artifacts are backed up and remotely verified:

```text
model repo:
HKReporter/rlvr-behavior-probe-maxrl-canonical-seed42-2026-09-05
commit:
e67069c666ea372ce4fc4f0dc14617f35a1fce0f

analysis repo:
HKReporter/rlvr-behavior-probe-maxrl-analysis-seed42-2026-09-05
commit:
88b7f3244de0c61488ff53dab63d29e2f8669642

verification:
313/313 files present
313 size-verified
99 LFS SHA256-verified
```

For the frozen hypothesis and outcome records, read:

- `docs/superpowers/specs/2026-09-03-maxrl-objective-intervention-amendment.md`
- `docs/superpowers/checkpoints/2026-09-05-maxrl-h2-h3-postoutcome-gate.md`
- `hf_bundles/2026-09-05-canonical-maxrl-seed42/README.md`

The A100 no-checkpointing/vLLM tuning path and six-run A100 replication plan
were both closed before scientific use. Do not revive them during paper
drafting.

## Designs and amendments

Earlier controlled-run amendments remain in force where they do not conflict with later documents:

- `docs/superpowers/specs/2026-08-26-qwen3-controlled-rlvr-design.md`
- `docs/superpowers/specs/2026-08-27-long-context-sft-compute-amendment.md`
- `docs/superpowers/specs/2026-08-29-grpo-completion-cap-amendment.md`
- `docs/superpowers/specs/2026-08-29-grpo-2xa40-batch-semantics-amendment.md`
- `docs/superpowers/specs/2026-08-30-grpo-truncation-policy-amendment.md`
- `docs/superpowers/specs/2026-09-01-signal-allocation-analysis-prereg.md`
- `docs/superpowers/specs/2026-09-03-maxrl-objective-intervention-amendment.md`

Later amendments supersede earlier ones only where they explicitly conflict.

## Hard rules

**Frozen values are frozen.** Any value in `GRPO_INVARIANTS` or `SFT_INVARIANTS` (`controlled_run/config.py`) is a scientific commitment. Never change one to make a run fit, a test pass, or an OOM disappear. A change requires a written amendment first.

**Pre-outcome and post-outcome records stay separate.** Never rewrite an old preregistration/checkpoint to make it match a later result. Add a dated post-outcome checkpoint or amendment instead.

**Gates are fail-closed by design.** Data hashes, lineage checks, runtime acceptance, FlashAttention2, topology, and canonical-mode checks must not gain silent fallback paths.

**Green tests are not scientific evidence.** A passing suite proves code semantics only. Scientific validation requires the relevant run/ledger/outcome checks.

**Never infer truncation or special tokens from decoded text.** Use token ids or sampler finish reason.

**Pilot and shakedown outputs are disposable.** They never enter the canonical lineage. Canonical objective runs restart from the untouched corrected `pi0` unless a later written design explicitly changes this.

**Do not overclaim the exposure result.** Say "no stable own-exposure advantage," "evidence against a strong prompt-local account," or "consistent with substantial cross-question transfer." Do not say transfer has been causally proven dominant.

## Closed questions — do not reopen casually

The Qwen3 thinking-control grammar (`<think>` / `</think>`) was measured and closed on 2026-08-30. Do not treat it as a blocker or propose SFT changes aimed at fixing it.

Extending the completion horizon does not repair the canonical truncation issue. That path is closed unless new evidence specifically reopens it.

Historical sequence-level IS and truncation masking effects are instrumentation history. Canonical token-level IS has ESS/N approximately 0.998; do not present the historical sequence-level ESS collapse as a canonical phenomenon.

## Compute lanes

M5 Pro / ordinary local development:

```text
pytest
analysis scripts
figures/tables
provenance and bundle verification
deterministic data preparation
MaxRL estimator derivation and synthetic tests
```

The ordinary M5 `.venv` must remain independent of CUDA, NCCL,
FlashAttention, canonical TRL/vLLM training, and vLLM-Metal. The optional
Apple-Silicon vLLM-Metal runtime remains a separate development/smoke
environment.

Controlled A40 lane:

```text
static runtime acceptance
2×A40 distributed NCCL preflight
canonical CUDA evaluation
GRPO/MaxRL pilot and shakedown
canonical GRPO/MaxRL training
```

Canonical SFT and canonical GRPO used 2 × A40. Do not silently change canonical
batch/topology semantics.

Before controlled 2-GPU training, require the default NCCL path to pass:

```bash
torchrun --nproc_per_node=2 \
  -m controlled_run.distributed_preflight
```

A failed default NCCL/P2P path rejects the pod. Do not make
`NCCL_P2P_DISABLE=1` a canonical runtime workaround.

## Artifact registry

Git contains code, configs, scientific specs/checkpoints, manifests/provenance,
lightweight derived tables, and paper-facing figures.

Private Hugging Face/local storage contains model checkpoints, raw rollout
JSONL, raw signal ledgers, snapshot raw banks, and large logs/intermediate
artifacts.

Prefer raw computational outputs under `controlled_run_outputs/`, which is
Git-ignored. Do not use `git add .` as an experiment-backup strategy.

For the analysis packaging maps, read:

- `hf_bundles/2026-09-03-canonical-grpo-seed42/README.md`
- `hf_bundles/2026-09-03-canonical-grpo-seed42/manifest.json`
- `hf_bundles/2026-09-05-canonical-maxrl-seed42/README.md`
- `hf_bundles/2026-09-05-canonical-maxrl-seed42/manifest.json`

## RunPod execution contract

The active controlled branch is:

```text
codex/signal-ledger
```

The active template may set:

```text
RLVR_EXPECT_COMMIT=<approved exact execution commit SHA>
RLVR_RUN_2XA40_PREFLIGHT=1
```

Bootstrap keeps `/start.sh` alive, verifies the requested Git commit, runs
static A40 acceptance, and then runs the real two-rank NCCL all-reduce
preflight. The distributed preflight writes:

```text
/workspace/rlvr-2xa40-preflight.json
```

A pod that fails the default collective path is rejected. P2P/cuMem transport
overrides are diagnostic only unless a later written amendment changes that
policy.

The corrected canonical pi0 path passed to training must be the directory that
directly contains `pi0_manifest.json`; under the current HF download layout:

```text
controlled_run_outputs/sft/pi_0/pi_0
```

Required lineage:

```text
f89fc90226a67a6a3c7374f9c13abadfcecda88f397ab812fa4130f1f425605b
```

For the infrastructure rationale and failure record, read:

- `docs/superpowers/specs/2026-09-03-m5-a40-execution-infra-amendment.md`

## Commands

Use module invocation for analysis scripts:

```bash
python -m pytest -q
python -m analyses.ledger_crossfit_signal_allocation
python -m analyses.exposure_split_adjusted
```

Direct `python analyses/foo.py` invocation can fail package imports; prefer `python -m analyses.foo`.

Long GPU jobs should run under `tmux` or a detached non-interactive launcher. Never rely on a foreground terminal session for canonical compute.

## Working style

Say "I was wrong" directly when backing off a claim.
Do not soften a failed acceptance check into partial success.
When a result contradicts the planned interpretation, record the contradiction before designing the next intervention.
Keep code/CPU verification separate from scientific validation.
