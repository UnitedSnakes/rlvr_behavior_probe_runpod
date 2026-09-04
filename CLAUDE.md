# RLVR Behavioral Probe — Project Instructions

Studying RLVR as behavior-probability dynamics, not benchmark optimization.

The current question is no longer only "given pre-RL probability `p0`, where does the objective allocate signal?" The canonical GRPO result shows that **question-level behavioral improvement is not stably localized to direct own exposure**. The next objective-level question is whether changing realized signal allocation with MaxRL actually changes where the model improves.

## Read first

Start every session by reading the newest file in `docs/superpowers/checkpoints/`. It is the authoritative handoff record.

As of 2026-09-03 the current scientific post-outcome handoff is:

- `docs/superpowers/checkpoints/2026-09-03-exposure-split-postoutcome-and-paper-claim-freeze.md`

The current live MaxRL execution handoff is:

- `docs/superpowers/checkpoints/2026-09-03-maxrl-20step-gpu-pilot-pass.md`

For the MaxRL scientific hypothesis, also read:

- `docs/superpowers/specs/2026-09-03-maxrl-objective-intervention-amendment.md`

Do not read the entire historical docs tree at startup. Open older checkpoints/specs only when the current handoff points to them or the task requires provenance.

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

The paper-facing claim hierarchy is frozen in the latest checkpoint. In particular:

1. main result: no stable own-exposure advantage;
2. supporting result: realized signal allocation and correctness movement have different allocation shapes;
3. implementation distortions are supporting instrumentation evidence, not the headline;
4. `DeltaT`, `DeltaC`, and `DeltaR` must remain separated.

## MaxRL status

Practical MaxRL-15 is implemented and has passed the CPU/TDD implementation
gate. The estimator changes only the group advantage while preserving the
matched canonical GRPO outer stack.

Frozen implementation semantics:

```text
G = N = 16
effective MaxRL order = 15
K = sum_i r_i
K = 0: A_i = 0
K > 0: A_i = (r_i - K/16) / (K/16)
epsilon = 0
```

The real 2×A40 20-step engineering pilot has passed its frozen structural
acceptance gate after one pre-step integration failure exposed that real TRL
1.12 stores `_logs["advantages"]` in a `deque`, not the list used by the
earlier CPU test double.

Accepted 20-step pilot summary:

```text
steps = 20
rows = 640
groups = 40
rank_files = 2
group_size = 16
max_advantage_error = 1.5894571969710114e-07
aggregate_token_is_ess_fraction = 0.9979146076241244
status = PASS
```

This remains an engineering/runtime result only. The next gate is a disposable
150-step MaxRL shakedown from the untouched corrected canonical pi0. Do not
treat the 20-step pilot as a scientific MaxRL outcome.

Before GPU interpretation, read:

- `docs/superpowers/specs/2026-09-03-maxrl-objective-intervention-amendment.md`
- `docs/superpowers/specs/2026-09-03-maxrl-practical-estimator-implementation-amendment.md`
- `docs/superpowers/checkpoints/2026-09-03-maxrl-implementation-ready-for-gpu-pilot.md`
- `docs/superpowers/checkpoints/2026-09-03-maxrl-pilot-acceptance-checker-complete.md`
- `docs/superpowers/checkpoints/2026-09-03-maxrl-20step-gpu-pilot-pass.md`

The pre-outcome hypothesis hierarchy remains:

1. H1 mechanism gate: realized signal allocation should qualitatively shift toward lower `p0` relative to GRPO if MaxRL is implemented faithfully.
2. H2 primary behavioral prediction: signal allocation may move materially while `DeltaC` allocation changes much less.
3. H3 predeclared alternative: if signal and behavior both shift, objective allocation influences behavioral allocation.
4. H4: if signal does not move as expected, stop interpretation and inspect estimator/implementation.

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

For the 2026-09-03 analysis packaging map, read:

- `hf_bundles/2026-09-03-canonical-grpo-seed42/README.md`
- `hf_bundles/2026-09-03-canonical-grpo-seed42/manifest.json`

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
