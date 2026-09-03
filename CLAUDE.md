# RLVR Behavioral Probe — Project Instructions

Studying RLVR as behavior-probability dynamics, not benchmark optimization.

The current question is no longer only "given pre-RL probability `p0`, where does the objective allocate signal?" The canonical GRPO result shows that **question-level behavioral improvement is not stably localized to direct own exposure**. The next objective-level question is whether changing realized signal allocation with MaxRL actually changes where the model improves.

## Read first

Start every session by reading the newest file in `docs/superpowers/checkpoints/`. It is the authoritative handoff record.

As of 2026-09-03 the current post-outcome handoff is:

- `docs/superpowers/checkpoints/2026-09-03-exposure-split-postoutcome-and-paper-claim-freeze.md`

For the next MaxRL phase, also read:

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

There is **no MaxRL implementation or validated estimator yet**.

Do not treat MaxRL as a config toggle on `train_grpo.py`. Before any GPU run, follow `docs/superpowers/specs/2026-09-03-maxrl-objective-intervention-amendment.md`.

The pre-outcome hypothesis hierarchy is:

1. H1 mechanism gate: realized signal allocation should qualitatively shift toward lower `p0` relative to GRPO if MaxRL is implemented faithfully.
2. H2 primary behavioral prediction: signal allocation may move materially while `DeltaC` allocation changes much less.
3. H3 predeclared alternative: if signal and behavior both shift, objective allocation influences behavioral allocation.
4. H4: if signal does not move as expected, stop interpretation and inspect estimator/implementation.

The exact finite-G estimator, scale/temperature parameter, interaction with DAPO normalization, and number of paired seeds are intentionally not frozen yet. Re-derive them from the source paper before code.

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

CPU / local dev:

```text
pytest
analysis scripts
figures/tables
provenance and bundle verification
MaxRL estimator derivation and synthetic tests
```

A40:

```text
runtime acceptance
canonical CUDA evaluation
MaxRL pilot/shakedown/canonical only after implementation gate
```

Canonical SFT and canonical GRPO used 2 x A40. Do not silently change canonical batch/topology semantics.

## Artifact registry

Large generated artifacts live in private Hugging Face repos under `HKReporter/`. Git holds code, configs, manifests, lightweight analysis summaries, and scientific provenance docs. Large checkpoints/raw rollouts do not belong in git.

For the 2026-09-03 analysis packaging map, read:

- `hf_bundles/2026-09-03-canonical-grpo-seed42/README.md`
- `hf_bundles/2026-09-03-canonical-grpo-seed42/manifest.json`

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
