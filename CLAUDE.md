# RLVR Behavioral Probe — Project Instructions

Studying RLVR as behavior-probability dynamics, not benchmark optimization.
Central question: given a behavior's pre-RL probability `p0`, how does the RL
objective allocate learning signal and move that probability? The eventual
comparison is GRPO vs MaxRL as two objective interventions over the same
exact `pi_0`.

## Read first

Start every session by reading the newest file in
`docs/superpowers/checkpoints/`. That is the handoff record and it is
authoritative about current state.

Design and amendments, in force order (later amendments supersede earlier
ones where they conflict):

- `docs/superpowers/specs/2026-08-26-qwen3-controlled-rlvr-design.md`
- `docs/superpowers/specs/2026-08-27-long-context-sft-compute-amendment.md`
- `docs/superpowers/specs/2026-08-29-grpo-completion-cap-amendment.md`
- `docs/superpowers/specs/2026-08-29-grpo-2xa40-batch-semantics-amendment.md`
- `docs/superpowers/specs/2026-08-30-grpo-truncation-policy-amendment.md`

Do not read all of these at startup. Read the newest checkpoint, then open
only the amendment relevant to the current task.

## Hard rules

**Frozen values are frozen.** Any value in `GRPO_INVARIANTS` or
`SFT_INVARIANTS` (`controlled_run/config.py`) is a predeclared scientific
commitment. Never change one to make a run fit, a test pass, or an OOM go
away. Changing one requires a written amendment in
`docs/superpowers/specs/` first, with the evidence that forced it. The
amendment comes before the code.

**Gates are fail-closed by design.** Data-bundle hash checks, the A40 runtime
acceptance gate, the FlashAttention2 check, and the 2xA40 topology gate exist
to stop silent substitution. Never weaken, skip, or add a fallback path to a
gate. If a gate fails, report it and stop.

**Green tests are not scientific evidence.** A passing suite means the code
does what it says. It never means a training run worked. Do not describe a
run as validated on the strength of tests.

**Never infer truncation or special tokens from decoded text.** vLLM omits
special tokens from text output by default. Use token ids or the sampler
finish reason. This already caused one misclassification.

**Pilot and shakedown outputs are disposable.** They never enter the
canonical lineage. Canonical GRPO always restarts from an untouched `pi_0`.

## Closed questions — do not reopen

The Qwen3 thinking-control grammar (`<think>` / `</think>`) is a closed
investigation. Measured 2026-08-30 on 256 GSM8K prompts: `<think>` appears in
1.05% of rollouts and `</think>` in 1.22%, while sample accuracy is 53.88%
and 85.94% of prompts have interior `p0`. The grammar is absent and affects
nothing this project measures. Do not treat it as a blocker, and do not
propose SFT changes aimed at fixing it.

Extending the completion horizon does not fix truncation. Of 55 trajectories
clipped at 2048, only 11 stopped naturally by 8192. Do not propose raising
the cap.

## Compute lanes

CPU / local dev — pytest, `prepare_data`, contamination and length audits,
data-bundle verification, provenance, all analysis. Use the project `.venv`
with `controlled_run/requirements-dev.txt`.

A40 — runtime acceptance, SFT smoke, canonical SFT, `p0` sampling, GRPO
pilot and canonical GRPO, canonical CUDA evaluation.

Canonical SFT and canonical GRPO both require **2 x A40**. A single A40 OOMs
on GRPO at `max_completion_length=2048`. Do not propose single-GPU canonical
runs.

## Artifact registry

Upstream models and datasets are pinned to exact commit SHAs. Generated
artifacts go to private Hugging Face repos under `HKReporter/` with a
`SHA256SUMS.txt`. Git holds code, configs, manifests, lightweight summaries,
and docs. Large checkpoints and raw rollouts never go in git.

## Commands

```bash
python -m pytest -q                          # full CPU suite
python -m controlled_run.data_bundle         # verify frozen SFT data bundle
python -m controlled_run.runtime_acceptance --attention-backend flash_attention_2
```

Long GPU jobs: run under tmux, or `nohup ... < /dev/null > log 2>&1 &`.
Backgrounded jobs that touch the terminal get stopped by SIGTTOU.

## Working style

Say "I was wrong" directly when backing off a claim; no hedging.
Do not soften a failed acceptance check into a partial success.
When a result contradicts the plan, say so before proposing the next step.
