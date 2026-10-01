# Pre-registration: camera-ready replication runs (seeds 43, 44, 45)

Status: **FROZEN** 2026-10-01 ~06:35 UTC, before the first real training run
(draft with all definitions pushed earlier in commit `b6b7753`). SHA-256 and
commit of this file are recorded in `PROGRESS.md`.

Everything in this document is fixed before any new-seed outcome exists. The
analyses of the existing seed-42 data in §F are post-outcome descriptive
analyses of the discovery run and are labelled so wherever they appear.

## 1. Bridge status and C4

Bridge verdict at freeze time: **pending** — π0 bridged on the box, not a GROSS
FAIL (Δ R −0.24, T −0.28, C −0.52 pp; all |Δ/SE| < 1); the eight seed-42
snapshots are still to be evaluated
(rules: `PREREG_BRIDGE.md`, pushed in commit `556a271`). The verdict changes
only (i) which π0 banks supply baselines and (ii) whether a supplementary mixed
estimate is shown (C4, verbatim in `PREREG_BRIDGE.md`). It never changes which
runs are trained or which pairs form the main sample. Until the bridge is final,
analyses are computed both ways and only the C4-selected version is reported.

Under NOT PASSED the baseline bank for direction "A-bin" is the regenerated box
bank B (and vice versa). The adjustment covariates "opposite-bank baseline
reward success" and "opposite-bank baseline completion length" are taken from
the same bank that supplies the baseline. Bin membership is always the paper's
frozen membership from the original A40 banks (Table 4 counts).

## 2. Code, patch and environment

- Training: checkout of `981475795538eee391c7e86aa022ee609b539770` (MaxRL
  seed-42 execution commit; GRPO path identical in behaviour to the GRPO
  seed-42 commit, `INVENTORY.md` §8). Both arms of a pair run from this one
  checkout in one environment.
- Evaluation: checkout of `1c26b1f0f3c5f6ea1187fd00318587388a891272`
  (canonical sequential evaluator; π0 bank collector `diagnose_p0_signal_budget.py`
  identical to `5b10742`).
- Camera-ready code: `camera_ready/launch_train.py`, `camera_ready/eval_run.py`
  at camera-ready commit `fd8252e3dc433f3fd3dad9d3cee51fcc2d9f6337` (SHA-256
  `launch_train.py` `f70de1a7…`, `eval_run.py` `75ab4443…`). Runtime patches
  (approved deviation): the seed invariant 42 → S; logging-only
  `INSTRUMENTATION.diff` (SHA-256 `a5db93bf…87ab`): `logging_steps = 1`, a rank-0
  log writer with read-only CUDA memory counters, and a dump of the resolved
  Trainer args.
- Environment lock: `/root/autodl-tmp/envs/rlvr`, Python 3.12.13, torch
  2.13.0+cu130, vLLM 0.27.1, transformers 5.15.0, datasets 5.0.1, accelerate
  1.14.0, TRL 1.12.0, flash-attn 2.8.3.post1 (built from source,
  `FLASH_ATTN_CUDA_ARCHS=80`, nvcc 13.0.48), NCCL 2.29.7; full `pip freeze`
  SHA-256 `43eaab61…59a9` (file `camera_ready/box/pip_freeze.txt`, 220 lines).
- Resource-only settings (logged; not part of the recipe): colocated vLLM
  `vllm_gpu_memory_utilization` is part of the frozen config (0.30). On the
  32 GB cards it is **kept at 0.30** (A40 absolute budget would need 0.45 and
  leaves too little for training; the frozen config check stays intact).
  Training processes run with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  (allocator only). Smoke peak allocated memory 29.9 GiB of 31.5 GiB per card,
  reached at step 2 and flat afterwards.
  Evaluator `gpu_memory_utilization` 0.50 (default). GPU pairs `0,1` and `2,3`.
  `WANDB_MODE=offline` (training reports to `none` anyway). Hub offline after
  caching. Snapshot retention below.

## 3. Runs

- Seeds 43, 44, 45 in this order; one GRPO and one MaxRL run per seed, trained
  concurrently. GPUs `0,1` go to GRPO for seed 43, to MaxRL for seed 44, to
  GRPO for seed 45 (alternating); the other arm gets `2,3`.
- Each run: `torchrun --nproc_per_node=2 camera_ready/launch_train.py
  --objective {grpo|maxrl} --seed S --pi0-dir <pi0> --output-dir <run>`.
  The only config difference from seed 42 is `seed: S`.
- What the seed changes (documented, not altered): the RepeatSampler order of
  the 7,472 training groups (shared by both arms of a pair) and the trainer /
  torch RNG. The colocated vLLM engine keeps TRL's fixed rank seed
  (`process_index`), as in the seed-42 runs.

## 3a. Smoke-test results used for this freeze (D2–D6)

Seed-42 configs of both arms through the launcher, concurrently (GRPO GPUs 0,1;
MaxRL GPUs 2,3; verified by GPU UUID), stopped at step 40, scratch outputs.
Both exit 0, no OOM. Resolved Trainer args equal the A40 MaxRL seed-42
`training_args.bin` in every field except `logging_steps` (10 → 1) and
`output_dir`; world size 2, per-device batch 4, accumulation 4,
`steps_per_generation` 4, 32 responses per step, caps 512/2,048/2,560,
temperature 0.8, top-p 1.0. Run manifests equal the seed-42 manifests in every
field. LR at steps 10–40 equals the A40 log. Question order of steps 0–39 equals
seed 42's. Ledger schema (36 fields) equal. Structural audit on the 40 steps:
no non-finite fields, max advantage error 2.24e-7 (GRPO) / 1.59e-7 (MaxRL),
token IS ESS/N 0.99801 / 0.99801. D4 comparison with the A40 seed-42 ledger,
steps 0–39 (80 groups each, group-clustered SEs): all |z| ≤ 1.43 for R, T, C,
length, cap-hit rate and live-group share — no alert. Throughput with both arms
concurrent: 17.41 (GRPO) and 17.68 (MaxRL) s/step over steps 1–40 (A40 seed-42
MaxRL: 20.38). Projected run time = 17.68 × 0.921 (A40 wall-clock
overall/first-40 ratio) × 3,736 s = 16.9 h. Initial E_eval for the time rule:
1.0 h (A40 K=16 snapshot evaluation took 0.78 h), replaced by measured box
values as soon as they exist.

## 4. Evaluation protocol

- `EVAL_STEPS` = 934, 1681, 2428, 3736 (snapshots `pi_025`, `pi_045`,
  `pi_065`, `pi_100`). K = 16 per question, one question per request, unchanged
  sampling settings, 2,048-token cap.
- Per-question seed for pair S: `S·100000 + i + 75000`, identical for both arms
  of the pair (Sam decision 1).
- Extra endpoint batches b = 1, 2, 3 at step 3736 for both arms: per-question
  seed `S·100000 + i + 75000 + b·1,000,000`. Collision check against every seed
  family used before (π0 banks A/B, seed-42 snapshots, all historical
  `42·100000 + i` diagnostics) and among the new families:
  `camera_ready/tools/check_seed_collisions.py` → none (whole train split,
  indices 0–7,472).
- Order (Sam decision 4, deviation from the brief's per-pair order): after each
  pair finishes training, both arms' K=16 endpoint evaluations; the K=16
  evaluations at 934/1681/2428 of all pairs after all training ends; extra
  endpoint batches after all required evaluations, skipped if time runs out.
  Implementation (`camera_ready/box/queue_runner.py`): when one arm finishes
  before the other, its integrity check and endpoint evaluation start at once on
  its two freed GPUs; one seed-42 bridge snapshot evaluation uses the other freed
  GPU (bridge evaluations fill otherwise idle GPUs and never delay a pair's
  start beyond the endpoint evaluations). Remaining bridge evaluations run after
  the deferred protocol evaluations. An extra-batch round (≤ 4 jobs) starts only
  if it is projected to finish by RESULTS_DUE − 6.5 h. A failed evaluation is
  re-run once from scratch (the evaluator never appends).

## 5. Time rule (Sam decision 3)

Seed 43 starts unconditionally. Seed 44 (then 45) starts only if, at its
start time t₀ and at the measured throughput,

- t₀ + T_train ≤ TRAIN_CUTOFF (2026-10-03 16:00 UTC), and
- t₀ + T_train + E_end + ⌈6k/4⌉·E_eval + 1.5 h (analysis) + 3.0 h
  (independent fact-check, fixes, re-check) + 2.0 h (buffer) ≤ RESULTS_DUE
  (2026-10-04 00:00 UTC),

where k = number of pairs launched including this one, T_train = the longer
arm's measured training wall time of the most recent completed pair (before any
pair has completed: the smoke projection, 16.9 h),
E_end = E_eval = the longest measured wall time of one K=16 evaluation on one
GPU on this box (1.0 h before any is measured); the endpoint evaluations of a
pair run in parallel and the deferred evaluations run four at a time. If the rule fails, the pair is not
launched and Sam is told. The rule is evaluated with the latest measured
throughput, never with outcomes.

## 6. Validity, integrity and exclusion (E3, verbatim intent)

A pair is valid, and must be reported, once both arms have finished training,
passed the structural audit and have the four K=16 protocol evaluations.
Missing extra endpoint batches never exclude a pair. Infrastructure failure of
training is the only ground for exclusion.

Integrity checks per run: 3,736 optimizer steps; ledger with 2 rank files,
119,552 rows, 7,472 groups of 16, every generation step 0–3,735 with 16 rows
per rank; `pi_025/045/065/100` present with `actual_step` 934/1681/2428/3736;
structural audit (`camera_ready/analysis/core.structural_audit`, the seed-42
checks applied to each arm with its own estimator): zero non-finite numeric
fields, max |advantage − reconstruction| ≤ 1e-6, aggregate token IS ESS/N in
(0, 1]. ESS/N < 0.99 is an **alert** (configuration review); ESS/N < 0.95 is a
**stop**. The seed-42 values are 0.99806 (GRPO) and 0.99805 (MaxRL).

Failure handling: an arm that dies for an infrastructure reason (OOM, node
error, disk, network) is restarted from scratch once with the same seed if the
time rule allows; never resumed. A second death stops the queue.

Snapshot retention: after a pair's integrity checks pass, policy snapshots
other than `pi_025/045/065/100` and the HF Trainer checkpoints (optimizer
state) are deleted; `trainer_state.json` files are kept.

## 7. Estimands (per pair)

Notation as in the paper. Bins: the frozen cross-fit bins (five supported bins
0, (0,.25], (.25,.5], (.5,.75], (.75,1); bin 1 omitted from bin comparisons).
Symmetric = equal-weight average of the A-bin and B-bin direction means.
Panel rates pool all responses of the 256 questions.

- d_b^X(t) = X_{M,b}(t) − X_{G,b}(t), q_b^X(t) = d_b^X(t) − [X_{M,panel}(t) − X_{G,panel}(t)],
  X ∈ {C, R, T}, t ∈ EVAL_STEPS. Baselines cancel.
- ΔX_{arm,b}(t) = X_{arm,b}(t) − baseline (C4 bank), reported, not primary.
- Advantage mass S_t(b) = (1/N_b) Σ_{q∈b} 1{s_q < t} Σ_i |A_{q,i}| (paper
  Appendix D), symmetric; mass ratio S^M_t(b)/S^G_t(b).
  Mass-share ratio = [S^M_t(b)/Tot^M_t] / [S^G_t(b)/Tot^G_t], Tot_t = Σ |A| over
  all ledger groups with generation step < t (all 7,472 training questions).
- A panel question that the run never samples (one of the 7,473 rows is
  dropped by the 7,472-group schedule) has s_q = +∞: unexposed at every cutoff,
  zero mass, still counted in N_b.
- **Primary cells (8):** t = 3736, b ∈ {0, (0,.25]}, X ∈ {C, R}, both d and q.
  Everything else is secondary.
- **Endpoint K:** if every pair in the main sample has all 64 endpoint
  responses (16 protocol + 3 × 16 extra), the primary version uses K = 64 for
  all pairs; otherwise K = 16 for all. K is never mixed within a sample. The
  K = 16 table is shown either way. The discovery pair is always K = 16.
- **Pre-exposure (GRPO arm of each pair):** the paper's frozen adjustment model
  (Appendix C) unchanged: per cutoff (934, 1681, 2428) × bin × direction, OLS of
  ΔC on exposure and the three standardized covariates (opposite-bank baseline
  reward success, opposite-bank baseline completion length, prompt token
  count; constant covariates dropped; cells with fewer than 2 exposed or 2
  unexposed questions skipped), adjusted means at the pooled covariate mean,
  directions averaged. Reported: adjusted not-yet-exposed gain ΔC_U in
  (0,.25] at each cutoff, and U − E. Prompt token counts are the frozen
  per-question values from the seed-42 balance table
  (`camera_ready/analysis/prompt_token_counts.json`).
- **Clipping diagnostics (per arm and pair):** from the step log, per optimizer
  step g: pre-clip norm n_g, clip coefficient c_g = min(1, 1.0/(n_g + 1e-6)),
  learning rate. Optimizer step g consumes ledger generation step g − 1.
  Reported: share of steps with c_g < 1; quantiles (0, 5, 25, 50, 75, 95,
  100 %) and mean of c_g; clip-weighted mass S^clip_t(b) = S_t(b) with each
  group's |A| multiplied by c of the step that consumed it, and its M/G ratio —
  labelled "log proxy, not a measure of parameter contribution". Seed 42: MaxRL
  logged every 10th step only (share and quantiles over logged steps; no
  clip-weighted mass); GRPO seed-42 logs were not kept (not available).

## 8. Statistics and reporting (G)

- Main inference: valid box pairs only. The A40 seed-42 pair is always its own
  row ("submitted discovery run"). A pooled "mixed" estimate (A40 pair + box
  pairs, K = 16, labelled as mixing the discovery run with replications across
  hardware) appears only as a supplement and only under bridge PASS.
- Every pair's own value is shown. Across the main sample (n pairs): mean and
  two-sided 95 % t-interval, mean ± t(0.975, n−1)·SD/√n, SD with ddof = 1.
  With n < 3: values listed, no interval. Primary cells also get the one-sided
  95 % upper bound mean + t(0.95, n−1)·SD/√n (n ≥ 3 only).
- Within-pair conditional sampling intervals (§F1 method) may accompany each
  pair's values; they are never presented as seed uncertainty.
- Wording for each primary cell, δ = `DELTA_REF` = 3 pp, from the main sample's
  two-sided interval (open comparisons):
  - interval entirely above 0 → "additional gain under MaxRL" (the paper's
    Test-2 sentence must then change);
  - entirely below 0 → "lower under MaxRL";
  - contains 0 and lies inside (−δ, δ) → "no additional gain larger than δ pp detected";
  - contains 0 and reaches beyond ±δ → "unresolved at this number of runs";
  - n < 3 (no interval) → "unresolved at this number of runs".
  The eight intervals carry no multiplicity adjustment, and RESULTS says so.
  Banned wording: "equivalent", "no effect", "not predictive", "per unit of
  advantage mass", or any causal decomposition of the gain; "not established"
  is never written as "rejected".
- Reweighting is "realized" if the mass ratio at t = 3736 is above 1 in both
  bin 0 and bin (0,.25] in every pair; whether this also holds at all four
  EVAL_STEPS is reported. Numbers are reported either way.
- Pre-exposure gain: per seed. "Replicated" only if ΔC_U in (0,.25] is
  positive at all three cutoffs in every new seed.
- Random-schedule exposure contrast (supplementary): adjusted U − E in (0,.25]
  at each cutoff, per seed; across-seed mean with a two-sided 95 % t-interval
  over seeds (training trajectories are the units; n < 3 → no interval).
  Described as the average effect, under random scheduling, of a question
  having entered training by the cutoff instead of a random other question
  taking its place; not pure self-influence and not a share of the total gain.
  A wide interval is called uninformative and left there.
- Discovery pair vs new pairs: for each primary cell, whether the seed-42 value
  lies inside or outside the [min, max] of the new pairs (descriptive only).
- All deviations and unresolved alerts are listed in RESULTS.md.

## F. Analyses of the existing seed-42 data (post-outcome, descriptive)

- **F1 Conditional sampling intervals.** For d_b and q_b on C, R, T at all 20
  snapshots: 3,000 draws; each draw resamples, independently for each arm and
  each question, the question's 16 responses with replacement (questions and
  bin membership fixed; both directions computed on the same draw); d and q
  recomputed (panel gap from the same draw); pointwise 2.5–97.5 % percentiles.
  RNG `numpy.random.default_rng(20261002)`. These are sampling uncertainty
  conditional on the two observed trajectories, not training-seed variability.
  The same method gives the within-pair intervals for new pairs (K as used).
- **F2 R/C/T alignment.** d_b and q_b for R, C, T, five bins, at EVAL_STEPS.
- **F3 Advantage mass.** Mass ratio and mass-share ratio (definitions §7) per
  bin at EVAL_STEPS.
- **F4 Batch structure.** Per arm, share of optimizer steps (= generation
  steps) with 0, 1, 2 live groups (live: not all 16 advantages zero); and the
  same restricted to steps containing a panel question, per bin (for each
  direction, steps whose panel question falls in the bin; directions averaged).
  Reading rule: with identical weights, rollouts and loss handling, a step with
  one live group gives loss gradients under the two estimators that differ only
  by a scalar — a statement about the estimators, not about the realized AdamW
  updates of the two runs. Shares only; no conclusions about realized updates.
- **F5 Bin contents.** Using the opposite bank (direction's baseline half):
  mean baseline C and T per bin, and the share of questions with baseline
  C ≥ 0.5; directions averaged. One table; no other stratification.

## 9. Fact-check

Before anything is called final, a separate agent with a fresh context
recomputes every number in RESULTS.md from the raw ledgers and evaluation
outputs with its own script (not importing `camera_ready/analysis`) and checks
the wording against §8. Output `camera_ready/FACTCHECK.md`; every discrepancy is
fixed and the check re-run.
