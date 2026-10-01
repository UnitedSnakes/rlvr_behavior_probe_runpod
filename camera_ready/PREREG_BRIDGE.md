# Pre-registration: hardware bridge (evaluation compatibility), RTX 4080 SUPER vs A40

Status: frozen before any evaluation output exists on the AutoDL box.
Written 2026-10-01 ~05:20 UTC. The commit and push time of this file is the
proof of ordering; its SHA-256 is recorded in `PROGRESS.md`.

Scope: this bridge compares generation and scoring of the **same frozen
weights** on the two machines. It cannot show that training on this box behaves
like training on A40. Nothing below claims more than evaluation compatibility.

## C1–C4 (verbatim from the camera-ready brief)

> C1. **Bridge set** — frozen checkpoints produced on A40: π0, and the seed-42 GRPO and MaxRL checkpoints at `EVAL_STEPS` (9 checkpoints). Re-evaluate each on this box with the unchanged evaluator: same panel, same per-question seeds, same K as the stored A40 evaluation (π0: both 16-response banks; snapshots: 16), same decoding settings, same 2,048-token cap. Identical seeds won't give identical tokens across GPUs; the comparison is statistical.
>
> π0 comes first and is mandatory before any training. The seed-42 snapshots don't block training: if they aren't on the box by `LAUNCH_TARGET`, finish the π0 part, go on to D and E, and complete the bridge whenever GPUs are free (after training at the latest). Fallback if the seed-42 snapshots can't be obtained at all: any RL-trained checkpoints with stored A40 evaluations that are quick to fetch (the bridge report lists the 2145 H=100 checkpoints, evaluated at K=64 on A40), with the arm interaction computed between whichever two arms they come from.
>
> C2. **Statistics.** For each checkpoint and each metric X ∈ {R, T, C}: Δ = panel mean over questions of (rate on this box − rate on A40), with SE from the across-question SD of the per-question differences. Pooled Δ over checkpoints, equal weights, with a question-level bootstrap 95 % interval (resample questions, 5,000 draws). Arm interaction: Δ(MaxRL checkpoint) − Δ(GRPO checkpoint) at the same step, pooled over the steps, same bootstrap. Also report, without gating on them: mean completion length, cap-hit rate, and per-bin Δ on the paper's frozen bins.
>
> C3. **Verdict.**
>
> - **PASS** — all four hold: (a) for each of R, T and C, the 95 % interval of the pooled Δ lies inside ±1.5 pp; (b) no checkpoint × metric cell has |Δ/SE| > 3.5, and at most one checkpoint has any cell with |Δ/SE| > 2.5; (c) for C and for R, the pooled arm-interaction point estimate is within ±1.5 pp AND its 95 % interval lies inside ±3.0 pp; (d) the set contains at least four RL-trained checkpoints from two arms.
> - **GROSS FAIL** — |Δ| above 5 pp for π0 or pooled, or malformed outputs. That is a stack or config problem, not a hardware property. No training on a stack in this state: diagnose, fix the environment, re-run; if you can't, stop and tell me.
> - **NOT PASSED** — anything else, including "only π0 could be bridged" and "bridge unfinished". Spend at most 45 minutes looking for a stack or config cause (package versions, sampling parameters, tokenizer / chat template, dtype, attention implementation). If you find and fix one before any training has started, re-run the bridge once under the same pre-registration. Otherwise go on and state that the cause is unidentified.
>
> C4. **Consequence.** The verdict changes only these two things; the training queue and the main sample are the same either way.
>
> - **Baselines.** PASS: baselines and bins are the paper's stored A40 banks, as in the paper. NOT PASSED: baselines come from the π0 banks regenerated on this box in C1 (bins selected by the original bank A use the regenerated bank B as baseline, and vice versa); bin membership stays the paper's frozen membership either way.
> - **Supplementary mixed estimate.** PASS: additionally report an estimate that pools the original A40 pair with the box pairs, at K=16, labelled as mixing the discovery run with replications across hardware. NOT PASSED: no pooled estimate; the original pair is shown separately, plus one sensitivity row using its checkpoints as re-evaluated on this box if C1 produced them.

## Exact bridge set and code

Panel: GSM8K train indices 0–255, revision `740312add88f781978c0658806c59bc2815b9866`.
Decoding: temperature 0.8, top-p 1.0, top-k 0, repetition penalty 1.0, max
completion 2,048, max model length 2,560, bf16, tensor parallel 1, one question
per request (`n` responses per request). Config:
`controlled_run/configs/grpo_qwen3_0_6b.yaml` unchanged (seed 42).

| # | Checkpoint | A40 reference file(s) | Box command (unchanged code) | Per-question seeds | K |
|---|---|---|---|---|---|
| 0 | π0 (`pi0_lineage_id f89fc902…`, HF `HKReporter/rlvr-behavior-probe-pi0-corrected-canonical-2026-08-30` `pi_0/`) | `p0_train_k32_top_p1_canonical/rollouts_shard{0,1}of2.jsonl` (sha256 `9a183634…`, `3f005b85…`) | `python diagnose_p0_signal_budget.py --policy <pi0> --shard s --num-shards 2` @`5b10742` (file identical at `1c26b1f`), s = 0, 1 | A: `42·100000 + i`; B: `42·100000 + i + 50000` | 16 + 16 |
| 1–4 | GRPO seed 42 `pi_025`, `pi_045`, `pi_065`, `pi_100` (steps 934, 1681, 2428, 3736; HF `HKReporter/rlvr-behavior-probe-grpo-canonical-seed42-2026-09-02` @`0ff3639`) | `snapshot_eval_train256_k16_cbank/pi_{025,045,065,100}/snapshot_raw.jsonl` | `python -m controlled_run.eval_snapshot --canonical-run-dir <grpo run> --snapshot-pct P --panel train256 --output-dir <out>` @`1c26b1f` | `42·100000 + i + 75000` | 16 |
| 5–8 | MaxRL seed 42 `pi_025`, `pi_045`, `pi_065`, `pi_100` (HF `HKReporter/rlvr-behavior-probe-maxrl-canonical-seed42-2026-09-05` @`e67069c`) | `maxrl_snapshot_eval_train256_k16_cbank/pi_{025,045,065,100}/snapshot_raw.jsonl` (HF dataset `HKReporter/rlvr-behavior-probe-maxrl-analysis-seed42-2026-09-05` @`88b7f32`) | same as 1–4 | same | 16 |

Resource-only settings that may differ from A40 and are logged: vLLM
`gpu_memory_utilization`, GPU assignment (one evaluation per GPU), output paths.
Downloaded weights are verified against the A40 `SHA256SUMS.txt` / HF LFS
SHA-256 before use. Before any box evaluation output is read for statistics,
each output must pass the existing loaders: 256 unique questions, the expected
per-question seed, K responses each, counts consistent with per-response flags,
R = T·C for every response. Failure of these checks is "malformed outputs".

## Formulas

Notation: question i ∈ {0,…,255}; checkpoint c; metric X ∈ {R, T, C};
x_{c,i}^{box}, x_{c,i}^{A40} = per-question rate (fraction of the K responses).
For π0 the per-question rate pools both 16-response banks (32 responses);
the two banks are also reported separately (not gating).

- Per-question difference: δ_{c,i}^X = x_{c,i}^{box} − x_{c,i}^{A40}.
- Checkpoint Δ: Δ_c^X = (1/256) Σ_i δ_{c,i}^X; SE_c^X = sd_i(δ_{c,i}^X)/√256 (sample SD, ddof = 1).
- Pooled Δ: Δ̄^X = (1/|S|) Σ_{c∈S} Δ_c^X, S = all bridged checkpoints (π0 included), equal weights.
- Arm interaction at step s: I_s^X = Δ_{MaxRL,s}^X − Δ_{GRPO,s}^X; pooled Ī^X = mean over the steps s for which both arms were bridged.
- Bootstrap: 5,000 draws; each draw resamples 256 question indices with replacement, the **same** index multiset applied to every checkpoint and metric; recompute Δ̄^X and Ī^X; 95 % interval = 2.5th–97.5th percentiles. RNG: `numpy.random.default_rng(20261001)`.
- Gate (a): for each X, the bootstrap interval of Δ̄^X ⊂ (−1.5 pp, +1.5 pp).
- Gate (b): over the |S| × 3 cells, max |Δ_c^X / SE_c^X| ≤ 3.5, and the number of checkpoints with any cell |Δ/SE| > 2.5 is ≤ 1. (A cell with SE = 0 and Δ = 0 counts as 0.)
- Gate (c): for X ∈ {C, R}: |Ī^X| ≤ 1.5 pp and its bootstrap interval ⊂ (−3.0 pp, +3.0 pp).
- Gate (d): S contains ≥ 4 RL-trained checkpoints spanning both arms.
- GROSS FAIL: |Δ_{π0}^X| > 5 pp or |Δ̄^X| > 5 pp for any X, or malformed outputs.
- Reported without gating: per-checkpoint mean completion length and cap-hit
  rate (share of responses with `terminated = false`) on both machines; per-bin
  Δ on the frozen cross-fit bins (bins from the original A40 banks, symmetric
  average of the two directions).

Interval boundaries are open: an endpoint exactly at ±1.5 pp (or ±3.0 pp)
fails. The verdict is computed by `camera_ready/analysis/bridge.py`, which is
committed before box outputs are analysed.

## Order and timing

1. π0 banks (two GPUs, one shard each) before any training. The GROSS FAIL
   test on π0 is applied immediately; training starts only if π0 is not a GROSS FAIL.
2. The eight seed-42 snapshots run when GPUs are free: before training if they
   are on the box in time, otherwise between pairs or after all training.
3. The verdict uses whatever set S exists when it is computed. Until all nine
   are done the status is "NOT PASSED (bridge unfinished)" for C4 purposes; the
   verdict is final once the nine have run or the GPUs are released.
4. The C1 fallback (project_2145 checkpoints) is used only if the seed-42
   snapshots cannot be obtained at all; those rows would live in a git-ignored
   file (`camera_ready/private_fallback/`).
