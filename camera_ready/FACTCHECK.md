# Independent fact-check: RESULTS.md and BRIDGE_REPORT.md

Checked 2026-10-03 (~17:00 UTC) by a fresh-context agent. Documents checked: `camera_ready/RESULTS.md` (as of 12:40 local, 37,260 bytes) and `camera_ready/BRIDGE_REPORT.md` (4,589 bytes). Rules: `PREREG_RUNS.md` §6–§8, `PREREG_BRIDGE.md`, and the paper source `attrib_draft_v5_source/paper.tex` (§2–§4, App. A–D).

**Bottom line.** Every scientific number checked matches: the primary cells and their wording labels, intervals, bridge statistics and gates, advantage mass, pre-exposure estimates, panel rates and secondary tables. Of the 2,157 parsed numbers and flags, 2,156 match. The result is still **FAIL**: one parsed number and two prose numbers are wrong as labelled (§2), and there is one wording issue plus one internal inconsistency (§4). None of these touches a primary cell, a pre-registered label or the bridge verdict.

## 1. Method

- **Script.** `camera_ready/factcheck/recompute.py`, written from scratch from the rule documents and the paper's definitions. It uses only numpy, json, os, math and re, and it imports and copies nothing from `camera_ready/analysis/` or `analyses/`. The only file it reads from `camera_ready/analysis/` is `prompt_token_counts.json`, and only as data. It has three modes:
  - `box` does all raw-data recomputation.
  - `laptop20` computes the F1 counts over all 20 seed-42 snapshots.
  - `compare` parses every table in RESULTS.md and BRIDGE_REPORT.md and compares each number with the recomputed value.
- **Box run.**
  - **Command:** the script was copied with scp to `/root/autodl-tmp/factcheck/` and run as `nohup nice -n 10 /root/autodl-tmp/envs/rlvr/bin/python recompute.py box /root/autodl-tmp/analysis_data /root/autodl-tmp/factcheck/prompt_token_counts.json /root/autodl-tmp/factcheck/fc_box.json`.
  - **Inputs, all read-only from `/root/autodl-tmp/analysis_data`:**
    - the A40 π0 banks and the box π0 banks;
    - `seed42_a40`, `seed43`, `seed44` and `seed45`, each with the 4 K=16 evaluations per arm, both ledger rank files, `step_log.jsonl`, `camera_ready_launch.json` and the training-log mtimes;
    - the bridge re-evaluations in `bridge_box/{grpo,maxrl}_seed42`;
    - for the seed-42 MaxRL clipping line only, `/root/autodl-tmp/a40/maxrl_trainer_meta/trainer_state.json`.
  - **Not touched:** `runs/`, `queue/`, `analysis_out/`, the GPUs, the screen sessions and the HF processes.
  - **Output:** only the gzipped JSON output (39 KB) was pulled back. Copies are in `camera_ready/factcheck/fc_box.json` and on the box.
- **Laptop run.** The F1 "20 snapshots" counts need the 16 non-EVAL seed-42 snapshots, which exist only on the laptop (`camera_ready/data/seed42_a40/*/eval/pi_005…pi_100`, local files). That part ran locally on local files, with no network transfer: `recompute.py laptop20 data/seed42_a40 data/banks/a40_original …`. Output: `camera_ready/factcheck/fc_laptop20.json`.
- **Comparison.** `recompute.py compare fc_box.json fc_laptop20.json RESULTS.md BRIDGE_REPORT.md compare.json` checked 2,157 numbers and flags. Prose numbers that are not in tables (§0, Validity, §5, §6, BRIDGE_REPORT notes) were checked by hand against the same outputs; see §3.
- **Tolerances.**
  - pp quantities: |difference| ≤ 0.01 pp.
  - Ratios and fractions: the displayed value must equal my value rounded to the displayed precision, or be within 1e-3 relative.
  - Bootstrap endpoints: the plan was max(0.15 pp, 3.5 Monte Carlo SE). This turned out not to be needed. All 288 conditional-sampling endpoints and all 12 bridge bootstrap endpoints agree to display precision (max |difference| 0.005 pp, i.e. rounding); spot checks at full precision agree to about 1e-13. My resampling draw order happens to coincide with the pipeline's:
    - F1: `default_rng(20261002)`, chunks of 250 draws, MaxRL indices then GRPO indices per chunk.
    - Bridge: `default_rng(20261001)`, a (5000, 256) index matrix.

    So the intervals are reproduced exactly, not merely within MC error.
- **Definitions used.**
  - Bins: from the mean `canonical_reward` of each A40 half.
  - d_b and q_b: symmetric averages of within-bin means of per-question rate differences; q subtracts the whole-panel difference.
  - Advantage mass: S_t(b) counts groups with generation step < t, averaged over the two directions; the ratio is S^M/S^G; Tot_t sums all groups with step < t.
  - Clipping: c_g = min(1, 1/(n_g + 1e-6)). A group from generation step s is weighted by c at optimizer step s+1.
  - Pre-exposure: OLS of ΔC (A40 opposite-bank baseline) on 1 + E + covariates. The covariates are opposite-bank reward rate, opposite-bank mean `n_tokens` and prompt tokens, standardized within the cell (SD ddof = 1, constant columns dropped). Cells with fewer than 2 exposed or 2 unexposed questions are skipped. U = α and E = α + τ, averaged over directions.
  - Token IS ESS/N: (Σ actual_is_ratio_sum)² / (Σ actual_is_ratio_sq_sum · Σ actual_is_ratio_count) over all ledger rows.
  - t quantiles: hard-coded (t.975 with df 2 = 4.3027, df 3 = 3.1824; t.95 with df 2 = 2.9200).
- **Implementation check against the submitted paper (seed 42).** All of the following reproduce exactly, to displayed precision:
  - bin counts A 33/89/57/56/20/1 and B 37/86/59/55/19/0;
  - App. B zero-bin rates 3.41 / 11.93 / 5.07 / 10.98 %;
  - Table 1 (π0 34.91/45.62/50.45; GRPO 53.03/77.25/57.74 with Δ +18.12/+31.63/+7.29; MaxRL 51.54/74.73/56.40 with Δ +16.63/+29.11/+5.94);
  - Table 2 (mass ratios 2.205/1.720/0.988/0.592/0.463; ΔC_G, ΔC_M, d_b, q_b; panel gap −1.343);
  - all 15 cells of the App. C exposure grid, including E/U counts, and the direction-specific low-bin U−E (5.55, 7.85, 4.27 and 2.00, 2.91, −0.46);
  - the App. A MaxRL audit (max advantage error 1.59e-7, ESS/N 0.99805).

## 2. Number-by-number results

| RESULTS.md / BRIDGE_REPORT.md item | Checked | Result |
|---|---:|---|
| Validity table (GPUs, training h, rows/groups/steps, max advantage error, ESS/N, grad-norm steps), 3 seeds × 2 arms | 33 | MATCH. Training hours were checked against file mtimes (launch JSON to end of training log): 17.07/17.21, 17.09/17.24, 17.11/17.20 h. That is within 0.02 h of the stated queue exit times. |
| Prose whole-panel rates at step 3736 (8 runs × R/T/C) | 24 | MATCH |
| Primary cells: per-pair values, within-pair intervals, mean, 95 % CI, one-sided upper bound, wording, discovery value and position | 128 | MATCH. All 8 labels follow mechanically. Every CI contains 0 and reaches beyond ±3 pp. The tightest case is d, C, bin 0: [−0.773, +3.077]. |
| Pre-registered sentences (8) | 48 | MATCH |
| Advantage mass ratio and mass-share ratio (4 pairs × 4 steps × 5 bins) | 160 | MATCH |
| Reweighting realized at 3736, and at all four steps | 2 | MATCH: True and True. The minimum ratio in bins 0 and (0,.25] over all pairs and steps is 1.582. |
| Pre-exposure table (E, U, U−E, counts) and the "replicated" flag | 49 | MATCH. Replicated = True; the minimum ΔC_U is 5.86 pp. |
| Random-schedule U−E (per seed, mean, CI) | 18 | MATCH |
| Supplementary mixed estimate (values, n = 4, mean, CI) | 64 | MATCH |
| Secondary d_b/q_b tables for C, R, T (mean, CI, three per-pair values) | 720 | MATCH |
| Whole-panel rates table | 96 | MATCH |
| Clipping table (steps logged, share with c_g < 1, mean, median, 5th percentile) | 30 | MATCH. The "Share clipped" column uses the pre-registered c_g < 1. |
| Clip-weighted mass ratios | 60 | MATCH |
| Discovery sensitivity row (A40 vs box re-evaluation, d/q for C) | 80 | MATCH |
| F1/F2 seed-42 d/q for C, R, T, with conditional intervals | 360 | MATCH |
| F1 counts across 20 snapshots | 10 | MATCH. The nearest interval endpoint to 0 is 0.086 pp away. |
| F3 mass, share ratios and total \|A\| | 44 | MATCH |
| F4 batch structure (step-level definition) | 36 | MATCH |
| F5 bin contents | 15 | MATCH |
| Seed-42 MaxRL clipping line | 6 | **5 MATCH, 1 MISMATCH** (see below) |
| BRIDGE_REPORT: per-checkpoint Δ/SE/max\|Δ/SE\|, pooled Δ and CI, interaction and CI, length and cap-hit, per-bin ΔC, per-bank π0 Δ, gates, verdict | 174 | MATCH. The gates I recompute are (a) True, (b) True (max \|Δ/SE\| 1.58, no checkpoint above 2.5), (c) True, (d) True, GROSS FAIL False, giving **PASS**. |

**Mismatches**

1. **RESULTS §4, seed-42 clipping line: "share clipped (grad norm > 1) 78.0 %".**
   - **What the data show:** of the 373 logged steps, 289 have a logged grad norm strictly above 1, which is **77.5 %**. The stated 78.0 % (291/373) is the pre-registered c_g < 1 share. That share also counts the 2 steps whose logged grad norm is exactly 1.0.
   - **Fix:** either label the value "c_g < 1 (logged norm ≥ 1.0)" or report 77.5 %.
2. **RESULTS §6 item 7: "77.5–83.8 % of optimizer steps had pre-clip gradient norm above 1.0 in every new run (GRPO 83.5–83.8 %, MaxRL 77.5–77.8 %)".**
   - **What the data show:** these are the c_g < 1 shares, which include 6–14 steps per run with a logged norm of exactly 1.0. Strictly above 1.0, the shares are **77.3–83.6 %** overall: GRPO 83.30–83.65 % and MaxRL 77.28–77.49 %. The per-run values are 83.65/77.49 (seed 43), 83.32/77.49 (seed 44) and 83.30/77.28 (seed 45).
   - **Fix:** change "above 1.0" to "c_g < 1 (logged norm ≥ 1.0)", or use the strict numbers.
3. **RESULTS Validity: "24 evaluations, 34–36 min each".**
   - **What the data show:** I measured each evaluation from the mtime of the provenance JSON written at start to the last write of `snapshot_raw.jsonl`. On that measure the 24 durations are **34.7–37.9 min**. The six step-934 evaluations (`pi_025`) took **37.2–37.9 min**; the other 18 took 34.7–35.9 min.
   - **Direction of any bias:** the PROGRESS figures for the endpoint evaluations (for example 35.0 and 35.3 min for seed 43) are 0.1–0.2 min longer than my measure. So the true durations are, if anything, slightly longer than mine.
   - **Fix:** write "35–38 min each".

## 3. Prose checked by hand

| Claim | Result |
|---|---|
| §0, all eight primary cells: every interval contains 0 and reaches beyond ±3 pp; none lies entirely above or below 0 | MATCH |
| §0, reweighting realized at 3736 and at all four steps | MATCH |
| §0, pre-exposure gain replicated | MATCH |
| §0, U−E per-seed values differ in sign | MATCH: seed 43 is positive at all cutoffs; seeds 44 and 45 are negative at all cutoffs |
| §0, bridge PASS over 9 checkpoints | MATCH |
| §0, mixed estimate n = 4 | MATCH |
| SHA-256 prefixes of `analysis.json` (2f1fa340…4419) and `bridge.json` (e816c7f0…b082) | MATCH |
| Commit `45d6130` exists | MATCH |
| Validity, training checkout `9814757` | MATCH (all six `camera_ready_launch.json`) |
| Validity, evaluation commit `1c26b1f` | MATCH (all 24 provenance files) |
| Validity, first attempt only | Consistent: only `*_attempt1.log` exists |
| Validity, no non-finite ledger fields | MATCH (0 in every run) |
| Validity, every panel question sampled | MATCH: 256/256 panel questions, each exactly once, and 7,472 distinct training questions per run |
| Validity, panel schedule identical across the arms of each pair | MATCH |
| Validity, K = 16 evaluations at all four steps for all 6 runs | MATCH: 256 questions × 16 responses each, per-question seed S·100000+i+75000, R = T·C per response |
| §5 item 2, evaluation seed rule | MATCH (asserted on every record) |
| §5 item 5, peak allocated memory 29.9 GiB | Consistent: the maximum logged `cr_max_memory_allocated_gib` is 29.89 GiB in all six runs. The 31.5 GiB capacity was not checked. |
| §5 item 6, vLLM 0.30 | MATCH (all six configs) |
| PREREG_BRIDGE commit `556a271` time | MATCH: committed 2026-10-01 05:13:35 UTC, before the claimed 05:13:41 push |
| PREREG_RUNS freeze commit `293891a` time | MATCH: 06:22:31 UTC |
| §6 item 2, per-seed ΔC_U ranges 10.0–12.5 / 5.9–6.7 / 6.2–8.6 pp | MATCH (10.007–12.500, 5.858–6.650, 6.215–8.588) |
| §6 item 3, mass ratios 1.58–2.78 at 3736, bins 0 and (0,.25] | MATCH (1.5816–2.7817) |
| §6 item 4, seed-42 value below the new pairs' range in 6 of 8 cells; −4.71 vs +0.48 to +2.00 | MATCH |
| §6 item 5, mixed q, C, (0,.25] interval [+0.13, +2.16] | MATCH ([+0.129, +2.155]) |
| §6 item 7, clip-weighted 1.11–1.39 vs unweighted 1.58–2.78 | MATCH (1.108–1.386); the shares in the same item are **MISMATCH** (see §2) |
| BRIDGE_REPORT, π0 per-bin regression toward the mean | Consistent: the reported per-bin π0 Δ reproduces only with the pooled 32-response A40 rate, which contains the bin-selecting bank |

**Not verifiable within the access constraints.** These are not counted as mismatches.

- **Extra endpoint batch status** (RESULTS lines 5, 30 and §5 item 12b). The extra batches are not under `analysis_data`, and `runs/` and `queue/` were off-limits. PROGRESS.md has no entry recording them. The K = 16 decision itself is consistent with the data: every pair has exactly 16 protocol responses per question.
- **"Smoke comparison … raised no > 3 SE difference"** (Validity). The smoke outputs are scratch; only PROGRESS.md (max |z| 1.43) supports it.
- **"382 checks … gave the same numbers"** (§5 item 12a). This would need the pipeline module, which I was not allowed to run.

## 4. Wording check (PREREG_RUNS §8)

**Passed.**

- **Banned terms.** None of "equivalent", "no effect", "not predictive", "per unit of advantage mass", "rejected" or a causal decomposition appears in a scientific statement. "no effect" occurs only in the heading "Operational issues (no effect on data)", which is about operations, not about an estimand.
- **Labels.** The 8 labels follow mechanically from the two-sided intervals with δ = 3 pp. The one-sided upper bounds below 3 pp (6 cells) are correctly not turned into "no additional gain larger than δ".
- **Statistics with n < 3.** None is presented. The mixed estimate (n = 4) is labelled supplementary and hardware-mixing, carries no wording rule, and is shown only under PASS.
- **Within-pair intervals.** They are labelled "not seed uncertainty" (§2) and "not training-seed variability" (F).
- **Dependent snapshots.** The F1 counts are called dependent, not independent trials.
- **Trajectories.** No traces from different seeds are read as one trajectory. Figure 2 is correctly called single-seed.
- **Summary ranges.** These are labelled as ranges across pairs, bins or runs. The per-seed ΔC_U ranges in §6 item 2 are genuine within-run ranges across cutoffs.
- **Random-schedule contrast.** It is described with the pre-registered estimand text and called "uninformative".
- **Bridge.** Nothing beyond evaluation compatibility is claimed (RESULTS §0, §6 item 8, BRIDGE_REPORT scope).

**Issues.**

- **W1. RESULTS §6 item 3: "This can be stated as replicated across seeds."**
  - **Why it is an issue:** PREREG_RUNS §8 defines "replicated" only for the pre-exposure gain. For advantage mass the pre-registered criterion and term is "realized", which RESULTS §0 and §2 use correctly. Calling the reweighting "replicated" applies a term the rules reserve for a different estimand.
  - **Suggested wording:** "realized in every new pair at all four EVAL_STEPS (pre-registered criterion)".
- **W2. The clipping "grad norm > 1" / "above 1.0" labels** (§4 seed-42 line; §6 item 7). The number does not follow from the stated definition; see mismatches 1 and 2.
- **W3. Internal inconsistency on the extra batches.**
  - Line 5 says: "seed 45's third extra batch could not be scheduled before the pre-registered cut-off, so K = 64 was not available for all pairs". §5 item 12b says the same.
  - Line 30 says: "Extra endpoint batches: complete for seed 43, partial for seeds 44–45 at the pre-registered cut-off".
  - **Why it matters:** if seed 44 was also partial, line 5 gives an incomplete reason. If seed 44 was complete, line 30 is wrong.
  - **Timing problem:** the analysis started at 16:34 UTC. That is before the extra-batch cut-off (RESULTS_DUE − 6.5 h = 17:30 UTC). So "at the pre-registered cut-off" cannot describe the state the analysis saw.
  - **Fix:** state what existed and when, from the queue records. I could not check this.

**Advisory, not counted.**

- **§6 item 6, "same software stack".** §5 items 8–9 record a different CUDA toolkit and flash-attn build flags, with "core versions identical". "Same core package versions" would be more exact.
- **§6 item 4, "should not present the seed-42 endpoint values as typical".** This is acceptable as descriptive advice. With 3 exchangeable new pairs, though, a fourth value falls outside their range with probability 1/2, and the 8 cells are strongly dependent. The "6 of 8 below" count is weak evidence by itself.
- **Discovery sensitivity row.** Re-evaluating the same seed-42 checkpoints on the box moves d, C, bin 0 at 3736 from −4.71 to −0.16 pp. That is consistent with the ±3 pp within-pair sampling intervals. A sentence saying so would help readers of that table.
- **Numbering.** §5 numbers items 12a–12c after item 13.

## 5. Verdict

**FAIL.** The problems are:

- Mismatch 1: seed-42 clipping share labelled "grad norm > 1" (77.5 %, not 78.0 %).
- Mismatch 2: §6 item 7 shares labelled "above 1.0" (77.3–83.6 %, not 77.5–83.8 %).
- Mismatch 3: evaluation durations (34.7–37.9 min, not 34–36).
- W1: "replicated across seeds" used for reweighting.
- W3: the extra-batch status statements are inconsistent.

Every primary-cell number, wording label, bridge statistic, gate and verdict, mass, pre-exposure, panel and secondary number matches the independent recomputation.

## Re-check (2026-10-03 ~17:40 UTC)

**Scope.** This re-check covers the revised documents at `e455f7d` (fixes in `246579c`). Since the original check (`2f1e7df`), RESULTS.md changed only in the Validity paragraph, the seed-42 clipping line, §5 items 14–16 (renumbered, with new sub-headings) and §6 items 3, 4, 6 and 7. BRIDGE_REPORT.md, `results/analysis.json` and `results/bridge.json` are unchanged. My script was not edited.

**Methods.**

- **Comparison re-run.** I re-ran `recompute.py compare factcheck/fc_box.json factcheck/fc_laptop20.json RESULTS.md BRIDGE_REPORT.md …`. It checked 2,157 items, of which 2,156 match.
  - **The one flagged item no longer applies.** It is my own label-specific assertion "as labelled (grad norm > 1)". The label now reads "share clipped (clip coefficient c < 1, the pre-registered definition; it includes logged norms equal to one) 78.0 %". The c < 1 check on the same number matches: 291/373 = 78.02 %, and 2 logged norms are exactly 1.0.
- **Queue evidence.** I checked the new prose against `camera_ready/box/logs/queue.log` and `state.json` by parsing the files, and against the scheduling code in `camera_ready/box/queue_runner.py`.
- **Extra batches on the box.** I ran one small read-only check of the staged extra-batch outputs in `/root/autodl-tmp/results_staging`:
  - seeds 43 and 44 have `pi_100_b1–b3` for both arms; seed 45 has `pi_100_b1–b2` for both arms;
  - each batch has 256/256 questions × 16 responses;
  - every per-question seed equals S·100000 + i + 75000 + b·1,000,000.

  Only the printed summary was pulled back.

**Results for the original findings.**

| Original finding | Revised text | Result |
|---|---|---|
| Mismatch 1: seed-42 clipping share labelled "grad norm > 1" | Labelled as the pre-registered c < 1, including logged norms equal to 1; 78.0 % | **Resolved**: 78.02 %, with 2 steps at exactly 1.0 |
| Mismatch 2: §6 item 7 shares labelled "above 1.0" | c < 1: 77.5–83.8 % (GRPO 83.5–83.8, MaxRL 77.5–77.8); 6–14 steps per run exactly 1.0; strictly above 1.0: 77.3–83.6 % | **Resolved**: c < 1 values are 83.81/83.70/83.54 and 77.84/77.81/77.52; exactly-1.0 counts are 6/13/14/12/9/9; strict values are 83.65/83.32/83.30 and 77.49/77.49/77.28 |
| Mismatch 3: evaluation durations | "35.0–38.3 min each by queue wall clock, including model loading" | **Resolved**: the 24 protocol "eval done" lines in queue.log are all ok and cover all 3 seeds × 2 arms × 4 steps, with wall 35.0–38.3 min (step 934: 37.3–38.3; others 35.0–36.3). This is consistent with my file-time measure (34.7–37.9), since queue times are polled and include model loading. |
| W1: "replicated" used for reweighting | §6 item 3 now says "realized", with "replicated" reserved for the pre-exposure gain | **Resolved**. "replicated" now appears only for the pre-exposure gain (§0, §2, §6 item 2). |
| W3: inconsistent extra-batch status | Validity paragraph and §5 item 16 | **Resolved**, except for one number (see below). queue.log and state.json show 16 extra jobs done, all `ok=True`: batches 1–3 for both arms of seeds 43 and 44, and batches 1–2 of seed 45. At 17:18:35 the queue logged "no time for 2 remaining extra batches; skipped". The two skipped jobs are seed 45 b3 for both arms; this follows from the queue's job order and `extra_done`. The staged outputs on the box agree. §0 line 5, the Validity paragraph and §5 item 16 are now mutually consistent. K = 16 is correct under the all-or-nothing rule, because seed 45 has only 48 endpoint responses per question. |

**§5 item 16 timing statements, checked against queue.log.**

- Round starts and ends:
  - At 16:34 the running round (seed 44 b2/b3) ended at 16:42:34–16:43:15.
  - Seed 45 b1/b2 ran 16:43:15 to 17:18:35.
  - The skip came at 17:18:35.
- The cut-off: in the code, `latest_start` = RESULTS_DUE − (1.5 + 3.0 + 2.0) h = 17:30 UTC, checked per round. This matches the item.
- The training hours in the Validity table now equal the queue clock in `state.json` exactly: 17.0805/17.2140, 17.0970/17.2471, 17.1136/17.2136 h.

**New prose in §6, checked against my recomputation.**

| Claim | Result |
|---|---|
| Item 4, box re-evaluation, d for C in bin 0: −0.16 vs −4.71 pp | MATCH (−0.156 vs −4.712) |
| Item 4, box re-evaluation, q for C in bin 0: +0.84 vs −3.37 pp | MATCH (+0.845 vs −3.369) |
| Item 4, "descriptive only" caveat | Appropriate |
| Item 6, "same core package versions … (flash-attn build toolchain and PyPI mirror differ; §5 items 8–9)" | Consistent with §5 items 8–9 |
| §5 renumbering | Items 1–16 are now in order. The only remaining "12a"-like string is part of a SHA (`740312a…`). |

**Wording.** I scanned the revised text again for banned or overreaching terms ("equivalent", "no effect" outside the operations heading, "not predictive", "per unit of advantage mass", "rejected", causal decomposition). There are no new issues. The added sentences in §5 item 16 and §6 items 4 and 7 stay descriptive. §5 item 15 (HF storage limit at 16:44, resumed 16:50) is operational and documented in PROGRESS.md only; I did not check it.

**Remaining discrepancy.**

- **R1. §5 item 16: "the queue skipped seed 45's batch 3 (projected end ≈ 17:54 UTC)".**
  - **How the queue projects:** the committed queue code (`queue_runner.py`, extra-batch loop) projects a round's end as `now + max(eval_wall_s)`.
  - **What it projected here:** at 17:18:35 the largest recorded wall was 2,300.6 s (38.3 min, the seed-45 MaxRL step-934 evaluation at 14:17:43). The queue's projected end was therefore 17:56:56 UTC, about **17:57**, not 17:54.
  - **Effect:** none. Any of these estimates exceeds the 17:30 cut-off, so the skip and K = 16 are unaffected.
  - **Fix:** write "≈ 17:57 UTC".

**Outside the scope of this check (PROGRESS.md, not RESULTS.md).** The 17:18 entry says the 16 extra jobs took "35.0–35.7 min each"; queue.log shows 34.3–35.7 min. The same entry has the same "≈ 17:54" projection.

**Re-check verdict: FAIL.** One remaining discrepancy (R1: the projected end in §5 item 16 should be ≈ 17:57 UTC, not ≈ 17:54) has no effect on any decision or result. All five original findings are resolved, and every number, label, interval, gate and verdict in RESULTS.md and BRIDGE_REPORT.md matches the independent recomputation.

## Final confirmation (2026-10-03, after commit 9fc7ec7)

- **Scope of the change.** `git diff 5b08828 9fc7ec7 -- camera_ready/RESULTS.md` changes one sentence only: §5 item 16. The same sentence changed identically in `results/notes/50_deviations.md`. PROGRESS.md gains an appended 20:10 correction entry. BRIDGE_REPORT.md, `results/analysis.json`, `results/bridge.json` and `analysis/` are unchanged. The working tree is clean at `9fc7ec7`.
- **Comparison re-run.** `recompute.py compare` on the final documents checks 2,157 items, of which 2,156 match. The single flagged item is the obsolete "as labelled (grad norm > 1)" assertion, which the Re-check above already found no longer applies. The label states c < 1, and that value matches (78.02 %).
- **R1 resolved.** §5 item 16 now reads "the queue's projection, now + longest recorded evaluation of 38.3 min, gave an end of ≈ 17:57 UTC". This matches the code and the logs:
  - `queue_runner.py` projects `now() + max(eval_wall_s)` against `latest_start` = RESULTS_DUE − 6.5 h = 17:30 UTC.
  - `state.json` records the skip at 17:18:35.77 UTC, with a maximum of 2,300.56 s (38.3 min) over the 48 recorded walls.
  - The projected end is therefore 17:56:56 UTC, about 17:57, which is past 17:30. The skip and K = 16 stand.
- **Outside scope.** The PROGRESS.md 20:10 correction gives the right values (≈ 17:57; extra jobs 34.3–35.7 min).

**Final verdict: PASS.** No mismatches and no wording issues remain in RESULTS.md or BRIDGE_REPORT.md.
