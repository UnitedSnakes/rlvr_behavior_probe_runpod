# Independent fact-check brief (for a fresh-context agent)

You are checking `camera_ready/RESULTS.md` for the camera-ready replication of
"Own Exposure, Advantage Reweighting, and Behavioral Improvement in RLVR".
Work only from raw files and the rule documents. **Do not import or execute
anything under `camera_ready/analysis/` or `analyses/`.** Write your own script
(e.g. `camera_ready/factcheck/recompute.py`) using only numpy/json/csv.

## Rule documents (read first)

- `camera_ready/PREREG_RUNS.md` — estimands (§7), statistics and wording (§8), validity (§6).
- `camera_ready/PREREG_BRIDGE.md` — bridge statistics and verdict rules.
- Paper definitions (for exact formulas): `~/Downloads/attrib_draft_v5_source/paper.tex`,
  §2–§4 and Appendices B–D.

## Where the raw data are (read this first)

The laptop is on a metered phone hotspot: **do not copy large files to the laptop.** All raw inputs
are on the GPU box (`ssh autodl-4080-4`, use `-o ConnectTimeout=20`; the gateway is sometimes flaky,
retry). Run your recompute script ON THE BOX with `/root/autodl-tmp/envs/rlvr/bin/python` (numpy
available), from a directory of your own such as `/root/autodl-tmp/factcheck/` (copy your script there
with scp; it is small). Pull back only your small JSON/Markdown outputs. Do not touch anything under
`/root/autodl-tmp/runs`, `/root/autodl-tmp/queue` or the GPUs. Box layout (`/root/autodl-tmp/analysis_data/`):

- `banks/a40_original/rollouts_shard{0,1}of2.jsonl` — A40 π0 banks (frozen bins; A40 baselines).
- `bridge_box/pi0_bank/rollouts_shard{0,1}of2.jsonl` — π0 banks regenerated on the box.
- `bridge_box/{grpo,maxrl}_seed42/pi_XXX/snapshot_raw.jsonl` — seed-42 checkpoints re-evaluated on the box.
- `seed42_a40/{grpo,maxrl}/{eval/pi_XXX,ledger}` — discovery pair (A40).
- `seed4{3,4,5}/{grpo,maxrl}/{eval/pi_XXX, eval_extra/..., ledger, step_log.jsonl, *_run_manifest.json, camera_ready_integrity.json}` — box pairs.
- Prompt token counts (frozen covariate): `camera_ready/analysis/prompt_token_counts.json` in the repo (read it as data; do not import the module).

## Raw inputs (format)

- Frozen bins and A40 baselines: `p0_train_k32_top_p1_canonical/rollouts_shard{0,1}of2.jsonl`
  (repo root of the main checkout; each record has `rollouts_A`, `rollouts_B` with
  per-response `canonical_reward`, `terminated`, `correct`, `n_tokens`). Bin of question i in
  direction A = bin of the mean `canonical_reward` of `rollouts_A`; direction B uses `rollouts_B`.
  Bins: 0, (0,.25], (.25,.5], (.5,.75], (.75,1), 1 (bin 1 excluded from bin comparisons).
- Box-regenerated π0 bank: `camera_ready/data/bridge_box/pi0_bank/`.
- Per pair `camera_ready/data/<pair>/{grpo,maxrl}/`: `eval/pi_{025,045,065,100}/snapshot_raw.jsonl`
  (per question: `rollouts` with `canonical_reward`, `terminated`, `correct`; `question_seed`),
  optional `eval_extra/pi_100_b{1,2,3}/`, `ledger/*.jsonl` (rows with `generation_global_step`,
  `dataset_index`, `advantage`, `canonical_reward`), `step_log.jsonl` (per optimizer step
  `grad_norm`, `learning_rate`).
- Discovery pair: `camera_ready/data/seed42_a40/` (same layout, A40 data).
- Bridge box outputs: `camera_ready/data/bridge_box/{grpo,maxrl}_seed42/pi_XXX/`.

## Recompute independently

1. Every number in every RESULTS.md table: panel rates; d_b and q_b (C, R, T; five bins;
   steps 934/1681/2428/3736); the primary cells and their across-pair mean, two-sided 95 %
   t-interval, one-sided 95 % upper bound; the K rule (64 only if every pair has 64 endpoint
   responses); mass ratios and mass-share ratios; pre-exposure adjusted gains (frozen OLS:
   ΔC ~ 1 + E + z(baseline reward rate, baseline mean completion length, prompt token count),
   per cutoff × bin × direction, constants dropped, cells with < 2 exposed or < 2 unexposed
   skipped, directions averaged); random-schedule contrast intervals; clipping shares;
   bridge cells, pooled Δ, interaction, gates, verdict.
2. Check the seed-42 reproduction: paper Table 1, 2, 4, 5 and Appendix A numbers
   (listed in `camera_ready/analysis/reproduce_seed42.py` as constants only — read them,
   do not run the module).
3. Integrity facts claimed for each run (3,736 steps, 119,552 rows, 7,472 groups, audit).

## Check the wording against PREREG_RUNS §8

Past failure modes in this project — look for each explicitly:
- "not established" written as "rejected" or "no effect"; "equivalent"; "not predictive";
  "per unit of advantage mass"; any causal decomposition of a gain.
- A registered statistic described as run when it was not (e.g. an interval claimed with n < 3).
- Traces or numbers from different seeds read as one trajectory.
- A summary range presented as the run-level range.
- Within-pair sampling intervals presented as seed uncertainty.
- A wording label that does not follow mechanically from the interval and δ = 3 pp.

## Output

`camera_ready/FACTCHECK.md`: for every table/number in RESULTS.md, MATCH or MISMATCH (your
value, their value, tolerance 0.01 pp for pp quantities, 1e-3 relative otherwise), and every
wording issue with the sentence quoted. End with a one-line verdict: PASS (no mismatches, no
wording issues) or FAIL.
