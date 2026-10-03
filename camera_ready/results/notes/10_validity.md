## Validity of the three box pairs

All six runs trained to completion on the first attempt (no OOM, no restart), from the same checkout (`9814757`) and environment, both arms of each pair concurrently. All three pairs are valid under PREREG_RUNS §6.

| Seed | GPUs (GRPO / MaxRL) | Training h (GRPO / MaxRL) | Rows / groups / steps | Max advantage error (GRPO / MaxRL) | Token IS ESS/N (GRPO / MaxRL) | Grad norm logged |
|---|---|---|---|---|---|---|
| 43 | 0,1 / 2,3 | 17.08 / 17.21 | 119,552 / 7,472 / 3,736 each | 2.24e-7 / 1.59e-7 | 0.99808 / 0.99808 | 3,736 / 3,736 |
| 44 | 2,3 / 0,1 | 17.10 / 17.25 | 119,552 / 7,472 / 3,736 each | 2.24e-7 / 1.59e-7 | 0.99807 / 0.99808 | 3,736 / 3,736 |
| 45 | 0,1 / 2,3 | 17.11 / 17.21 | 119,552 / 7,472 / 3,736 each | 2.24e-7 / 1.59e-7 | 0.99809 / 0.99807 | 3,736 / 3,736 |

No non-finite ledger fields in any run. Every panel question was sampled in every run (no never-sampled panel question). K = 16 protocol evaluations exist for all four EVAL_STEPS of every run (24 evaluations, 35.0–38.3 min each by queue wall clock, including model loading). Extra endpoint batches (16 responses each, both arms): batches 1–3 for seeds 43 and 44, batches 1–2 for seed 45; seed 45's batch 3 was not started under the pre-registered cut-off (see §5 item 16). They are not used for the primary cells.

Whole-panel rates at step 3736 (%, K = 16; R / T / C): seed 43 GRPO 51.78 / 75.22 / 56.76, MaxRL 51.54 / 74.46 / 56.98; seed 44 GRPO 51.17 / 74.41 / 56.25, MaxRL 51.05 / 74.68 / 56.47; seed 45 GRPO 51.54 / 74.58 / 56.45, MaxRL 52.71 / 74.90 / 57.81; seed 42 (A40) GRPO 53.03 / 77.25 / 57.74, MaxRL 51.54 / 74.73 / 56.40.

Compatibility alerts: none. The smoke comparison against the A40 seed-42 ledger (first 40 steps) raised no > 3 SE difference; the bridge passed.
