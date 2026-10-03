Every item is logged with time and evidence in `PROGRESS.md`.

**Approved by Sam before launch**

1. *Seed check relaxed at runtime.* The seed-42 code hard-wires `seed == 42` in trainer and evaluator. `camera_ready/launch_train.py` and `camera_ready/eval_run.py` replace only that invariant (42 → S) at runtime; every other config, manifest, lineage and data check runs unchanged. The training and evaluation checkouts stay clean at `9814757` / `1c26b1f`.
2. *Evaluation seed base = training seed* (S·100000 + i + 75000), identical within a pair. Extra endpoint batches add b·1,000,000 (collision check: none).
3. *Evaluation order.* Endpoint K=16 evaluations right after each pair; the 934/1681/2428 evaluations after all training; extra endpoint batches last, skipped if time runs out.
4. *Time rule.* A pair launches only if its training and all launched pairs' evaluations, analysis and fact-check fit before RESULTS_DUE with ≥ 2 h buffer, and training ends before TRAIN_CUTOFF (both moved by Sam).

**Resource-only or network-only (no recipe change)**

5. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for training (first-step OOM from allocator fragmentation on 32 GB cards without it). Peak allocated memory 29.9 of 31.5 GiB.
6. Colocated vLLM fraction kept at the frozen 0.30 instead of the A40 absolute budget (0.45 would starve training on 32 GB; changing it would also need a second invariant relaxed).
7. π0 bank collector `gpu_memory_utilization` 0.95 instead of 0.70 (closest feasible to the A40 absolute KV budget).
8. flash-attn compiled with `MAX_JOBS=24 NVCC_THREADS=2` instead of 1/1; userspace CUDA 13.0.0 toolkit (nvcc 13.0.48) because the box has only system CUDA 12.9.
9. uv default index: aliyun PyPI mirror over HTTPS (PyPI via the proxy dropped connections). Core versions identical to the A40 record; full lock in `box/pip_freeze.txt`.
10. Training processes look up the GSM8K `main` revision through `https://hf-mirror.com` (proxy off during training); resolved SHA `740312a…` verified in every manifest.
11. Logging-only instrumentation (`INSTRUMENTATION.diff`): `logging_steps = 1`, step log JSONL with read-only CUDA memory counters, resolved Trainer args dump.

**Scope limits that are not deviations but must be stated**

12. The GRPO seed-42 execution commit is not recorded; inferred `c664e26`. Image and pod identity of the two seed-42 arms cannot be verified (`INVENTORY.md` §8).
13. Seed-42 grad norms exist only for MaxRL at every 10th step; GRPO seed-42 training logs were not kept.

**Decided by Sam during the run (10-03)**

14. *Data movement.* After seeds 43/44 were pulled in full, only small analysis outputs are pulled to the laptop (metered hotspot). The main analysis and the bridge statistics ran on the box with the committed code (commit `45d6130`); the discovery-pair results were computed on the laptop from the A40 data already there (`results/discovery_seed42.json`) and passed in. A box-side reproduction check (382 checks at the four EVAL_STEPS) gave the same numbers as the laptop.
15. *Backups.* Private HF dataset and checkpoint repos (Sam-approved), uploaded from the box only; HF-side SHA-256 verification (`PROGRESS.md`). The account's private storage limit (100 GB) blocked the seed-45 intermediate checkpoints at 16:44 UTC; Sam upgraded the plan and the upload resumed at 16:50 UTC.

**Outcome of the pre-registered extra-batch cut-off**

16. *Endpoint K.* The primary cells use K = 16 for every pair. By the pre-registered all-or-nothing rule, K = 64 requires all 64 endpoint responses in every pair, and an extra-batch round starts only if it is projected to finish by RESULTS_DUE − 6.5 h (10-03 17:30 UTC). Final state: batches 1–3 for seeds 43 and 44 and batches 1–2 for seed 45 (all 16 jobs succeeded); at 17:18:35 UTC, when GPUs freed, the queue skipped seed 45's batch 3 (projected end ≈ 17:54 UTC). The main analysis had been run earlier, at 16:34 UTC, on the queue's projection: at that time all four GPUs were committed until ≈ 16:43 and then to seed 45's batches 1–2 until ≈ 17:19, so batch 3 could not start in time. The queue's skip at 17:18 confirms that projection; no result changes.

**Operational issues (no effect on data)**

- The AutoDL SSH gateway was intermittent on 10-02/10-03 (minutes-long connect timeouts); the box-side queue was unaffected.
- The laptop agent session restarted once (no check-ins 10-01 17:52–21:59 UTC) and later missed the end of the required evaluations (finished 10-03 14:56 UTC; analysis started 16:34 UTC) because a watcher timed out during an SSH outage. No decision depended on these gaps.
- `camera_ready/results/` was initially not tracked (repo-root `.gitignore` rule `results*/`); fixed 10-03 08:20 UTC.

**Unresolved alerts**

*(none)*
