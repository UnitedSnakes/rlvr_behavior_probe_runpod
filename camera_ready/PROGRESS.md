# Camera-ready progress log

All times UTC. Deviations are tagged **DEVIATION**, alerts **ALERT**.

| Time (UTC) | Event |
|---|---|
| 2026-10-01 04:52 | Session start. Main checkout `codex/signal-ledger` @`d1b54f5` has untracked `analyses/strict_extractor_robustness/`; left untouched. Work happens in worktree `../rlvr_camera_ready` on new branch `camera-ready` from `d1b54f5`. |
| 04:53 | Box `autodl-4080-4` probed: 4× RTX 4080 SUPER 32,760 MiB, cc 8.9, driver 580.95.05, `/` 4.9 GB free, `/root/autodl-tmp` 200 GB free, no python on PATH, system CUDA 12.9 only. |
| 04:59 | Sent Sam the A-phase request list (seed hard-wired to 42 in trainer and evaluator; HF access; one-pair contingency; paper-source location). |
| 04:59 | Box env build started under `screen` (`cr_env`: uv 0.11.13 → Python 3.12.13 venv `/root/autodl-tmp/envs/rlvr` → `vllm==0.27.1 --torch-backend=cu130` → `docker/requirements-vllm.txt` + `controlled_run/requirements-a40.in`). In parallel (`cr_cuda`): userspace CUDA 13.0.0 toolkit runfile into `/root/autodl-tmp/cuda-13.0` (toolkit only, no driver) to compile flash-attn against torch cu130; the A40 image is `runpod/pytorch:1.0.3-cu1300`, i.e. CUDA 13.0. |
| 05:05 | `INVENTORY.md` written. |
| 05:00 | Sam's replies received (decisions 1–6, new TRAIN_CUTOFF 2026-10-03 16:00Z, RESULTS_DUE 2026-10-04 00:00Z). Recorded verbatim in §Decisions below. |
| 05:08 | **DEVIATION (resource-only)**: uv default index switched to the aliyun PyPI mirror over HTTPS after PyPI through the academic proxy dropped connections; the torch cu130 index is still download.pytorch.org through the proxy. Package versions are pinned as in `docker/Dockerfile`; the lock is recorded after install. |
| 05:10 | Userspace CUDA toolkit 13.0.0 (`nvcc V13.0.48`) installed at `/root/autodl-tmp/cuda-13.0` (runfile `cuda_13.0.0_580.65.06_linux.run`, toolkit only). |
| 05:12 | `INVENTORY.md` §8: training-path diff of the two seed-42 arms (only the estimator differs at code level; image/pod identity unverifiable). Seed-42 reproduction gate `camera_ready/analysis/reproduce_seed42.py`: 1,278 checks against the paper's rounded numbers and the bundle exports at all 20 snapshots, 0 failures (commit `844e8f1`). |
| 05:13:41 | **PREREG_BRIDGE.md pushed** to `origin/camera-ready`, commit `556a2715438c578f375eb9e16ec04ff8c03a96b1`, file SHA-256 `8a07cd1b48c2f93fa3e5847b79c24c77872114e57f248b2f0ac8373511aee829`. No box evaluation output existed at that time (box had no Python environment yet). |
| 05:14 | HF read-only download to `/root/autodl-tmp/a40` started (token via stdin into the screen session's environment only; unset at the end). Box worktrees: `train_9814757`, `eval_1c26b1f`, `cr` (camera-ready). |

## Decisions from Sam (2026-10-01 ~05:00 UTC)

1. Launcher and evaluation wrapper approved. Only the `seed == 42` check is relaxed; all other manifest/config/data checks stay. Evaluation seed base = each pair's training seed (43/44/45), identical for both arms of a pair. Extra endpoint batches add +1,000,000 / +2,000,000 / +3,000,000 to the per-question seed; collisions with every used seed (incl. π0 banks) are checked before `PREREG_RUNS.md`. The launcher is first run with seed 42 to confirm the resolved config equals the original run field by field. **DEVIATION** (logged).
2. HF token via stdin, read-only downloads, never printed, no shell tracing, unset after use.
3. Seeds 43, 44, 45 launch automatically in order. Each pair launches only if, with its training projected to end, all launched pairs' four K=16 evaluations, analysis and the independent fact-check can finish before RESULTS_DUE with ≥ 2 h buffer (training uses all four GPUs; deferred evaluations count). Seed 43 launches unconditionally. Training must end by TRAIN_CUTOFF.
4. Evaluation order: after each pair, its endpoint K=16 evaluations first; the other three K=16 steps after all training; extra endpoint batches after all required evaluations, skipped if time runs out. Pair validity unchanged; missing extra batches never exclude a pair. **DEVIATION** (logged).
5. Training-path diff of the two seed-42 commits before launch (done, INVENTORY §8).
6. Paper source: `~/Downloads/attrib_draft_v5_source/` must reproduce brief §3 (done, PASS); compare against the accepted PDF for text differences (non-blocking).
