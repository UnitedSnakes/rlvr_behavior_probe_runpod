# Camera-ready progress log

All times UTC. Deviations are tagged **DEVIATION**, alerts **ALERT**.

| Time (UTC) | Event |
|---|---|
| 2026-10-01 04:52 | Session start. Main checkout `codex/signal-ledger` @`d1b54f5` has untracked `analyses/strict_extractor_robustness/`; left untouched. Work happens in worktree `../rlvr_camera_ready` on new branch `camera-ready` from `d1b54f5`. |
| 04:53 | Box `autodl-4080-4` probed: 4× RTX 4080 SUPER 32,760 MiB, cc 8.9, driver 580.95.05, `/` 4.9 GB free, `/root/autodl-tmp` 200 GB free, no python on PATH, system CUDA 12.9 only. |
| 04:59 | Sent Sam the A-phase request list (seed hard-wired to 42 in trainer and evaluator; HF access; one-pair contingency; paper-source location). |
| 04:59 | Box env build started under `screen` (`cr_env`: uv 0.11.13 → Python 3.12.13 venv `/root/autodl-tmp/envs/rlvr` → `vllm==0.27.1 --torch-backend=cu130` → `docker/requirements-vllm.txt` + `controlled_run/requirements-a40.in`). In parallel (`cr_cuda`): userspace CUDA 13.0.0 toolkit runfile into `/root/autodl-tmp/cuda-13.0` (toolkit only, no driver) to compile flash-attn against torch cu130; the A40 image is `runpod/pytorch:1.0.3-cu1300`, i.e. CUDA 13.0. |
| 05:05 | `INVENTORY.md` written. |
