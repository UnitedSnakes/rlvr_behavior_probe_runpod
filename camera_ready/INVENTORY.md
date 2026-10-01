# Camera-ready inventory (A1–A3)

Written 2026-10-01 ~05:05 UTC on branch `camera-ready` (worktree off `d1b54f5`).
Every claim cites a file and line at the commit stated. Items marked **PENDING**
could not be resolved from local material and need Hugging Face access to the
private repos.

## 1. Code and commands that produced the seed-42 runs

### 1.1 Training execution commits

| Arm | Execution commit | Evidence |
|---|---|---|
| MaxRL seed 42 | `981475795538eee391c7e86aa022ee609b539770` (2026-09-03 21:49 -0400) | `docs/superpowers/checkpoints/2026-09-04-maxrl-canonical-structural-pass.md:24-25,50` (`execution_commit` from the structural checker); `hf_bundles/2026-09-05-canonical-maxrl-seed42/manifest.json` `training_execution_commit` |
| GRPO seed 42 | **not recorded in any local document**. Inferred: `c664e26` (branch HEAD when the ledger was opened; ledger file stamp `20260902T042720Z`, previous commit `c664e26` at 2026-09-02 04:20:37Z) | `signal_ledger/signal_ledger_20260902T042720Z_rank*.jsonl` file names; `git log` window 2026-09-02 00:00–04:30Z. **PENDING**: confirm from `grpo_run_manifest.json` / launch log in HF repo `HKReporter/rlvr-behavior-probe-grpo-canonical-seed42-2026-09-02` |

`git diff c664e26 9814757 -- controlled_run/` touches only `train_grpo.py` on the
GRPO path, and that change is a behaviour-preserving refactor (objective hooks
that default to `None`; `run_grpo` delegates to `_run_controlled_grpo` with no
transform). `signal_ledger.py`, `rewards.py`, `data.py`, `config.py`,
`checkpointing.py` and `configs/` are byte-identical between the two commits.
Both arms can therefore run from `9814757` and the GRPO arm executes the same
training semantics as at `c664e26`.

### 1.2 Launch commands (documented form)

```bash
torchrun --nproc_per_node=2 -m controlled_run.train_grpo  --pi0-dir <pi0> --output-dir controlled_run_outputs/grpo_canonical --mode canonical
torchrun --nproc_per_node=2 -m controlled_run.train_maxrl --pi0-dir <pi0> --output-dir controlled_run_outputs/maxrl_canonical_seed42 --mode canonical
```

The `--mode canonical` form and `torchrun --nproc_per_node=2` come from
`docs/superpowers/checkpoints/2026-09-03-maxrl-pilot-acceptance-checker-complete.md:291-298`
(pilot form) and `controlled_run/train_grpo.py:366` @9814757 (`--mode {pilot,canonical}`).
Output directories come from the evaluation manifests
(`snapshot_eval_train256_k16_cbank/pi_005/snapshot_eval_manifest.json`:
`canonical_run_dir = controlled_run_outputs/grpo_canonical`). The `--config`
default is `controlled_run/configs/grpo_qwen3_0_6b.yaml`.

### 1.3 Config (authoritative)

`controlled_run/configs/grpo_qwen3_0_6b.yaml` @9814757, validated against
`GRPO_INVARIANTS` (`controlled_run/config.py:34-74` @9814757). It matches the
paper's Appendix A table item by item, including
`vllm_gpu_memory_utilization: 0.30` (`config.py:61`), `save_steps: 0.25`
(`train_grpo.py:109`), `report_to: "none"` (`train_grpo.py:110`).
Config SHA-256 recorded by the evaluator: `c18b6656c50abdc139fda2c15e890dae5cd0b425caab9b1b011aff143633a71f`
(`snapshot_eval_manifest.json` `grpo_config_sha256`).

### 1.4 Evaluator and π0 banks

| Artifact | Code | Evidence |
|---|---|---|
| Snapshot evaluation (canonical = sequential, one question per request) | `controlled_run/eval_snapshot.py` @`1c26b1f` (GRPO evaluation ran on the GRPO-only predecessor `1524571`; the only diff is MaxRL-manifest support) | `hf_bundles/2026-09-05-canonical-maxrl-seed42/manifest.json` `sequential_evaluator_implementation_commit`; the batched variant `dcb6f65` failed parity (`docs/superpowers/checkpoints/2026-09-04-maxrl-cbank-batching-parity-fail.md`) and is not canonical |
| π0 K=32 bank (banks A and B) | `diagnose_p0_signal_budget.py` @`5b10742` | `docs/superpowers/checkpoints/2026-09-02-k32-train-p0-rebaseline.md:30-34`; `p0_train_k32_top_p1_canonical/summary_shard*.json` |

Evaluator settings at `eval_snapshot.py` @1c26b1f: vLLM `LLM(dtype=bfloat16,
tensor_parallel_size=1, gpu_memory_utilization=0.50, max_model_len=2560)`
(lines 337, 396-403); per-question `SamplingParams(n=16, temperature, top_p,
top_k, repetition_penalty, max_tokens=2048, seed=question_seed)`.

### 1.5 A40 software stack (seed-42 run records win)

From the A40 evaluation manifests (`runtime` block, `snapshot_eval_manifest.json`):
Python 3.12.13, torch 2.13.0+cu130 (CUDA 13.0), transformers 5.15.0,
datasets 5.0.1, vLLM 0.27.1, GPU "NVIDIA A40". From the image recipe
(`docker/Dockerfile` asserts, lines with `assert ... __version__`): accelerate
1.14.0, TRL 1.12.0, flash-attn 2.8.3.post1 built from source with
`FLASH_ATTN_CUDA_ARCHS=80`; base image `runpod/pytorch:1.0.3-cu1300-torch290-ubuntu2404`;
uv 0.11.13. These agree with the stack in the 3090 bridge report; no
disagreement found. A full A40 `pip freeze` is not stored locally (**PENDING**,
may exist in the HF run repos); transitive packages (cuBLAS, NCCL, Triton,
FlashInfer) are pinned through torch 2.13.0+cu130 and vLLM 0.27.1.

## 2. Artifacts: location and reachability (checked 2026-10-01 05:05 UTC)

| Artifact | Where | Reachable now |
|---|---|---|
| π0 (`pi0_lineage_id f89fc902…`) | HF `HKReporter/rlvr-behavior-probe-pi0-corrected-canonical-2026-08-30`; partial local copy `~/rlvr_data/weights/pi0/pi_0` | Local copy lacks `training_args.bin`, which `load_pi0_manifest` fingerprints (`controlled_run/checkpointing.py:69-92` @9814757) → training needs the HF copy. **Needs HF token.** |
| GRPO seed-42 snapshots pi_025/045/065 | HF `HKReporter/rlvr-behavior-probe-grpo-canonical-seed42-2026-09-02` | **Needs HF token** |
| GRPO seed-42 pi_100 | local `~/rlvr_data/weights/grpo_end/pi_100` (+ HF) | yes |
| MaxRL seed-42 snapshots pi_025/045/065 | HF `HKReporter/rlvr-behavior-probe-maxrl-canonical-seed42-2026-09-05` | **Needs HF token** |
| MaxRL seed-42 pi_100 | local `~/rlvr_data/weights/maxrl_end/pi_100` (+ HF) | yes |
| GRPO ledger (2 rank files) | local `signal_ledger/` (sha256 `8925c395…`, `23b9fa7a…`) | yes |
| MaxRL ledger (2 rank files) | local `~/rlvr_data/maxrl_seed42/…canonical…/signal_ledger/` (sha256 `a1419f80…`, `b41afab1…`) | yes |
| π0 banks A/B (K=32 split into two n=16 calls) | local `p0_train_k32_top_p1_canonical/rollouts_shard{0,1}of2.jsonl` (sha256 `9a183634…`, `3f005b85…`) | yes |
| GRPO per-snapshot evaluations (20 × question-level + response-level with token ids) | local `snapshot_eval_train256_k16_cbank/pi_XXX/snapshot_raw.jsonl` | yes |
| MaxRL per-snapshot evaluations (20) | local `~/rlvr_data/maxrl_seed42/…analysis…/fixed_panel/maxrl_snapshot_eval_train256_k16_cbank/pi_XXX/` | yes |
| Run manifests (`grpo_run_manifest.json`, `maxrl_run_manifest.json`, `policy_snapshot_schedule.json`, `pi_XXX/policy_metadata.json`) | HF model repos | **Needs HF token** (the evaluator requires them, `eval_snapshot.py` @1c26b1f lines 48-120) |
| Trainer logs / `trainer_state.json` (grad norms) | HF model repos | **Needs HF token**, PENDING |
| Paper source and exports | `~/Downloads/attrib_draft_v5_source/` (not in the repo; v5, 2026-09-06; `data_manifest.sha256` included) | yes; to be confirmed as the accepted version |

## 3. What `seed` controls

| Mechanism | Uses the training seed? | Evidence |
|---|---|---|
| Question order (RepeatSampler shuffle) | **yes**, `seed=self.args.seed`; same for both arms with the same seed | TRL v1.12.0 `trl/trainer/grpo_trainer.py:1277-1284`; `args.seed = config["seed"]` (`train_grpo.py:111` @9814757) |
| `data_seed` | set to the same value | `train_grpo.py:112` @9814757 |
| torch / trainer RNG | yes, `set_seed(args.seed, device_specific=True)` | TRL v1.12.0 `grpo_trainer.py:1083` |
| Colocated vLLM rollout sampler | **not the training seed**: the engine gets a fixed rank seed `accelerator.process_index // tensor_parallel_size` (0 on rank 0, 1 on rank 1); no per-request seed is passed | TRL v1.12.0 `trl/generation/vllm_generation.py:357`. Every run, whatever its training seed, starts its rollout sampler from the same rank seeds; the rollouts differ across training seeds because the prompts and the policy differ. Token-level reproduction across hardware is not guaranteed. The original implementation is kept unchanged |
| Snapshot evaluation seed | `config["seed"]*100000 + dataset_index + 75000`; the "42" is the **training-config seed field**, not a constant | `eval_snapshot.py:29-35,236` and `sample_p0.py:45` @1c26b1f |
| π0 bank seeds | A: `42*100000 + idx`; B: `42*100000 + idx + 50000`; engine `LLM(seed=42)` | `diagnose_p0_signal_budget.py:66-73,285` @5b10742; checkpoint `2026-09-02-k32-train-p0-rebaseline.md:30-34` |

**Seed is hard-wired to 42.** `GRPO_INVARIANTS["seed"] = SEED = 42`
(`config.py:73`, `constants.py:13`), and `validate_grpo_config` is enforced by
the trainer, the MaxRL entrypoint and the evaluator. The evaluator also requires
`canonical_manifest["config"] == loaded config` (`eval_snapshot.py:346` @1c26b1f).
So at the seed-42 commit neither training nor evaluation of a seed ≠ 42 runs
without a code change. Later commit `14aba88` added an A40 replication lane
whitelisting only seeds 43 and 44 (`config.py` `GRPO_A40_REPLICATION_ALLOWED_SEEDS`).
Raised with Sam before any change (see `PROGRESS.md`).

## 4. Question schedule under a new seed

The order of the 7,472 training groups is the RepeatSampler permutation drawn
from `args.seed`, so a new seed gives a new random order shared by both arms of
the pair. No config knob beyond `seed` is needed. Ledger evidence of shuffling:
the first seed-42 GRPO ledger row is `dataset_index 7035`, `generation_global_step 0`.

## 5. Checkpoints, logging, reporting

- `PolicySnapshotCallback` saves model + tokenizer + `policy_metadata.json` at
  every 5 % (`progress_step_map`, `train_grpo.py:50-61,128-170` @9814757). With
  3,736 steps: 25 % → 934, 45 % → 1681, 65 % → 2428, 100 % → 3736
  (`docs/superpowers/checkpoints/2026-09-02-grpo-canonical-integrity.md`).
- HF Trainer checkpoints (with optimizer state) at `save_steps 0.25` under
  `trainer/` (`train_grpo.py:108-109`).
- `report_to: "none"` (`train_grpo.py:110`); no W&B. `logging_steps` is not set,
  so the Transformers default applies. Whether per-step `grad_norm` is in the
  seed-42 trainer logs is **PENDING** (needs the HF trainer logs).

## 6. Dataset identity

GSM8K `openai/gsm8k`, config `main`, revision
`740312add88f781978c0658806c59bc2815b9866` (training manifest field
`gsm8k_dataset_sha`; evaluation manifest `dataset.sha`). The trainer resolves
the `main` revision at launch (`train_grpo.py:258` @9814757); `main` still
points to `740312a…` today (HF API, 2026-10-01 05:00 UTC). Byte identity of the
256 panel prompts and 7,472 training rows on the box is checked after download
by hashing `build_gsm8k_rl_rows` output and comparing with the panel questions
stored in the seed-42 evaluation rows (to be recorded in `PROGRESS.md`).

## 7. Runtime facts used for planning

- A40 GRPO: ledger opened 2026-09-02 04:27Z; first snapshot evaluation started
  22:46Z → training ≤ ~18 h.
- A40 snapshot evaluation (pi_005): 22:46:20 → 23:32:51 = 47 min per snapshot
  at K=16 (`snapshot_eval_train256_k16_cbank/pi_005.log`).
- Box: 4 × RTX 4080 SUPER 32 GB (compute capability 8.9, no ECC), driver
  580.95.05, system CUDA 12.9 only, no system Python on PATH; container limits
  64 CPUs and 320 GB RAM.

## 8. Training-path differences between the two seed-42 runs (Sam item 5)

Checked 2026-10-01 05:30 UTC. GRPO execution commit is not recorded anywhere
(not in `grpo_run_manifest.json`, not in the GRPO HF repo, which holds no
training log); the inference `c664e26` from §1.1 stands. MaxRL execution commit
`9814757` is confirmed by `execution_git_commit.txt` in the MaxRL HF model repo.
`c664e26` is an ancestor of `9814757`.

### 8.1 Code (`git diff c664e26 9814757`, every file outside docs/tests/analyses)

| File | Change | On the training path? |
|---|---|---|
| `controlled_run/train_grpo.py` | `run_grpo` body moved into `_run_controlled_grpo(recorder_factory=RewardBatchRecorder, trainer_transform=None, manifest_filename="grpo_run_manifest.json", manifest_extra=None, result_extra=None)`; `run_grpo` calls it with defaults | Yes, but behaviour-preserving for GRPO: with the defaults the trainer class, recorder, ledger wrapper, arguments and manifest are exactly those of `c664e26` |
| `controlled_run/maxrl.py` (new) | `compute_practical_maxrl_advantages`; `MaxRLRewardBatchRecorder.peek`; `PracticalMaxRLTrainer._generate_and_score_completions` replaces `output["advantages"]` after TRL computed its GRPO advantages; one extra `accelerator.gather` of rewards; overwrites the latest batch in TRL's `_logs["advantages"]` diagnostic deque | MaxRL arm only. The replaced tensor is the **advantage estimator** (the intended intervention). The extra all-gather is a collective with no RNG use; the `_logs` overwrite is logging-only |
| `controlled_run/train_maxrl.py` (new) | MaxRL entry: same config validation, calls `_run_controlled_grpo` with the MaxRL recorder/transform, writes `maxrl_run_manifest.json` with an extra `objective` block | MaxRL arm only; no other argument differs |
| `controlled_run/eval_snapshot.py`, `prepare_analysis_inputs.py`, `maxrl_pilot_acceptance.py`, `artifact_registry.json` (new) | evaluation / analysis / acceptance | No |
| `controlled_run/distributed_preflight.py` (new), `docker/rlvr-bootstrap.sh` | exact-commit gate and NCCL all-reduce preflight run by the pod bootstrap before training | No (pre-training gates; not imported by training) |
| `.github/workflows/build-runpod-image.yml` | image builds now also triggered from `codex/signal-ledger` | Not code on the training path, but see 8.3 |

Unchanged between the two commits: `signal_ledger.py`, `rewards.py`, `data.py`,
`config.py`, `checkpointing.py`, `provenance.py`, `configs/grpo_qwen3_0_6b.yaml`,
`docker/Dockerfile`, `docker/requirements-vllm.txt`, `controlled_run/requirements-a40.in`.

### 8.2 Run manifests

`grpo_run_manifest.json` (GRPO HF repo @`0ff3639`) and `maxrl_run_manifest.json`
(MaxRL HF repo @`e67069c`) are identical in every shared field — `config`,
`gsm8k_dataset_sha`, `runtime_batch`, `pi0_manifest`, `pi0_lineage_id`,
`prompt_length_audit`, `signal_ledger`, `core_diagnostics`, `mode`,
`scientific_use`, `pilot_steps`. The only extra field is MaxRL's `objective`
block. `policy_snapshot_schedule.json` files are byte-identical.

### 8.3 Environment (not fully identifiable)

- Recorded for both arms (evaluation manifests and MaxRL trainer README):
  Python 3.12.13, torch 2.13.0+cu130, transformers 5.15.0, datasets 5.0.1,
  vLLM 0.27.1; MaxRL additionally TRL 1.12.0, tokenizers 0.22.2, NCCL 2.29.7,
  driver 580.159.04 (`distributed_preflight.json`). No GRPO-side record of TRL,
  NCCL, driver or GPU UUIDs exists.
- The image workflow change (8.1) means the MaxRL pod may have used a later
  build of the same Dockerfile than the GRPO pod. The Dockerfile pins the
  packages above; transitive packages are pinned through torch/vLLM, but an
  image digest for either run is not recorded, so identical images cannot be
  confirmed. The two runs also ran on different days and, very likely,
  different A40 pods.

**Conclusion.** At code level the only training difference is the advantage
estimator (plus logging and one extra collective). Image identity and pod
hardware between the two seed-42 runs cannot be verified from the records. The
new pairs remove this ambiguity by construction: both arms of each pair run
from one checkout of `9814757`, in one environment, on the same box,
concurrently. They can be described as an estimator-only comparison at the
code level; the seed-42 pair can be described that way at the code level only,
with the environment caveat above.

## 9. Further facts found after the first push

- Seed-42 trainer logging used the Transformers default `logging_steps = 10`:
  the MaxRL log has 373 metric records over 3,736 steps, each with `grad_norm`
  (pre-clip norm at that step), `learning_rate`, `step_time`
  (`logs/maxrl_canonical_seed42.log` in HF dataset
  `HKReporter/rlvr-behavior-probe-maxrl-analysis-seed42-2026-09-05`).
  MaxRL `trainer_state.json` files are in the MaxRL model repo under
  `trainer/checkpoint-{934,1868,2802,3736}/`. No GRPO training log or trainer
  state was uploaded. So seed-42 clipping diagnostics can only be a
  1-in-10-step sample for MaxRL and are unavailable for GRPO.
- A40 MaxRL wall clock: launch 2026-09-04 01:56Z, exit 21:25Z → 19.5 h for
  3,736 steps; ~21 s/step at the start, ~17–18 s/step late.
- GSM8K train has 7,473 rows (`prompt_length_audit.count`), the schedule uses
  7,472 groups, so one question per seed is never sampled. Under seed 42 every
  panel question was sampled. Under a new seed a panel question can be the
  unsampled one (probability 256/7,473); it is then "not yet exposed" at every
  cutoff and has zero advantage mass.
- `FLASH_ATTN_CUDA_ARCHS=80` (the A40 image setting) is kept on the sm_89
  RTX 4080 SUPER; whether it runs correctly is decided by the smoke test.
