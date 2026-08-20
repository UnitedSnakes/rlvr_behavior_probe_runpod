# RLVR Behavioral Probe

Separate repo for the same higher-level research question:

> When RL post-training improves reasoning, does it mainly sharpen probability on
> solutions already reachable under the SFT policy, or expand observed solution coverage?

This is behavior-first. CKA/SAE/interventions come later, after there is a behavioral
phenomenon worth explaining.

## Default checkpoints

- SFT: `ns-0/qwen-2.5-1.5b-instruct-reasoning-sft`
- RLVR/PPO: `expx/qwen-2.5-1.5b-rlvr-ppo`

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

If needed:

```bash
huggingface-cli login
```

## First M5 Pro run

```bash
python run_probe.py \
  --questions 30 \
  --rollouts 8 \
  --batch-rollouts 4 \
  --max-new-tokens 384 \
  --temperature 1.0 \
  --top-p 0.95 \
  --device mps \
  --dtype bfloat16 \
  --resume
```

Then:

```bash
python summarize_results.py --result-dir results
```

The run saves each completed question immediately, so `--resume` is safe.

## Definitions

For each problem, let `c_SFT` and `c_RL` be the number correct among K rollouts.

- `c_SFT > 0` and `c_RL > c_SFT`: probability sharpening.
- `c_SFT == 0` and `c_RL > 0`: observed coverage expansion.
- `c_RL < c_SFT`: regression.

Observed coverage expansion is **not** proof of a newly created capability; finite
sampling can miss very low-probability SFT solutions.

Outputs:
- `results/sft_raw.jsonl`
- `results/rl_raw.jsonl`
- `results/per_question.csv`
- `results/summary.txt`
- `results/summary.json`

If the 30x8 run shows signal, scale behavior first (100-200 questions, K=16-32,
intermediate PPO checkpoints) before doing mechanistic analysis.


## Important SFT repository detail

The SFT repository's `main` branch is metadata-only; actual SFT checkpoints are
stored on branches. This v2 code automatically lists the repository refs and
selects the highest-numbered checkpoint branch.

It also deliberately uses the official
`Qwen/Qwen2.5-1.5B-Instruct` tokenizer/chat template for both SFT and RL
checkpoints so the tokenizer itself is not a confound.

If auto-selection prints multiple branches and you want a specific one:

```bash
python run_probe.py ... --sft-revision <branch-name>
```
