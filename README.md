# RLVR Behavioral Probe

A small study of what changes during RL post-training. I am mainly looking at whether RLVR improves reasoning by putting more probability on solutions the SFT model can already reach, or by expanding observed solution coverage.

## Preliminary results

On 30 GSM8K test problems with 8 rollouts per problem and a 2048-token budget:

- Sample accuracy: **55.4% (SFT) -> 72.9% (final RLVR)**.
- The RL advantage shrinks with larger `k`: **+17.5 pp at pass@1** and **+3.3 pp at pass@8**.
- Across intermediate checkpoints, about **89–97% of positive gains** come from problems solved at least once by the SFT model.
- Most of the response-length change happens in the first 100 PPO steps, but accuracy continues to change afterward.

This is consistent with substantial probability sharpening, but `0/8` under SFT does not mean a solution had zero probability under the SFT policy.

![Pass@k comparison](figures/pass_at_k.png)

![Accuracy trajectory](figures/trajectory_accuracy.png)

![Response-length trajectory](figures/trajectory_length.png)

## Setup

- Model family: Qwen2.5-1.5B
- SFT: `ns-0/qwen-2.5-1.5b-instruct-reasoning-sft`
- RLVR: `expx/qwen-2.5-1.5b-rlvr-ppo`
- Dataset: fixed 30-problem GSM8K test subset (`seed=42`)
- 8 rollouts per problem
- temperature 1.0, top-p 0.95
- main generation budget: 2048 new tokens

For a problem with `K=8` rollouts, I call a positive gain **already covered** if SFT gets at least one rollout correct and RL gets more correct rollouts. If SFT gets `0/8` and RL gets at least one, I call it **observed coverage expansion**.

## Reproduce the analysis

The raw rollout text is included, so reproducing the analysis does not require running the models again.

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

python summarize_results.py --result-dir results_2048_batched
python sanity_check_results.py \
  --result-dir results_2048_batched \
  --question-file data/gsm8k_subset.jsonl
python plot_results.py
```

For the intermediate checkpoints:

```bash
for d in trajectory/step-*; do
  python summarize_results.py --result-dir "$d"
done
```

If the answer-extraction logic changes, saved rollouts can be rescored without regenerating them:

```bash
python rescore_results.py --result-dir results_2048_batched --in-place
```

To regenerate rollouts:

```bash
python run_probe_batched.py \
  --questions 30 \
  --rollouts 8 \
  --max-new-tokens 2048 \
  --temperature 1.0 \
  --top-p 0.95 \
  --device auto \
  --dtype bfloat16 \
  --result-dir results_2048_batched \
  --resume
```

## Notes

This is a preliminary result from one model family, one dataset, one fixed 30-problem subset, and 8 rollouts per problem. I am using it as a starting point for larger-scale and more controlled tests of what RL changes during post-training.
