# Controlled Qwen3 RLVR — 2026-08-27 Design Amendment

This amendment supersedes the GRPO batching/truncation details in `2026-08-26-qwen3-controlled-rlvr-design.md` and the corresponding Task 6 mapping assertions in the implementation plan. All other design commitments remain in force unless explicitly changed below.

## Scientific framing

The controlled run is not a benchmark-maximization exercise. The primary object is the flow of per-problem behavior probability under RL:

```text
pre-RL probability / reachability p_i(0)
        ↓
objective- and sampling-induced learning signal
        ↓
p_i(t) / logit shift beta_i(t)
        ↓
behavior redistribution
        ↓
new answer-level or strategy-level support, if any
```

The main question remains how much RLVR improvement is amplification/reweighting of behavior already reachable under `pi_0`, and where that explanation fails. A complementary objective-level question is how the training rule allocates signal as a function of pre-RL pass rate.

## Revised canonical GRPO recipe

Use the following single-device canonical recipe unless the engineering pilot forces a documented change:

```text
training data: GSM8K train only
epochs: 1
reward: binary final-answer correctness only

num_generations: 16
temperature: 0.8
top_p: 0.95
top_k: 0
repetition_penalty: 1.0

hard max prompt length: 512; never silently truncate
max completion length: 1024
mask_truncated_completions: true
vllm_max_model_length: 1536

learning rate: 1e-6
scheduler: cosine
warmup ratio: 0.10
optimizer: fused AdamW
max_grad_norm: 1.0

bf16
FlashAttention 2
gradient checkpointing
colocated vLLM; gpu memory utilization 0.30

beta: 0.0
epsilon: 0.2
num_iterations: 1
loss_type: dapo
scale_rewards: group

per_device_train_batch_size: 8
gradient_accumulation_steps: 4
generation_batch_size: 32
seed: 42

vLLM importance-sampling correction: true
IS mode: sequence_mask
IS cap: 3.0
```

On one device, `generation_batch_size / num_generations = 2`, so each generation batch contains two independent prompts with 16 completions each rather than one prompt with eight completions. The effective optimizer batch is 32 examples on one device.

### Why G=16 rather than G=8

With binary group-relative rewards, an all-correct or all-wrong group has zero reward variance and therefore no useful centered policy-gradient signal. The probability of such a group is `p^G + (1-p)^G`, which is especially high for rare-success prompts. Since low-probability reachable behavior is central to the scientific question, `G=8` would create avoidable gradient starvation exactly in the regime of interest.

`G=16` is a compromise: it materially improves the chance of observing rare successes without making rollout cost or the number of distinct training prompts per generation batch pathological. `G` remains a scientifically meaningful future ablation rather than an invisible engineering parameter.

### Reward scaling is part of the mechanism

The canonical run keeps `scale_rewards=group` because it represents a standard GRPO-style training rule. This is not assumed to be neutral: group standard-deviation scaling changes difficulty-dependent weighting. The implementation therefore permits predeclared control runs with reward scaling disabled (`False` / `none`) or batch-scaled, while keeping all other settings fixed.

A later objective comparison may add MaxRL from the same `pi_0`, data, rollout budget, and decoding protocol. That comparison is a scientific intervention, not a replacement for the canonical GRPO run.

### DAPO scope

`loss_type=dapo` means DAPO-style global active-token loss normalization only. The canonical run is not full DAPO: it does not introduce DAPO dynamic sampling, clip-higher, or other objective changes. This normalization is retained to avoid a known response-length normalization artifact while response length itself is a scientific outcome.

## Truncation policy

Keep the initial completion cap at 1024 tokens, but mask truncated completions from the policy loss. The pilot must record at least completion clipped ratio and maximum terminated length. If a material fraction of trajectories hits the cap, raise the cap to 2048 only as a documented engineering correction before the canonical run and restart from untouched `pi_0`.

A hard length boundary must not silently become a reward against long reasoning.

## Pilot gate

Before canonical GRPO, run 20–50 optimizer steps with `scientific_use=false`. Inspect at minimum:

- reward;
- `frac_reward_zero_std` (primary group-degeneracy diagnostic);
- completion clipped ratio / maximum terminated length;
- gradient norm;
- entropy;
- clipping diagnostics;
- vLLM sampling-log-probability mismatch;
- importance-sampling ratio / masking diagnostics.

Do not tune against final GSM8K test performance. Hyperparameter changes are allowed only for an identified engineering/scientific confound and must be documented before restarting the canonical run from `pi_0`.

## SFT 2048-token audit gate

The 2048-token SFT cutoff is retained provisionally, but it must be audited before canonical SFT. The denominator for this audit is the set of verified, contamination-clean candidate traces before applying the length cutoff.

Record:

```text
pre-length-filter candidate count
p50 / p75 / p90 / p95 / p99 formatted token length
fraction > 2048
fraction > 4096
fraction > 8192
removed_too_long
```

The purpose is to detect whether a nominal context limit is actually selecting a substantially shorter/easier reasoning distribution. Do not raise the limit merely to maximize SFT quality; change it only if the live audit shows that 2048 materially changes the intended training population.

## Compute policy

Start the engineering gate on one A40 48 GB. This keeps the batch semantics above exact and avoids silently changing global batch size through data parallelism. If one A40 is insufficient for memory or throughput, a two-A40 run is allowed only after explicitly remapping batch/generation settings so the scientific batch semantics are preserved.
