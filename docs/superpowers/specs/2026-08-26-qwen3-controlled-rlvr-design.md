# Controlled Qwen3 RLVR Experiment Design

Date: 2026-08-26

## Goal

Add a small, fully traceable reasoning post-training experiment that can serve as the causal backbone for the project’s main scientific question:

> How much of RLVR improvement can be explained by amplification or redistribution of behavior already reachable under the exact pre-RL policy, and where does that explanation break down?

The existing Qwen2.5 SFT/RLVR comparison remains useful as a discovery pilot, but its exact pre-RL lineage is not fully documented. The controlled run must remove that ambiguity by constructing and preserving the entire chain ourselves:

```text
Qwen3-0.6B-Base
  -> reasoning SFT
  -> exact pre-RL policy pi_0
  -> GRPO / RLVR
  -> intermediate policies pi_t
  -> held-out evaluation
```

The defining requirement is:

```text
GRPO step-0 weights == the exact saved final SFT weights
```

No automatic “latest checkpoint” selection, branch guessing, or substitution is allowed for the canonical run.

## Scientific Role

The controlled Qwen3 run becomes the main experiment for causal and mechanistic claims about RL-induced probability redistribution.

The earlier Qwen2.5 experiment remains a pilot / external contrast because it already revealed three motivating phenomena:

- deeper pre-RL sampling substantially weakens apparent answer-level RL-only success;
- a one-parameter global reward tilt explains a nontrivial fraction of the observed per-question change;
- large problem-specific deviations remain, suggesting strongly heterogeneous redistribution.

The controlled run is designed to test those observations again with exact lineage, predefined checkpoint timing, and strict train/eval separation.

## Scope

This design covers:

1. a deterministic SFT data pipeline;
2. full-parameter reasoning SFT of Qwen3-0.6B-Base;
3. a GRPO/RLVR training pipeline initialized from the exact SFT endpoint;
4. dense policy checkpoint preservation across RL training;
5. shallow full-test evaluation and deep fixed-subset sampling;
6. reachability and frozen-pool analyses defined before observing the new results;
7. contamination checks and provenance metadata needed to reproduce the run.

Out of scope for the first canonical run:

- LoRA or QLoRA training;
- 8-bit optimizers;
- multi-GPU FSDP or ZeRO;
- representation probing or latent interventions;
- strategy embeddings as the first explanatory model;
- Qwen3.5 hybrid architectures;
- base-to-RL training without an SFT stage;
- tuning hyperparameters against GSM8K test performance.

A direct Base -> RLVR run may later be added as an ablation, but it is not part of the first controlled experiment.

## Hardware Boundary

Canonical training and canonical probability measurements use Linux + CUDA.

The target training machine is:

```text
1 x NVIDIA A40 48 GB
```

Apple Silicon remains useful for local development, data inspection, parser/scorer work, smoke tests, and small exploratory inference. It is not the canonical training backend for this experiment.

The training implementation should favor standard PyTorch/TRL/Open-R1-compatible components over platform-specific optimizations.

## Model and Provenance

### Base model

Use:

```text
Qwen/Qwen3-0.6B-Base
```

The exact Hugging Face revision must be resolved to and recorded as an immutable commit SHA before training begins.

The experiment manifest must record at least:

```text
base model repository
base model commit SHA
tokenizer repository and commit SHA
SFT dataset repository and revision
RL dataset repository and revision
software/package versions
random seeds
training configs
output checkpoint paths / revisions
```

### Exact pre-RL policy

The final SFT model is named `pi_0` for analysis purposes.

GRPO must load that exact saved checkpoint directly. The training code must not infer the checkpoint from a repository name, choose the numerically largest checkpoint, or resolve `latest` / `auto` for the canonical run.

The run manifest must therefore be sufficient to prove:

```text
pi_GRPO(step=0) == pi_SFT(final)
```

## Data Design

### SFT data

Use a fixed 10,000-example subset of:

```text
open-r1/OpenR1-Math-220k
```

The goal is to create a competent but non-saturated reasoning policy, not to maximize benchmark accuracy.

The subset construction must be deterministic and saved as an explicit manifest of source example identifiers and/or stable hashes. The subset is fixed before SFT begins.

Selection rules:

- one verified-correct reasoning trace per problem;
- trace must be complete rather than truncated;
- formatted training sequence must fit within 2048 tokens;
- examples that conflict with contamination rules below are removed and replaced deterministically until the subset again contains exactly 10,000 examples.

Do not inspect GSM8K test performance to decide which SFT checkpoint becomes `pi_0`.

### RL data

Use the GSM8K training split only:

```text
D_RL = GSM8K train
```

The first canonical reward is deliberately minimal:

```text
r(x, z) = 1 if the extracted final numeric answer is correct, else 0
```

Do not add a format reward in the first canonical run. SFT is responsible for establishing the reasoning/output format. Keeping RL reward binary makes the later probability-redistribution analysis easier to interpret.

### Evaluation data

Use GSM8K test only for evaluation:

```text
D_eval = GSM8K test
```

GSM8K test must not enter SFT example selection, SFT training, GRPO training, hyperparameter selection, or checkpoint selection.

The existing fixed 30-question GSM8K test subset remains the deep-sampling subset so the new controlled run can be compared with the earlier Qwen2.5 pilot.

## Contamination Audit

Before SFT training, compare the candidate OpenR1 10k subset against the union of GSM8K train and test.

Perform at least three levels of duplicate screening:

1. normalized exact text match;
2. match after standardizing whitespace, punctuation, and numeric formatting;
3. near-duplicate screening using a deterministic token- or shingle-based similarity rule.

Any suspicious overlap is removed from the SFT subset and replaced deterministically from the remaining eligible OpenR1 pool.

Record:

```text
candidate count
removed exact duplicates
removed normalized duplicates
removed near-duplicates
final 10k source IDs / hashes
matching thresholds / normalization rules
```

The audit is a reproducibility artifact, not an informal manual check.

## SFT Training Design

Use full-parameter SFT rather than LoRA so the pre-RL policy is not constrained to a low-rank adaptation subspace.

Canonical SFT configuration:

```text
model: Qwen/Qwen3-0.6B-Base
data: fixed 10,000-example OpenR1 subset
training: full parameter
epochs: 2
max sequence length: 2048
precision: bf16
attention: FlashAttention 2
gradient checkpointing: enabled
sequence packing / padding-efficient batching: enabled
optimizer: fused AdamW
learning rate: 2e-5
scheduler: cosine
warmup ratio: 0.03
weight decay: 0.01
per-device batch size: 8
gradient accumulation steps: 8
effective batch size: 64
seed: 42
```

The final epoch-2 checkpoint is `pi_0` by definition.

An epoch-1 checkpoint may be preserved for diagnostics, but it must not replace `pi_0` based on GSM8K evaluation results.

The SFT training output must contain a self-describing config and provenance manifest.

## GRPO / RLVR Training Design

GRPO starts only from the exact `pi_0` checkpoint.

Canonical GRPO configuration:

```text
training data: GSM8K train
epochs: 1
reward: binary final-answer correctness only

num_generations: 8
temperature: 0.8
top_p: 0.95
top_k: 0
repetition_penalty: 1.0

max prompt length: 512
max completion length: 1024

learning rate: 1e-6
scheduler: cosine
warmup ratio: 0.10
optimizer: fused AdamW
max_grad_norm: 1.0

precision: bf16
attention: FlashAttention 2
gradient checkpointing: enabled

vLLM generation: enabled
vLLM mode: colocate

KL beta: 0.0
clip epsilon: 0.2
num_iterations: 1
loss_type: dapo
scale_rewards: group
seed: 42
```

The config must write all experiment-defining values explicitly rather than relying on library defaults.

### Why no KL in the first run

The first canonical run follows the common modern GRPO/R1-style setting with `beta=0`, avoiding a separate reference-model KL term and reducing memory pressure. If the run exhibits pathological policy drift, a positive-KL run becomes a separate ablation rather than a silent change to the canonical objective.

### Why DAPO-style loss normalization

Use explicit `loss_type=dapo` rather than the original response-normalized GRPO loss because response-length changes are themselves scientifically relevant in this project. Avoiding a known length-normalization bias makes later interpretation cleaner.

## Training Pilot Before the Canonical GRPO Run

Before the full canonical GRPO run, execute a short 20-50 optimizer-step engineering pilot from `pi_0`.

The pilot checks only:

- reward has nonzero variance across groups;
- loss and gradients remain finite;
- completion lengths do not overwhelmingly hit the 1024-token cap;
- vLLM colocate works reliably;
- GPU memory use is safe on A40 48 GB;
- throughput is sufficient for the planned run;
- saved checkpoints can be reloaded.

Do not use pilot benchmark performance to tune the canonical hyperparameters.

After the pilot passes, discard it as scientific data and restart the canonical run from the untouched `pi_0` checkpoint.

If a purely engineering failure forces a configuration change, document the reason, update the design/spec before the canonical run, and restart from `pi_0`.

## Performance Optimizations Allowed in the Canonical Run

Allowed engineering optimizations must preserve the intended model class and objective:

- bf16;
- FlashAttention 2;
- gradient checkpointing;
- gradient accumulation;
- sequence packing / padding-efficient SFT batching;
- fused AdamW;
- vLLM generation in colocate mode;
- vLLM sleep/offload mode only if needed for memory stability and documented explicitly.

Do not introduce LoRA/QLoRA, 8-bit optimizer states, quantized training weights, FSDP, or ZeRO solely to make the A40 run fit unless a later design change explicitly approves them.

## Checkpoint Policy

Preserve model-only policy checkpoints every 5% of canonical GRPO training progress:

```text
pi_0, pi_5, pi_10, ..., pi_100
```

`pi_0` is the final SFT checkpoint and is not regenerated by GRPO.

Preserve resumable optimizer/scheduler state at approximately:

```text
25%, 50%, 75%, 100%
```

The exact mapping from percentage to optimizer step must be determined from the finalized number of canonical training steps before the run begins and written to the run manifest.

The primary scientific analysis is predefined to use five policies closest to:

```text
pi_0, pi_25, pi_50, pi_75, pi_100
```

The denser 5% checkpoints are retained for later trajectory inspection but are not used to cherry-pick interesting turning points in the initial analysis.

## Evaluation Protocol

Canonical evaluation uses the project’s CUDA vLLM inference path with an explicit decoding configuration.

### Wide evaluation

For each primary checkpoint, evaluate the full GSM8K test set with:

```text
K = 8 rollouts per question
```

Record at least:

- sample accuracy;
- pass@k curve available from K=1..8;
- response-length statistics;
- parsed-answer / extraction diagnostics;
- reward/correctness distribution.

The purpose is broad learning-curve measurement, not deep support estimation.

### Deep evaluation

For the fixed 30-question subset, evaluate each primary checkpoint with:

```text
K = 256 rollouts per question
```

This yields per-question estimates:

```text
p_i(0), p_i(25), p_i(50), p_i(75), p_i(100)
```

Analyze both probability changes

```text
Delta_i(t) = p_i(t) - p_i(0)
```

and effective log-odds shifts

```text
beta_i(t) = logit(p_i(t)) - logit(p_i(0))
```

with appropriate finite-sample handling near probabilities 0 and 1.

### Pre-RL deep reachability bank

For `pi_0`, generate an additional deep bank on the same 30 questions:

```text
K = 1024 rollouts per question
```

Use this fixed bank to estimate observed answer-level reachability as a function of sampling budget.

Construct resampling/subsampling curves for at least:

```text
K = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
```

The scientific wording must remain observational:

```text
not observed in K pre-RL samples
```

not:

```text
outside the support of the pre-RL model
```

## Frozen-Pool Analysis Hierarchy

Predefine increasingly flexible explanations rather than starting with a high-capacity model.

### Level 1: global reward tilt

For each evaluation time `t`, fit a single common shift:

```text
q_i(t; beta) = sigmoid(logit(p_i(0)) + beta_t)
```

Equivalent interpretation for binary reward:

```text
correct-trajectory odds are multiplied by exp(beta_t)
```

This is an explanatory fixed-pool reweighting baseline, not a simulation of the actual GRPO optimizer.

Primary fitting/evaluation should use leave-one-question-out or another predefined held-out procedure so a question’s post-RL count does not directly tune the parameter used to predict that same question.

### Level 2: difficulty-conditioned tilt

If Level 1 leaves substantial structured residuals, allow a small number of pre-RL difficulty bins with separate tilt parameters, for example:

```text
beta_easy, beta_medium, beta_hard
```

Difficulty bins must be defined from `pi_0` information only and before inspecting the held-out post-RL residual for a question.

### Level 3: strategy redistribution

Only after the scalar and low-capacity controls are evaluated should the analysis inspect richer behavior classes such as:

- final-answer modes;
- response length;
- interpretable reasoning-strategy categories;
- clustering of reasoning traces.

Embedding or representation methods are not the first explanatory model and require a separate design if they become part of the main analysis.

## Predefined Primary Metrics

The main frozen-pool comparison must report at least three complementary quantities.

### Held-out binomial negative log likelihood

Compare post-RL counts under:

```text
null: p_i(t) = p_i(0)
```

against the held-out frozen-pool tilt prediction.

### Delta SSE explained

Report:

```text
1 - sum_i (Delta_i - Delta_hat_i)^2 / sum_i Delta_i^2
```

This measures reduction in squared prediction error relative to the zero-change baseline.

### Correlation of actual and predicted changes

Report:

```text
corr(Delta_i, Delta_hat_i)
```

This guards against a model looking strong only because it captures the broad mean improvement while failing to predict which individual questions improve, stagnate, or regress.

Aggregate accuracy alone is not evidence that the frozen-pool model predicts per-question behavior.

## Finite-Sample Uncertainty

The first analysis may use empirical probabilities for descriptive plots, but the formal analysis must acknowledge that both pre-RL and post-RL probabilities are estimated from finite rollout banks.

At minimum, use bootstrap/resampling over stored rollout outcomes to quantify uncertainty in:

- reachability curves;
- per-question probability changes;
- global tilt parameters;
- held-out prediction metrics.

Near-zero and near-one probabilities require finite-sample-safe log-odds handling rather than raw infinite logits.

## Interpretation Rules Defined Before Seeing the New Results

The experiment is not framed as a binary proof/disproof of capability expansion.

### Outcome A: strong redistribution explanation

If deep `pi_0` sampling covers nearly all later answer-level successes and low-capacity tilt models explain most post-RL change, the supported claim is:

> RLVR gains in this controlled setting are largely consistent with redistribution of behavior already observable under deep sampling of the pre-RL policy.

Do not generalize this to all tasks or all LMs.

### Outcome B: high reachability, weak scalar reweighting

If later answer-level successes are largely reachable under `pi_0`, but global/difficulty tilt has poor held-out predictive power and strategy composition changes strongly, the supported claim is:

> RLVR largely reuses answer-level behavior reachable under the pre-RL policy, while reorganizing that behavior in a strongly selective or strategy-dependent way.

### Outcome C: persistent observed boundary expansion

If a stable set of later successes remains absent from the 1024-sample `pi_0` bank and appears systematically over training, the supported claim is:

> The run provides evidence consistent with behavioral boundary expansion beyond what deep pre-RL sampling reveals.

This still does not establish literal support expansion for a softmax language model.

## Result Storage and Reproducibility

Training artifacts and canonical raw results must not rely on ephemeral Pod storage.

Persist at least:

- final SFT / `pi_0` weights;
- GRPO policy checkpoints required by this design;
- resumable canonical checkpoints;
- training configs and manifests;
- contamination-audit output;
- deterministic SFT subset manifest;
- wide/deep evaluation raw rollouts;
- pre-RL 1024-rollout bank;
- analysis summaries.

Large model checkpoints should live in a model/artifact store rather than Git. Raw rollout text should continue to use private storage during the research stage. Git tracks code, configs, manifests, lightweight summaries, and design/analysis documents.

## Testing Strategy

Implementation follows TDD where behavior can be isolated.

Unit/integration tests should cover at least:

- deterministic SFT subset selection;
- contamination normalization and matching;
- final subset size exactly 10,000;
- explicit model/dataset revision recording;
- exact `pi_0` path propagation into GRPO initialization;
- reward extraction and numeric correctness;
- explicit sampling/training configuration serialization;
- checkpoint-step mapping;
- evaluation subset identity;
- frozen-pool metric calculations.

Hardware acceptance on A40 should verify:

- SFT smoke training can save/reload a checkpoint;
- GRPO engineering pilot produces finite loss/reward and reloadable output;
- vLLM colocate generation works;
- canonical checkpoint metadata points back to the exact `pi_0` lineage;
- CUDA vLLM evaluation can load the saved policies.

## Failure Handling

- If Qwen3-0.6B-Base cannot run through the chosen SFT/TRL stack without model conversion, stop and treat conversion as a separate design decision.
- If the 20-50 step pilot shows widespread truncation at 1024 tokens, inspect the pre-RL length distribution before changing the cap. Any new cap must be fixed before restarting the canonical run.
- If A40 memory is insufficient despite the approved efficiency stack, first inspect vLLM colocate/cache allocation and batch sizing. Do not silently switch to LoRA/QLoRA or quantized optimizer states.
- If the binary reward parser is unreliable, stop before canonical RL training and fix/validate scoring first.
- If contamination removal materially changes the available SFT pool, preserve the audit and deterministic replacement process rather than manually selecting replacements.
- If a canonical run is interrupted, resume only from a documented resumable checkpoint. Do not splice pilot and canonical trajectories.

## Acceptance Criteria

The controlled experiment design is successfully implemented when all of the following are possible:

1. deterministically materialize the same contamination-audited 10k SFT subset from pinned source revisions;
2. full-finetune Qwen3-0.6B-Base on A40 and save the predefined final SFT policy `pi_0`;
3. prove from saved metadata that GRPO initializes from exactly `pi_0`;
4. run the approved GRPO recipe and preserve predefined policy checkpoints through `pi_100`;
5. run full-test K=8 evaluation on the five primary policies;
6. run 30-question K=256 deep evaluation on the five primary policies;
7. run a 30-question K=1024 pre-RL reachability bank;
8. reproduce reachability curves and the predefined frozen-pool metrics from saved raw results without rerunning the models;
9. persist all canonical configs, provenance, checkpoints, and raw research data outside ephemeral Pod storage;
10. keep the Qwen2.5 pilot scientifically distinguishable from the controlled Qwen3 causal run.

## Expected Workflow

```text
M5 / local development
  -> implement and unit-test data/reward/config logic
  -> build contamination-audited SFT manifest
  -> inspect configs and smoke inference where useful

A40 Pod
  -> verify CUDA/software environment
  -> SFT engineering smoke
  -> canonical 2-epoch SFT
  -> freeze final SFT as pi_0
  -> short GRPO engineering pilot from pi_0
  -> discard pilot outputs as scientific data
  -> canonical GRPO restart from untouched pi_0
  -> save predefined pi_t checkpoints
  -> canonical CUDA evaluations
  -> upload checkpoints and raw results
  -> terminate Pod

Local analysis
  -> reachability-depth analysis
  -> global frozen-pool tilt
  -> difficulty-conditioned control if justified
  -> residual / strategy analysis
```

This workflow keeps model lineage exact, training data boundaries explicit, and the primary analyses fixed before the new controlled results are observed.