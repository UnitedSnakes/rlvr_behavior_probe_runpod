# Controlled Qwen3 RLVR — Conversation / Implementation Checkpoint

> **Purpose:** This is a handoff checkpoint for continuing the project in a fresh chat window after the previous conversation reached maximum length. Read this file first, then the linked design and implementation plan. It records both the scientific state and the exact implementation/TDD state.
>
> **Branch:** `controlled-qwen3-rlvr`
>
> **Primary design:** `docs/superpowers/specs/2026-08-26-qwen3-controlled-rlvr-design.md`
>
> **Implementation plan:** `docs/superpowers/plans/2026-08-26-qwen3-controlled-rlvr.md`

## 1. What the project is trying to answer

The central scientific question is:

> **How much of RLVR improvement can be explained by amplification of behavior already reachable under the pre-RL policy, and where does that explanation break down?**

The intended hierarchy of evidence is:

```text
reachability
> frozen-pool control
> strategy redistribution
> objective comparison
> representation probing
```

Important terminology:

- **pre-RL** means the exact policy checkpoint immediately before RL starts, regardless of whether it was produced by SFT or is a base model.
- **post-RL** means a checkpoint after RL.
- Do not call a model “the pre-RL SFT checkpoint” unless exact lineage proves that the RL run initialized from that exact checkpoint.
- Finite `0/K` under pre-RL sampling does **not** imply true zero probability.
- Answer-level reachability is weaker than strategy-level equivalence.

The current controlled-run design was created specifically because the older Qwen2.5 pilot does not give fully transparent RL initialization lineage.

---

## 2. Existing Qwen2.5 pilot: what is solid and what is not

Current old-model pair in the repository:

```text
SFT repo:      ns-0/qwen-2.5-1.5b-instruct-reasoning-sft
SFT revision:  checkpoint-8-of-10
RL repo:       expx/qwen-2.5-1.5b-rlvr-ppo
RL revision:   main
```

Why `checkpoint-8-of-10` was used: it was the latest actual numbered SFT checkpoint branch available when the probe was built. It was **not** chosen because 80% SFT was theoretically preferred, and we never established that PPO initialized from this exact branch.

Therefore the rigorous statement for the old pilot is:

```text
pi_SFT(checkpoint-8) vs pi_final-RL
```

not proven:

```text
pi_RL-step-0 vs pi_RL-final
```

The RL model card/base reference points to the SFT repository, but the exact SFT revision used to initialize PPO was not found. This means some observed SFT→RL difference could in principle contain a checkpoint-lineage mismatch. Treat Qwen2.5 as strong pilot/motivating evidence, not the final causal/mechanistic backbone.

### Initial 30 × 8 pilot

- pass@1: SFT `55.4%` → RL `72.9%` (`+17.5 pp`)
- advantage shrinks to about `+3.3 pp` at pass@8
- roughly 90% of positive gains occurred on problems the SFT model had solved at least once
- response length fell about one third in the first ~100 PPO steps while pass@8 stayed nearly unchanged; later accuracy rose about `+7.9 pp`
- representation correct/incorrect separation did not survive repeated-split testing: 95% interval approximately `[-0.057, 0.044]`

### Deep reachability sampling

On the fixed 30 GSM8K problems, the older SFT policy was sampled to K=256. All 30 problems were solved at least once.

```text
K    SFT covered       RL-success unresolved
1    17.50 [13,21]     11.50 [8,16]
2    22.42 [19,26]      6.59 [3,10]
4    25.82 [23,28]      3.20 [1,6]
8    27.53 [26,29]      1.50 [0,3]
16   28.24 [27,29]      0.82 [0,2]
32   28.61 [28,30]      0.52 [0,1]
64   29.03 [28,30]      0.24 [0,1]
128  29.49 [29,30]      0.02 [0,0]
256  30.00 [30,30]      0.00 [0,0]
```

Safe pilot claim:

> On these 30 GSM8K problems, apparent RL-only answer success disappears under deeper sampling of the pre-RL SFT policy.

Better general framing:

> Observed capability expansion is highly sensitive to the sampling budget used to define pre-RL reachability.

Do **not** generalize this to “RLVR never learns new capabilities.”

---

## 3. Decoder/backend discrepancy that was already resolved

Canonical protocol A:

```text
temperature=1.0
top_p=0.95
top_k=0
repetition_penalty=1.0
```

Under protocol A:

```text
SFT 0.7031 → RL 0.7697   (+6.65 pp)
25 questions gain, 2 tie, 3 loss
```

HF-like protocol B:

```text
temperature=1.0
top_p=0.95
top_k=20
repetition_penalty=1.1
```

Under protocol B:

```text
SFT 0.58646 → RL 0.72214   (+13.57 pp)
28 gain, 1 tie, 1 loss
```

Changing A→B hurts SFT much more than RL. The earlier HF-vLLM discrepancy was traced to inherited Hugging Face generation settings (`top_k=20`, `repetition_penalty=1.1`) versus historical vLLM defaults (`top_k=0`, repetition penalty `1.0`). Matched vLLM reproduces HF almost exactly.

Safe conclusion: the discrepancy came from unmatched effective sampling parameters, primarily repetition penalty, not a material backend effect.

The historical directories:

```text
backfills/results_sft_256_vllm
backfills/results_rl_256_vllm
```

are canonical protocol A even though their old `run_config` metadata did not explicitly record top-k/repetition penalty; the historical code omitted those arguments and vLLM defaults were `0` and `1.0` respectively.

---

## 4. Frozen-pool / reward-tilt spike already completed

This was an analytical spike, not a training algorithm and not actual resampling/generation.

Given an empirical pre-RL success probability `p`, the simplest binary-reward exponential tilt predicts:

```text
q = p exp(beta) / (1 - p + p exp(beta))
```

Equivalently:

```text
q_beta(z) ∝ p_0(z) exp(beta r(z))
```

For binary reward, `exp(beta)` is the success-odds multiplier. The global beta was fit by maximum likelihood against real RL binomial counts, with leave-one-question-out prediction used to assess per-question fit.

Global result on the 30-question protocol-A rollout banks:

```text
questions:                  30
full beta:                  +0.5418
success odds multiplier:    1.719x

SFT aggregate:              0.7031
RL aggregate:               0.7697
predicted aggregate:        0.7700
actual gain:                +0.0665
predicted gain:             +0.0669

null MAE:                   0.0827
tilt MAE:                   0.0512
null NLL:                   2725.10
tilt NLL:                   2606.25
delta SSE explained:        0.395
actual/pred delta corr:      0.288
```

Interpretation:

- a single positive reward tilt is a meaningful first-order approximation;
- the aggregate match is partly mechanical because beta is MLE-fitted;
- the low delta correlation (`0.288`) shows the global tilt poorly predicts **which** problems change how much;
- `delta SSE explained=0.395` means the one-parameter tilt reduces per-question delta squared error by 39.5% relative to a zero-shift baseline. Do **not** translate this into “39.5% of the RL mechanism is sharpening.”

Largest leave-one-question-out residuals included:

```text
qid  p_sft  p_rl   pred   signed_resid
13   .551   .352   .701   -.349
3    .734   .973   .820   +.153
11   .762   .730   .851   -.120
2    .590   .805   .706   +.099
24   .766   .754   .853   -.099
22   .160   .168   .252   -.084
29   .484   .684   .613   +.071
0    .758   .898   .841   +.057
16   .609   .777   .725   +.052
8    .680   .832   .782   +.050
```

Per-question effective tilt:

```text
beta_i = logit(p_i^RL) - logit(p_i^SFT)
```

with Jeffreys boundary correction gave:

```text
common beta                   +0.5418
mean beta_i                   +0.7701
SD beta_i                      0.7039
conditional heterogeneity deviance 302.01
reference df (if SFT p fixed) 29
```

Strong examples:

- q13: `0.551 → 0.352`, effective beta about `-0.813`; huge negative deviation from global tilt.
- q3: `0.734 → 0.973`, effective beta about `+2.492`; huge positive deviation.
- q4: `0.395 → 0.543`, effective beta about `+0.598`; close to the global beta and a useful ordinary/control example.

The current qualitative picture is:

> **broad positive reward tilt + strong problem-specific redistribution**

This is descriptive, not yet a causal mechanism claim. A more formal future model should account for finite-sample uncertainty in both SFT and RL rollout banks, e.g. bootstrap both or fit a hierarchical/binomial measurement model.

---

## 5. Why we switched to a new controlled Qwen3 run

The old Qwen2.5 results are scientifically useful but lineage opacity weakens causal attribution to RL. The main experiment therefore needs:

```text
exact base revision
+ exact SFT data revision/subset
+ exact final SFT weights
+ exact RL initialization from those weights
+ exact training recipe
+ intermediate RL checkpoints
+ verifiable binary reward
```

The selected main controlled architecture is:

```text
Qwen/Qwen3-0.6B-Base
    ↓
small reasoning SFT
    ↓
pi_0  (exact saved/fingerprinted final SFT checkpoint)
    ↓
GRPO on GSM8K train
    ↓
pi_t checkpoints
    ↓
held-out GSM8K test + deep fixed-subset evaluation
```

We chose SFT→RL rather than Base→RL because the scientific target is realistic post-training behavior and it aligns with the original question. Qwen3-0.6B is intentionally small so deep rollout banks and checkpoint sweeps are affordable. Qwen3-1.7B can later be a replication if needed; Qwen3.5 is not the first controlled backbone because its newer/hybrid tooling adds unnecessary variables.

Training should happen on a standard CUDA Pod, initially an NVIDIA A40 48 GB environment. The M5 Mac is useful for design, data/scoring work, and smoke tests, but should not be the canonical RLVR training backend.

---

## 6. Frozen controlled-run recipe

The authoritative recipe is in the design + implementation-plan docs. Key invariants:

### SFT

```text
base: Qwen/Qwen3-0.6B-Base
SFT data: deterministic contamination-audited 10k subset of open-r1/OpenR1-Math-220k
trace: one complete verified-correct reasoning trace per problem
formatted max length: 2048 tokens
full-parameter training
2 epochs
bf16
FlashAttention 2
gradient checkpointing
packing
AdamW fused
LR 2e-5
cosine
warmup_ratio 0.03
weight_decay 0.01
per-device batch 8
grad accumulation 8
seed 42
```

The final epoch-2 SFT weights are `pi_0` by definition. GRPO must load this exact local checkpoint and verify its file fingerprint before trainer construction.

### GRPO

```text
data: GSM8K train only
GSM8K test: evaluation-only
one epoch
reward: binary final-answer correctness only
num_generations: 8
temperature: 0.8
top_p: 0.95
top_k: 0
repetition_penalty: 1.0
max completion: 1024
hard prompt preflight: <=512 tokens; never silently truncate
vllm_max_model_length: 1536
LR: 1e-6
cosine
warmup: 0.10
AdamW fused
max_grad_norm: 1.0
bf16
FlashAttention 2
gradient checkpointing
colocated vLLM
beta: 0
epsilon: 0.2
num_iterations: 1
loss_type: dapo
scale_rewards: group
seed: 42
per_device_train_batch_size: 8
gradient_accumulation_steps: 1
generation_batch_size: 8
vLLM IS correction: True
IS mode: sequence_mask
IS cap: 3.0
```

Policy-only snapshots are intended every 5% of GRPO progress, with resumable trainer checkpoints near 25/50/75/100%. Initial scientific analysis uses `pi_0`, `pi_25`, `pi_50`, `pi_75`, `pi_100`.

Canonical evaluation plan:

```text
full GSM8K test: K=8 for five primary policies
fixed 30-question subset: K=256 for five primary policies
same 30-question subset: pi_0 at K=1024
```

Future useful time-series quantity:

```text
beta_i(t) = logit p_i(t) - logit p_i(0)
```

This can distinguish a broad synchronized early tilt from problem-specific redistribution that emerges later.

---

## 7. Current implementation branch and testing environment

Implementation branch:

```text
controlled-qwen3-rlvr
```

It was created from:

```text
difficulty-bin-analysis
```

Local execution constraint encountered during implementation: the available container could not resolve `github.com`, so a normal local clone/worktree + pytest loop was unavailable.

To preserve the Superpowers/TDD RED→GREEN discipline rather than silently skipping tests, a feature-branch-only GitHub Actions workflow was added:

```text
.github/workflows/test-controlled-qwen3.yml
```

It uses CPU PyTorch for the test runner and runs the full pytest suite. This workflow is an implementation aid; it is not the A40 training environment.

Baseline tests were verified clean before controlled-run implementation. Every new production module so far has been preceded by a failing test commit and an observed RED workflow, then a GREEN implementation commit.

---

## 8. Exact implementation status at this checkpoint

### Task 1 — configuration boundary and isolated A40 dependency intent: GREEN

Implemented:

```text
controlled_run/__init__.py
controlled_run/constants.py
controlled_run/config.py
controlled_run/configs/sft_qwen3_0_6b.yaml
controlled_run/configs/grpo_qwen3_0_6b.yaml
controlled_run/requirements-a40.in
.gitignore additions
tests/test_controlled_run_config.py
```

The config validators lock the approved experiment-defining values rather than allowing silent drift.

Relevant implementation commit:

```text
4bb54a302f1e66c27d9c31e54bce24fc71d754d2
Add controlled Qwen3 experiment configuration
```

The CI workflow was subsequently adjusted to install CPU PyTorch so tests do not download multi-GB CUDA wheels on every run.

### Task 2 — immutable provenance and directory fingerprints: GREEN

Implemented:

```text
controlled_run/provenance.py
tests/test_controlled_run_provenance.py
```

Behavior includes:

- resolving model/dataset revisions to immutable Hub SHAs;
- SHA256 file hashing;
- lexical recursive directory fingerprints;
- missing/extra/mismatched file detection;
- stable sorted/indented JSON output;
- optional exclusions so `pi0_manifest.json` does not recursively hash itself.

Implementation commit:

```text
8499cebc8b1b3ed0777ae3ec3f7ec8e48ab2113d
Add controlled-run provenance fingerprints
```

### Task 3 — deterministic SFT subset / contamination logic / materialization CLI: GREEN in unit tests

Implemented pure logic:

```text
controlled_run/data.py
tests/test_controlled_run_data.py
```

Locked behavior includes:

- select the lowest-index complete, verifier-correct OpenR1 reasoning trace;
- basic normalization = NFKC + lowercase + whitespace collapse;
- aggressive normalization additionally removes digit-group commas, maps punctuation to spaces, collapses whitespace;
- near duplicate = Jaccard >=0.80 over aggressive-normalized 5-word shingles;
- inverted shingle index for candidate filtering;
- stable candidate order from SHA256 of `f"{seed}:{uuid}"`, independent of input row order/Python hash randomization;
- skip contaminated or >2048-token formatted candidates and count them in audit;
- deterministic exact target size or fail loudly.

Implementation commit:

```text
f19f5ef40eed07170d9f1225eb7f5b904141109c
Add deterministic controlled-run data selection
```

Materialization CLI implemented:

```text
controlled_run/prepare_data.py
tests/test_prepare_controlled_data.py
```

It resolves immutable SHAs first, then loads the Qwen3 tokenizer, OpenR1, and GSM8K at those pinned revisions, and writes:

```text
data/controlled_run/manifests/sft_10k_manifest.jsonl
data/controlled_run/manifests/contamination_audit.json
data/controlled_run/manifests/source_revisions.json
```

plus ignored full training records:

```text
data/controlled_run/generated/sft_10k_records.jsonl
```

Implementation commit:

```text
ff987532752c0d26bc8a822359f65906740d9967
Add controlled data materialization CLI
```

**Crucial pending point:** the real 10k data materialization has **not yet been executed and audited against the live pinned Hugging Face sources** in this implementation session. Unit tests are green, but do not claim that the real `sft_10k_manifest.jsonl` / contamination audit has already been generated or inspected. Do not commit a real 10k manifest until `prepare_data.py` has run successfully and the audit has been reviewed.

### Task 4 — full SFT + exact pi_0: CURRENTLY IN RED

This is the exact point where implementation stopped for the conversation handoff.

Two test files have been added first, as required by TDD:

```text
tests/test_controlled_run_checkpointing.py
tests/test_controlled_run_sft.py
```

Checkpointing contract currently requires:

- `freeze_pi0(trainer, tokenizer, pi0_dir, lineage)` saves a dedicated exact policy directory;
- manifest contains `policy_name="pi_0"` plus:
  - `base_model_sha`
  - `sft_dataset_sha`
  - `sft_data_manifest_sha256`
  - `sft_config_sha256`
  - exact relative-file SHA256 map;
- `load_pi0_manifest()` re-verifies every tracked file;
- mutated weights fail verification;
- an untracked extra file also fails verification;
- freezing into a non-empty destination raises instead of overwriting stale state.

SFT contract currently requires:

- `build_sft_arguments(config, output_dir)` maps the YAML to the exact canonical SFTConfig kwargs;
- `load_prompt_completion_jsonl(path)` preserves `uuid`, conversational `prompt`, and conversational `completion` columns;
- malformed records missing prompt/completion fail loudly.

Current branch head at the moment this checkpoint was prepared:

```text
2b2b3a2176b8d8da995f5ffb45f341e68c1cd0ca
test: define controlled SFT training contract
```

The preceding Task-4 RED commit is:

```text
b1b2e0966a6da7161e73bc6b491ee40a8d423da7
test: define exact pi0 checkpoint contract
```

Latest feature-branch GitHub Actions run for head `2b2b3a...` is **failing intentionally at the RED stage**. Production implementations for Task 4 have not yet been added. Do not “fix” this by weakening or deleting the tests.

---

## 9. Exact next action in a fresh window

Resume **Task 4, GREEN phase** from the implementation plan.

Implement only the minimal production code required by the already-failing tests:

```text
controlled_run/checkpointing.py
controlled_run/train_sft.py
```

Then run/observe the focused tests and full suite through the available test runner. The expected path is:

```text
current RED
→ implement checkpointing helpers
→ implement exact SFTConfig mapping + JSONL loader
→ GREEN focused tests
→ GREEN full pytest
→ continue the remaining Task 4 training-entry-point / pi_0-freeze steps from the plan
```

Do not skip ahead to GRPO, evaluation, or analysis before Task 4 is green and `pi_0` provenance is nailed down. The entire reason for this controlled run is exact lineage.

After Task 4, continue the written plan task-by-task; use the Superpowers execution/TDD process rather than improvising a second design.

---

## 10. Operational caveats for the next window

1. **Do not restart architecture brainstorming.** The Qwen3 controlled-run design has already been approved and written. Continue the implementation plan unless a real incompatibility forces a design revision.
2. **Do not silently change canonical hyperparameters to fit hardware.** If the A40 pilot OOMs, record it and revise the design/config explicitly.
3. **Do not train on GSM8K test.** It is evaluation-only and must not influence SFT checkpoint selection, GRPO tuning, or checkpoint selection.
4. **Do not let Hub `main` drift into the experiment.** Resolve and record immutable model/dataset SHAs before real materialization/training.
5. **Do not treat the frozen-pool tilt as RL.** It is a null/explanatory control over an existing rollout pool.
6. **Do not overclaim old Qwen2.5 lineage.** Exact PPO start checkpoint remains unverified.
7. **Do not infer literal support expansion from finite sampling.** A softmax LM generally gives nonzero token probabilities; the empirical question is practical reachability under a defined sampling budget.
8. **Persist checkpoints and raw rollout banks off ephemeral Pod storage before destroying a Pod.**
9. **Keep old Qwen2.5 inference defaults working.** The controlled Qwen3 pipeline is a separate named profile, not a replacement that is allowed to break the pilot runtime.

---

## 11. Existing runtime work that is separate from the controlled-run training implementation

Cross-platform vLLM runtime work was already merged previously:

```text
PR #4
merge SHA: 0b5904750c505694d007fc25562d65935c84c608
```

M5 exact SFT smoke passed after the Metal revision workaround. The vLLM-Metal shutdown segfault in `at::accelerator::emptyHostCache()` was independently reproduced as an upstream shutdown defect; worker shutdown/results were complete and shell exit was 0.

Before the science pivot, remaining runtime acceptance work was a fresh RunPod canonical smoke (`top_k=0`, repetition penalty `1`) and a top-k JIT smoke (`top_k=20`, repetition penalty `1.1`) confirming `ninja`, followed by stable-image promotion. Do not claim those remaining RunPod acceptances passed unless there is later evidence.

Prior stable image:

```text
ghcr.io/unitedsnakes/rlvr-vllm:0.27.1
```

Private result backup dataset:

```text
HKReporter/rlvr-behavior-probe-results
```

This runtime thread is secondary to finishing the controlled Qwen3 experiment backbone right now.

---

## 12. Literature/positioning context worth retaining

The MaxRL paper under discussion was *Maximum Likelihood Reinforcement Learning* (arXiv `2602.02710v2`, Aug 19 2026). Relevant framing:

- `p = P(success | x)`;
- `pass@k = 1 - (1-p)^k`;
- RL objective uses `p`, ML objective uses `log p`;
- `log p = -sum_k (1-p)^k/k`, relating the ML gradient to weighted pass@k gradients;
- a frozen SFT rollout pool is an **off-policy learner restricted to a fixed pool**, not the same objective as live on-policy RL.

Nearby evidence already considered:

- Yue et al. (arXiv `2504.13837`, NeurIPS 2025 Oral): small-k RLVR can improve while large-k base may remain higher; reasoning patterns were bounded by base in the studied setting.
- Yuan et al. (arXiv `2606.15455`, June 2026): high-k decline does not by itself rule out boundary expansion; initially unsolvable instances may become solvable.

The controlled experiment should therefore be positioned to measure where amplification/reweighting explains results and where it breaks, rather than begin from a universal “RL cannot create capability” premise.

---

## 13. One-sentence handoff

**Start the next chat by reading this checkpoint + the Qwen3 design + implementation plan, inspect branch `controlled-qwen3-rlvr`, note that Tasks 1–3 are unit-test GREEN but real data materialization is still pending, and resume Task 4 from the intentional RED tests at head `2b2b3a...` by implementing `controlled_run/checkpointing.py` and `controlled_run/train_sft.py` without changing the approved experiment design.**
