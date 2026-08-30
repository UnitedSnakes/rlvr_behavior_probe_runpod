# Controlled Qwen3 RLVR — SFT Termination Repair Checkpoint

## Status

Checkpoint date: 2026-08-30.

Active branch: `controlled-qwen3-rlvr-task6`.

The corrected canonical two-epoch SFT has completed at:

```text
controlled_run_outputs/sft_corrected
```

Produced checkpoints:

```text
trainer/checkpoint-56
trainer/checkpoint-112
pi_0
```

The native-EOS SFT repair has passed teacher-forced endpoint checks and generated-state checks. The remaining 2048-token clipping is now best classified as a **GRPO completion-horizon mismatch**, not evidence that EOS termination is still broken. Do not start canonical GRPO until the completion horizon is re-audited and amended from evidence.

The original frozen pi0 with SHA256
`7ade572b243ddd782f102a6b7ddafd14eecf242c06f2d4fd75e4e99e194c619c`
and all p0/GRPO results derived from it remain diagnostic lineage only.

## Critical bugs and fixes from 2026-08-29 to 2026-08-30

### 1. GRPO pilot exposed severe clipping

The original 2048-token two-A40 GRPO pilot fit in memory but showed roughly 93–95% length clipping and gradient norms around `1e-9`. The run was stopped rather than promoted to canonical GRPO.

### 2. Runtime stop-list hypothesis was ruled out

Direct sampling from the original pi0 showed that the model rarely emitted any natural stop token: in a 32-sample audit, `<|im_end|>` appeared 0 times, native `<|endoftext|>` appeared 5 times, and 27/32 samples hit the cap. Merely adding another runtime stop token could not solve the defect.

### 3. Source-tail and TRL-loss hypotheses were ruled out

The canonical OpenR1 traces usually end almost immediately after the last boxed answer, and TRL 1.12 prompt-completion loss does supervise the assistant terminal token. The issue was not hidden post-answer data tails or masked terminal loss.

### 4. Root cause: assistant terminal-token conflict

Pinned `Qwen/Qwen3-0.6B-Base` already terminates strongly with native EOS `<|endoftext|>` (`151643`). At true SFT endpoints the Base model had median `P(EOT) ≈ 0.9906`, rank 1.

The original chat template instead supervised `<|im_end|>` (`151645`) as the assistant terminal. Cross-entropy rapidly suppressed the already-good native EOT before learning `<|im_end|>` strongly enough:

```text
Base:          EOT median ≈ 0.9906, rank 1
old epoch 1:   EOT median ≈ 0.000355
old epoch 1:   im_end median ≈ 0.000399
```

The model therefore entered OOD continuation after otherwise valid answers.

### 5. Strict one-token A/B confirmed causality

A disposable A/B changed only:

```text
151645 <|im_end|>  ->  151643 <|endoftext|>
```

Generation prompts remained byte-for-byte and token-for-token identical.

At one epoch, native EOS recovered to median probability about `0.9962`, rank 1. On 128 matched GSM8K rollouts, natural stopping improved from 10.16% to 48.44%, clipping fell from 89.84% to 51.56%, and stop+correct rose from 4.69% to 31.25%.

### 6. Production repair

Controlled SFT now patches only the pinned Qwen3 assistant terminal site. System/user formatting and generation prompts remain unchanged. The canonical SFT config fingerprints:

```yaml
assistant_terminal_token: "<|endoftext|>"
assistant_terminal_token_id: 151643
```

The runner verifies these values against the Base tokenizer before constructing `SFTTrainer`. Unit/integration tests and a one-step production-path smoke passed.

### 7. Test-double regressions

Two unrelated SFT tests used minimal fake tokenizers that did not satisfy the new terminal contract. Those tests were isolated from terminal internals while dedicated terminal tests retained coverage. The full suite returned green.

### 8. RunPod Git author identity

A WIP commit initially failed with `Author identity unknown`. `docker/rlvr-bootstrap.sh` now configures repo-local Git identity after sync, with environment overrides, avoiding ad-hoc global setup on fresh Pods.

## Corrected canonical SFT endpoint acceptance

At `checkpoint-56`:

```text
EOT mean:        0.9980149299
EOT median:      0.9998474419
EOT rank median: 1
EOT rank1:       20 / 20
im_end median:   1.3863e-16
```

At final corrected `pi_0`:

```text
EOT mean:        0.9983654410
EOT median:      0.9998703599
EOT rank median: 1
EOT rank1:       20 / 20
im_end median:   1.1307e-16
```

The original teacher-forced termination collapse is repaired through two epochs.

## 2048-token free-rollout diagnostic

An 8-prompt × 16-generation GSM8K audit with the frozen GRPO sampling settings produced:

```text
n:              128
natural stop:    73/128 = 57.03%
length clipped:  55/128 = 42.97%
boxed:           81/128 = 63.28%
correct:         92/128 = 71.88%
stop+correct:    63/128 = 49.22%
clip+correct:    29/128 = 22.66%
length p50:      1561
length p75:      2048
length p90:      2048
```

Among the 55 clipped samples, only 8 contained a boxed answer. Thus most clipping happened before the model reached the standard answer state, not after an answer followed by failure to stop.

A prompt-7 deep diagnostic also showed that clipped-correct cases were mostly genuine `boxed` / `final_phrase` extractions rather than accidental `last_number` matches.

## Generated-state EOT probe

For naturally stopped prompt-7 trajectories, real generated endpoints had native EOT rank 1 with probabilities approximately `0.808`, `0.964`, `0.997`, and `0.995`.

For one clipped trajectory that produced multiple boxes, EOT probability immediately after each box varied with the actual generated context:

```text
box 1: P(EOT)=0.000256, rank 9; model strongly preferred a unit / punctuation continuation
box 2: P(EOT)=0.485886, rank 1; competing token ` pounds` had probability 0.428793
box 3: P(EOT)=0.000065, rank 7; model strongly preferred punctuation / continued reasoning
```

This is consistent with ordinary context-dependent sampling, not a runtime that ignores EOS.

## SFT trajectory-length audit: decisive horizon evidence

The exact frozen 10,000-example SFT bundle was re-tokenized using the corrected tokenizer.

### Full assistant suffix through EOS

```text
p50:  4709.5
p75:  7670
p90: 11093.3
p95: 13176.2
p99: 15545.0

<= 1024:   0.87%
<= 2048:  11.39%
<= 4096:  42.40%
<= 8192:  78.21%
<=12288:  93.25%
<=16384: 100.00%
```

### Through first boxed answer

```text
p50:  4003.5
p75:  6946.75
p90: 10395.3
p95: 12502.95
p99: 14940.6

<= 1024:   3.05%
<= 2048:  20.82%
<= 4096:  51.00%
<= 8192:  81.97%
<=12288:  94.53%
<=16384: 100.00%
```

### Through last boxed answer

```text
p50:  4706.5
p75:  7666
p90: 11091.6
p95: 13173.6
p99: 15537.1

<= 2048: 11.41%
<= 4096: 42.45%
<= 8192: 78.40%
<=12288: 93.27%
```

Only 2/10,000 examples lacked a boxed answer.

### Tail after last box

```text
p50: 3 tokens
p75: 4
p90: 4
p95: 5
p99: 28
```

This is the key reclassification: the SFT data teaches long reasoning trajectories, with median first-box position around 4k tokens and median full completion around 4.7k. A 2048-token GRPO horizon therefore truncates the majority of trajectories before the training distribution normally reaches its answer state. Once the final box is reached, the source distribution ends almost immediately.

## Current gate

Do not change canonical GRPO configuration yet. First measure a small matched horizon curve on the corrected pi0 under the same sampling settings:

```text
max_completion = 2048  -> known clipping 42.97% on the 128-rollout diagnostic
max_completion = 4096  -> measure
max_completion = 8192  -> measure
```

Use sufficient `vllm_max_model_length` only for this diagnostic. The purpose is to choose the GRPO completion horizon from observed pi0 behavior and the frozen SFT trajectory distribution, rather than treating the earlier 2048 value as an invariant.

If 4096/8192 sharply reduce clipping and trajectories terminate normally once they reach the answer state, amend the GRPO recipe accordingly before any new pilot. If clipping remains high even at 8192, investigate a separate long-reasoning / looping behavior before GRPO.
