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

Teacher-forced endpoint acceptance has passed. The 128-sample free-rollout termination acceptance **did not pass**: natural stopping was 57.03% and length clipping remained 42.97%. **Do not start GRPO until the remaining exposure/termination failure is understood.**

The original frozen pi0 with SHA256
`7ade572b243ddd782f102a6b7ddafd14eecf242c06f2d4fd75e4e99e194c619c`
and all p0/GRPO results derived from it are diagnostic lineage only.

## Critical bugs and fixes from 2026-08-29 to 2026-08-30

### 1. GRPO pilot exposed severe completion clipping

The 2048-token two-A40 GRPO pilot fit in memory but showed roughly 93–95% length clipping and gradient norms around `1e-9`. The run was stopped rather than promoted to canonical GRPO.

Increasing the completion cap was explicitly rejected as a fix because it would hide rather than explain the termination failure.

### 2. Runtime stop-list hypothesis was ruled out

Direct vLLM sampling from the original pi0 showed that the model rarely emitted any natural stop token at all: in a 32-sample audit, `<|im_end|>` appeared 0 times, native `<|endoftext|>` appeared 5 times, and 27/32 samples hit the length cap.

Therefore simply adding `<|im_end|>` to the runtime stop list could not solve the problem.

### 3. Long post-answer source tails and TRL loss masking were ruled out

The canonical OpenR1 source traces usually ended shortly after the final boxed answer; the tail after the last boxed answer was tiny for almost all examples. TRL 1.12 prompt-completion masking was also checked and the assistant terminal token was supervised.

These checks shifted the investigation away from data tails and masking.

### 4. Root cause: assistant terminal-token conflict

Pinned `Qwen/Qwen3-0.6B-Base` already terminated extremely strongly with native EOS `<|endoftext|>` (`151643`). At true SFT endpoints the Base model had median `P(EOT) ≈ 0.9906`, rank 1.

The original chat template instead supervised `<|im_end|>` (`151645`) at the assistant endpoint. Cross-entropy training rapidly suppressed the already-good native EOT before it learned `<|im_end|>` strongly enough:

```text
Base:          EOT median ≈ 0.9906, rank 1
old epoch 1:   EOT median ≈ 0.000355
old epoch 1:   im_end median ≈ 0.000399
```

The model therefore entered an OOD continuation regime after producing a correct answer and commonly ran to the generation cap.

### 5. Strict one-token A/B confirmed causality

A disposable A/B changed only the final assistant terminal token:

```text
151645 <|im_end|>  ->  151643 <|endoftext|>
```

Generation prompts stayed byte-for-byte and token-for-token identical.

At one epoch, the native-EOS variant preserved EOT with median probability about `0.9962`, rank 1. On 128 matched GSM8K rollouts, natural stopping improved from 10.16% to 48.44%, clipping fell from 89.84% to 51.56%, and stop+correct rose from 4.69% to 31.25%.

This established the terminal-token mismatch as a causal defect.

### 6. Production repair

The controlled SFT runner now patches only the pinned Qwen3 assistant-terminal site in the chat template. System/user formatting and the generation prompt are preserved.

The canonical SFT config now fingerprints:

```yaml
assistant_terminal_token: "<|endoftext|>"
assistant_terminal_token_id: 151643
```

The runner verifies that these values equal the Base tokenizer's native EOS and applies the patch before constructing `SFTTrainer`.

Tests cover the pinned template branch, EOS guards, and integration ordering. A production-path one-step smoke also confirmed that a saved/reloaded tokenizer renders an assistant completion ending in `<|endoftext|>` while user/system boundaries still use `<|im_end|>`.

### 7. Test-double regressions after the new tokenizer contract

Two older SFT tests used minimal fake tokenizers without `eos_token`, `eos_token_id`, or a real chat template. Once production correctly enforced the native-EOS contract, those unrelated tests failed.

They were fixed by isolating their original concerns and mocking `configure_sft_tokenizer_terminal` as an identity in those tests. A separate integration test remains responsible for verifying that terminal configuration happens before `SFTTrainer` construction.

The full controlled test suite was green after these fixes.

### 8. RunPod Git author identity missing

A WIP commit initially failed with `Author identity unknown` in the RunPod container. `docker/rlvr-bootstrap.sh` was updated to configure repo-local Git identity after repository sync, with `RLVR_GIT_USER_NAME` and `RLVR_GIT_USER_EMAIL` overrides and safe defaults. This avoids requiring ad-hoc global Git config on fresh Pods.

## Corrected canonical SFT endpoint acceptance

At `checkpoint-56`:

```text
EOT mean:        0.9980149299
EOT median:      0.9998474419
EOT rank median: 1
EOT rank1:       20 / 20
im_end median:   1.3863e-16
```

At final corrected `pi_0` (epoch 2):

```text
EOT mean:        0.9983654410
EOT median:      0.9998703599
EOT rank median: 1
EOT rank1:       20 / 20
im_end median:   1.1307e-16
```

Thus the original teacher-forced termination collapse is repaired and remains repaired through two epochs.

## Corrected final pi0 free-rollout acceptance

A 128-rollout audit used 8 GSM8K train prompts × 16 generations with the frozen GRPO sampling parameters (`temperature=0.8`, `top_p=0.95`, `top_k=0`, repetition penalty 1.0, max completion 2048, seed 42).

Per-prompt natural-stop counts were:

```text
[0] 12/16
[1]  9/16
[2]  8/16
[3] 13/16
[4]  9/16
[5] 11/16
[6] 11/16
[7]  0/16
```

Aggregate result:

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

This is materially better than the one-epoch native-EOS diagnostic (`48.44%` natural stop, `51.56%` clipping), but it remains far outside the working acceptance target of >80% natural stop / <20% clipping. The terminal-token repair therefore solved one causal defect without fully solving free-generation termination.

The unusually concentrated failure on prompt 7 (`0/16` natural stop, `5/16` correct) is a high-priority diagnostic target. The next step is to inspect clipped trajectories, especially prompt 7, for repeated reasoning loops, post-answer continuation, answer-without-EOT behavior, or other exposure-shift patterns. Do not start GRPO, silently raise the generation cap, or mask this failure as acceptable.

## Current gate

The next diagnostic should classify the 55 clipped completions into at least:

```text
answer produced, then continued
no answer produced by cap
repetition / looping
other long-but-progressing reasoning
```

and inspect whether native-EOT probability is high immediately after model-generated boxed answers. This will distinguish a failure to reach an answer state from a failure to terminate after reaching one.
