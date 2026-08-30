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

Teacher-forced endpoint acceptance has passed. Free-rollout termination acceptance is still pending. **Do not start GRPO until the free-rollout gate passes.**

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

## Current gate

Run a 128-sample free-rollout acceptance on the corrected final `pi_0` with the frozen GRPO sampling parameters and record vLLM `finish_reason`, token length, boxed-answer presence, and correctness.

Working acceptance target:

```text
natural stop: preferably >80%
length clipping: preferably <20%
correctness: no material regression relative to the matched diagnostic baseline
```

If clipping remains around 30–50%, the native-EOS bug is fixed but a second exposure/termination issue remains and must be investigated before GRPO. Do not mask truncated samples, silently raise the generation cap, or promote this pi0 to canonical GRPO without the rollout result.
