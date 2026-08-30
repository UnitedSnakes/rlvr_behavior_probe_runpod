# Controlled Qwen3 SFT Native-EOS Termination Amendment

## Problem

The controlled SFT lineage starts from `Qwen/Qwen3-0.6B-Base`.

The pinned base model uses:

- native EOS token: `<|endoftext|>`
- native EOS token id: `151643`

The inherited chat template terminates assistant turns with:

- `<|im_end|>`
- token id `151645`

The original controlled SFT therefore supervised `<|im_end|>` as the assistant
terminal token while the Base model had already learned strong native
`<|endoftext|>` termination.

## Diagnostic evidence

At true OpenR1 completion endpoints:

- Base median `P(<|endoftext|>) = 0.990576`, median rank 1.
- Original one-epoch SFT median `P(<|endoftext|>) = 0.000355`.
- Original one-epoch SFT median `P(<|im_end|>) = 0.000399`.

A single-variable intervention changed only the assistant terminal token from
`<|im_end|>` to the Base-native `<|endoftext|>`.

At one epoch:

- native-EOS variant median `P(<|endoftext|>) = 0.996234`, median rank 1.

On 128 matched GSM8K rollouts at one epoch:

| metric | original terminal | native EOS |
|---|---:|---:|
| natural stop | 10.16% | 48.44% |
| length clipped | 89.84% | 51.56% |
| correct | 39.84% | 44.53% |
| stop + correct | 4.69% | 31.25% |
| clipped + correct | 35.16% | 13.28% |

This identifies the assistant-terminal mismatch as a causal defect in the
original SFT recipe.

## Amendment

For controlled SFT only:

1. Preserve the pinned Qwen3 Base tokenizer and vocabulary.
2. Preserve system/user chat formatting.
3. Preserve the generation prompt byte-for-byte and token-for-token.
4. Replace only the final assistant-turn terminal token:
   `<|im_end|>` -> `<|endoftext|>`.
5. Require the pinned Base tokenizer to report:
   - `eos_token == "<|endoftext|>"`
   - `eos_token_id == 151643`
6. Record the terminal token and id in the canonical SFT config so the SFT
   config fingerprint commits to the termination semantics.
7. Do not add special stop-token weighting.
8. Do not change dataset selection, sequence filtering, packing, optimizer,
   learning rate, batch size, epochs, or any other SFT hyperparameter.
9. The previous frozen pi0 and all downstream p0/GRPO runs derived from it are
   diagnostic lineage only and must not be mixed with the corrected lineage.

## Acceptance

Before corrected canonical SFT:

- automated tests must verify that the generation prompt is unchanged;
- automated tests must verify that a completed assistant sequence differs by
  exactly one token at its terminal position:
  `151645 -> 151643`;
- the full test suite must pass.

The corrected canonical two-epoch SFT is then run once from the untouched
pinned Base model. Its epoch-1 and epoch-2 checkpoints are used for termination
acceptance diagnostics; no separate full-length diagnostic SFT is required.

## Corrected canonical endpoint result

The corrected two-epoch run completed under
`controlled_run_outputs/sft_corrected` with checkpoints `checkpoint-56`,
`checkpoint-112`, and final frozen `pi_0`.

Teacher-forced endpoint acceptance on 20 canonical SFT examples passed at both
checkpoints:

| checkpoint | EOT mean | EOT median | EOT rank-1 | im_end median |
|---|---:|---:|---:|---:|
| epoch 1 / checkpoint-56 | 0.99801493 | 0.99984744 | 20/20 | 1.3863e-16 |
| epoch 2 / final pi0 | 0.99836544 | 0.99987036 | 20/20 | 1.1307e-16 |

This confirms that the original teacher-forced termination collapse is repaired
and remains repaired through the full two-epoch canonical SFT.

The final acceptance gate is still a 128-sample free-rollout check on the
corrected `pi_0` with the frozen GRPO sampling parameters. GRPO must not start
until that check shows acceptable natural stopping / clipping behavior.
