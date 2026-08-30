# Controlled Qwen3 RLVR — Thinking-Transition Handoff

Checkpoint date: 2026-08-30.

Active branch: `controlled-qwen3-rlvr-task6`.

Do **not** merge and do **not** start canonical GRPO from the current corrected `pi_0` yet.

## Executive status

The original assistant-terminal bug is causally repaired: replacing the SFT assistant terminal from `<|im_end|>` (`151645`) with the Base-native `<|endoftext|>` (`151643`) restored strong teacher-forced EOS behavior.

However, the corrected two-epoch `pi_0` still does not reproduce the intended Qwen3 reasoning grammar reliably. The emerging core issue is no longer EOS termination; it is weak learning of the structural thinking-control tokens `<think>` (`151667`) and `</think>` (`151668`), plus severe free-generation distribution shift from the SFT format.

Current intended SFT grammar is approximately:

```text
<|im_start|>assistant
<think>
reasoning...
</think>

final answer...
<|endoftext|>
```

The current `pi_0` usually does not start generation with `<think>`, and almost never emits `</think>` in the audited rollout batch.

## Canonical corrected SFT

Output:

```text
controlled_run_outputs/sft_corrected
```

Checkpoints:

```text
trainer/checkpoint-56
trainer/checkpoint-112
pi_0
```

The final `pi_0` is weight-identical to checkpoint-112 as expected.

### Native-EOS endpoint acceptance

On 20 true SFT endpoints:

```text
checkpoint-56:
EOT mean:        0.9980149299
EOT median:      0.9998474419
EOT rank1:       20/20
im_end median:   1.3863e-16

final pi_0:
EOT mean:        0.9983654410
EOT median:      0.9998703599
EOT rank1:       20/20
im_end median:   1.1307e-16
```

Conclusion: the original EOT/im_end terminal conflict is fixed.

## 2048-token free-rollout acceptance

8 GSM8K train prompts × 16 generations, frozen GRPO sampling:

```text
n:              128
natural stop:    73/128 = 57.03%
length clipped:  55/128 = 42.97%
boxed:           81/128 = 63.28%
correct:         92/128 = 71.88%
stop+correct:    63/128 = 49.22%
clip+correct:    29/128 = 22.66%
length p50/p75/p90: 1561 / 2048 / 2048
```

This failed the original working rollout gate, but later evidence showed that 2048 is also badly mismatched to the SFT trajectory-length distribution.

## Exact SFT trajectory-length audit

Re-tokenized the exact frozen 10,000 SFT examples with the corrected tokenizer.

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

<= 2048: 20.82%
<= 4096: 51.00%
<= 8192: 81.97%
<=12288: 94.53%
```

### Through last boxed answer

```text
p50: 4706.5
p75: 7666
p90: 11091.6
p95: 13173.6
p99: 15537.1

<= 2048: 11.41%
<= 4096: 42.45%
<= 8192: 78.40%
<=12288: 93.27%
```

### Tail after last box

```text
p50: 3 tokens
p75: 4
p90: 4
p95: 5
p99: 28
```

Only 2/10,000 examples lacked a box. This proves that 2048 is too short for the demonstrated reasoning distribution, but does not by itself explain the later 8k behavior.

## Matched horizon curve

One 8192-token batch was generated and the exact same trajectories were virtually truncated at 2048/4096 for nested comparison.

```text
cap=2048
  stop:         73/128 = 57.03%
  clipped:      55/128 = 42.97%
  boxed:        82/128 = 64.06%
  correct:      86/128 = 67.19%
  stop+correct: 64/128 = 50.00%

cap=4096
  stop:         82/128 = 64.06%
  clipped:      46/128 = 35.94%
  boxed:        91/128 = 71.09%
  correct:      85/128 = 66.41%
  stop+correct: 68/128 = 53.12%

cap=8192
  stop:         84/128 = 65.62%
  clipped:      44/128 = 34.38%
  boxed:        92/128 = 71.88%
  correct:      87/128 = 67.97%
  stop+correct: 68/128 = 53.12%
```

Of the 55 trajectories clipped at 2048, only 9 naturally stopped by 4096 and only 11 by 8192. Therefore simply increasing the completion horizon does not solve the long-generation population.

Raw diagnostic file path on RunPod:

```text
controlled_run_outputs/sft_corrected/horizon_curve_2048_4096_8192.json
```

## 8192-clipped trajectory analysis

44/128 trajectories remained clipped at 8192.

Initial text-based classification incorrectly treated absence of visible `</think>` as proof of no `</think>`; vLLM skips special tokens in text by default. Token-id analysis corrected this.

Robust repetition evidence on the 44 clipped trajectories:

```text
restart >= 3 markers: 44/44
repeat 32-token ngram >= 4 times: 42/44
repeat 64-token ngram >= 3 times: 42/44
```

Restart-marker totals across the 44:

```text
maybe:            3482
alternatively:    1606
wait:             1534
let_me:           1465
another_approach: 230
check_again:      127
however:          10
actually:          8
```

Many trajectories repeated a correct strict answer multiple times before continuing. Examples included 54, 94, 136, 216, 259, 332, and even 716 correct strict-answer events in one trajectory, indicating severe self-restart/repetition modes rather than merely slightly-too-short reasoning.

Raw analysis path:

```text
controlled_run_outputs/sft_corrected/analysis_8192_clipped.json
```

## Canonical SFT think-tag structure audit

Raw SFT completion content already contains explicit `<think>` and `</think>` markers in all 10,000 examples.

Rendered training example inspection confirmed the long reasoning is actually inside `<think>...</think>`; there is no broad train/inference formatting accident where reasoning was placed outside an empty think block.

Structural audit:

```text
(num_start, num_end):
  (1,1): 9999
  (2,1):    1

multi-start:       1/10000
multi-end:         0/10000
open think at EOS: 1/10000
```

The sole malformed row is source index 4800 in the frozen records. It has:

```text
<think> at token 0
</think> at token 9358
second <think> at token 9360
EOT at token 9627
```

This single malformed trace is too rare to explain the population-level failure.

## SFT `</think>` position distribution

Ignoring the single malformed example:

```text
position through </think>:
p50:  4150
p75:  7106.5
p90: 10552.4
p95: 12619.6
p99: 15046.1

<= 2048: 19.65%
<= 4096: 49.24%
<= 8192: 81.14%
<=12288: 94.31%
<=16384: 100.00%
```

After `</think>`, the final-answer phase is much shorter:

```text
tokens from </think> through EOS:
p50: 525
p75: 634
p90: 736
p95: 797
p99: 944
<=2048: 100%
```

Thus the demonstrated grammar has a long think phase followed by a short final phase.

## Reward-vs-thinking audit

Current `controlled_run/rewards.py::gsm8k_binary_reward` scores numeric correctness only. It does not require a valid think/final phase structure.

On the 8192 batch, using token-id presence of `</think>`:

```text
('clipped', 'correct', 'open') = 19
('clipped', 'wrong',   'closed') = 1
('clipped', 'wrong',   'open') = 24
('stopped', 'correct', 'closed') = 1
('stopped', 'correct', 'open') = 67
('stopped', 'wrong',   'open') = 16
```

86/128 trajectories received reward=1 while having no `</think>` token in their generated continuation.

Extraction methods among those 86:

```text
boxed:        73
final_phrase: 10
last_number:   3
```

Important nuance discovered later: most generated trajectories also do not contain `<think>` at all, so the label `open` here means "no generated `</think>`", not necessarily a valid opened think block.

This still means the current correctness-only reward does not enforce the demonstrated reasoning grammar and could reward structurally off-distribution outputs if used for GRPO.

## Stop-reason inspection

Naturally stopped outputs with no generated `</think>` are genuinely ending on native EOT. Example pattern:

```text
... Thus, the total number ... is \boxed{72}.<|endoftext|>
```

vLLM reported `finish_reason='stop'`, `stop_reason=None`, and the final token id was `151643`.

Therefore these stops are not a logging artifact; the model often goes directly from ordinary reasoning/final-answer text to EOT without emitting the expected think/final boundary.

## TRL label supervision audit

Using the actual pinned TRL SFT preprocessing path with `completion_only_loss=True` and `packing=True`, a representative real row showed:

```text
<think>   position 171  label=151667  SUPERVISED=True
</think>  position 952  label=151668  SUPERVISED=True
EOT       position 1155 label=151643  SUPERVISED=True
```

Thus `</think>` is not being masked out by completion-only loss or BFD packing. The phase-boundary problem is not explained by TRL label masking.

## Teacher-forced thinking-control token probes

A correct full-conversation teacher-forced probe on 20 fixed SFT rows gave the following final `pi_0` behavior at true `</think>` targets:

```text
P(</think>) mean:   0.00370245
P(</think>) median: 0.00376895
median rank:        8
rank1:              0/20
```

For comparison, the native-EOS endpoint is ~0.9999 median probability and rank1 on all 20, because Base already had a very strong EOT prior.

Training-trajectory probe on the same 20 rows:

### Base

```text
<think> median P:   5.733e-08
<think> median rank: 69922.5

</think> median P:  4.027e-06
</think> median rank: 14435
```

### Epoch 1 / checkpoint-56

```text
<think> median P:   0.00254299
<think> median rank: 14

</think> median P:  0.00228220
</think> median rank: 17
```

### Epoch 2 / checkpoint-112 / final pi0

```text
<think> median P:   0.00425930
<think> median rank: 6

</think> median P:  0.00376895
</think> median rank: 8
```

Both thinking-control tokens improve by orders of magnitude from Base, but remain low-probability structural choices after two epochs. This is not a `</think>`-only defect; `<think>` is also weakly learned.

## Actual free-generation first-token audit

On the 128 matched 8192 trajectories:

```text
first generated token == <think>: 2/128
contains <think> anywhere:         3/128
contains </think> anywhere:        2/128
```

Think-structure counts:

```text
(no_start, no_end, stop): 80
(no_start, no_end, clip): 43
(start,    no_end, stop): 3
(no_start, end,    stop): 1
(no_start, end,    clip): 1
```

Most first tokens were low-probability-looking multilingual / garbage tokens rather than `<think>` or normal English reasoning tokens. Examples among the most common first tokens included unrelated Arabic fragments, odd Unicode/subword fragments, and only two `<think>` tokens.

This shows the free-generation path is already substantially off the demonstrated SFT grammar at token 1 for most trajectories.

## One-token forced-`<think>` intervention — partial result

A diagnostic currently running appends exactly:

```text
<think>\n
```

to the generation prompt before sampling. No model weights, reward, or sampling hyperparameters are changed.

Partial results available so far:

```text
prompt 0: stop= 9/16, close_think=0/16
prompt 1: stop=10/16, close_think=0/16
prompt 2: stop=11/16, close_think=0/16
prompt 3: stop=12/16, close_think=0/16

aggregate first 4 prompts:
stop:        42/64
close_think:  0/64
```

This is preliminary because prompts 4–7 are still running as of this checkpoint.

Even the partial result already argues strongly against the hypothesis that failure to sample `<think>` at token 1 is the sole cause of the missing `</think>` transition: forcing entry into the demonstrated think phase still produced 0/64 closes in the first half of the batch.

Raw forced-intervention output is intended to be saved at:

```text
controlled_run_outputs/sft_corrected/forced_think_start_8192.json
```

when the run completes.

## Hypotheses ruled out or substantially weakened

1. **Runtime stop list ignores the correct terminal** — ruled out. Natural stops end on token `151643`.
2. **Native EOS itself is broken after SFT** — ruled out at true endpoints; EOT is extremely strong.
3. **Long post-answer tails in SFT source** — ruled out; last-box-to-EOS tail is tiny.
4. **TRL masks the assistant terminal** — ruled out.
5. **TRL/BFD packing masks `</think>`** — ruled out by actual label inspection.
6. **Broad malformed `<think>` data** — ruled out; only 1/10,000 structurally abnormal row.
7. **Train formatting puts reasoning outside think block** — ruled out; raw completion and rendered training both contain the long reasoning inside `<think>...</think>`.
8. **2048 horizon alone explains clipping** — ruled out as sole explanation; 4096→8192 adds only two additional natural stops.
9. **Failure to sample `<think>` at token 1 alone explains missing `</think>`** — strongly weakened by forced-`<think>` partial 0/64 close result.

## Current interpretation

The corrected SFT successfully learns content/reasoning behavior and native EOT, but it does not robustly learn the Qwen3 thinking-control grammar. Both `<think>` and `</think>` remain low-probability structural tokens under teacher forcing, and free generation commonly bypasses them entirely. A substantial subset then enters long self-restart/repetition modes.

This may reflect a mismatch between:

- a 0.6B Base model with essentially no prior for the Qwen3 thinking-control tokens;
- only 10,000 very long traces;
- each structural boundary token appearing only once per example;
- ordinary token-level CE giving those phase-boundary events very little aggregate mass relative to thousands of reasoning tokens.

This mechanism is a hypothesis, not yet causally proven.

## Current gate / next decisions

Do not start GRPO from this `pi_0` yet.

Before another expensive canonical SFT rerun, choose a minimal causal intervention and test it cheaply. Candidate directions for discussion/experimentation include:

1. Remove explicit `<think>` / `</think>` structural dependence and train a simpler completion grammar.
2. Keep the grammar but increase structural-token supervision / weighting.
3. Shorten or otherwise reshape reasoning traces for a 0.6B Base model.
4. Use a model with a stronger prior/capacity for the intended reasoning grammar.
5. Keep correctness-only GRPO only if deliberately accepting structurally off-format outputs; otherwise define a format-validity condition/reward as an explicit objective change rather than silently folding it into correctness.

Do not blindly add more SFT epochs, increase GRPO cap to 16k/32k, or add format reward without first deciding which scientific objective is intended.
