# Controlled Qwen3 p0 Sampling Design

## Status

Approved in chat on 2026-08-29 after discovering that the historical `run_probe.py` uses the Qwen2.5-Instruct tokenizer and an older system prompt, so its 30x16 run is an inference sanity check only and must not enter canonical p0 analysis.

## Goal

Measure pre-RL correctness/reachability probabilities from the exact frozen `pi_0` under the same sampling policy and prompt construction used by canonical GRPO, while preserving lineage and making later `pi_t` evaluation reusable.

## Scientific invariants

- `pi_0` is the exact frozen SFT checkpoint and must be verified from `pi0_manifest.json` before sampling.
- The tokenizer must be loaded from the evaluated policy directory itself; no historical Qwen2.5 tokenizer is allowed.
- GSM8K prompts must come from the same `controlled_run.data.build_gsm8k_rl_rows()` path used by GRPO.
- The system prompt therefore comes from `controlled_run.constants.CONTROLLED_SYSTEM_PROMPT` through the shared row builder; no duplicate prompt string is introduced in the sampler.
- Sampling parameters for canonical p0 match frozen GRPO generation semantics: `num_generations=16`, `temperature=0.8`, `top_p=0.95`, `top_k=0`, `repetition_penalty=1.0`, `max_completion_length=1024`, `seed=42`.
- Prompt length uses the same hard preflight limit as GRPO: 512 tokens, no truncation.
- Correctness scoring uses the same GSM8K final-answer semantics as the controlled reward path.
- The historical 2026-08-29 30x16 `run_probe.py` result is explicitly non-canonical because it used the old prompt/tokenizer path.

## Architecture

Create a new controlled-run entry point, `controlled_run/sample_p0.py`. Do not modify or repurpose `run_probe.py`; it remains historical Qwen2.5 infrastructure.

The new runner has four responsibilities:

1. Verify policy lineage from `pi0_manifest.json` and record its lineage ID.
2. Load the policy-local tokenizer and construct GSM8K rows through `build_gsm8k_rl_rows()`.
3. Perform canonical vLLM sampling with frozen GRPO generation settings.
4. Write raw per-question rollouts plus a run manifest containing policy identity, dataset revision, prompt audit, sampling settings, and shard metadata.

Later `pi_t` evaluation should reuse the same sampling/evaluation core rather than create another checkpoint-specific probe path.

## CLI

The first implementation exposes:

```text
python -m controlled_run.sample_p0 \
  --policy-dir PATH \
  --output-dir PATH \
  [--start-index N] \
  [--end-index N] \
  [--gpu-memory-utilization F]
```

Defaults:

```text
policy-dir: required
output-dir: controlled_run_outputs/p0
start-index: 0
end-index: full GSM8K test split
vLLM gpu memory utilization: 0.50
```

`start-index` is inclusive and `end-index` is exclusive. Shards must be deterministic contiguous slices of the canonical GSM8K test rows. This enables independent multi-A40 sampling without tensor parallelism.

The scientific sampling configuration is not exposed as ordinary CLI knobs in canonical mode. It is read from the frozen controlled GRPO config so p0 cannot silently drift from GRPO.

## Dataset and prompt construction

Use the pinned `openai/gsm8k` dataset revision resolved through the existing provenance helper. For p0/evaluation use the held-out `test` split, not the GRPO training split.

Rows are created by `build_gsm8k_rl_rows()`. Run `assert_prompt_token_limit(..., max_tokens=512)` before any model generation. Any over-limit prompt fails closed rather than being truncated.

For each row, convert the prompt messages to token IDs with the policy-local tokenizer using:

```python
tokenizer.apply_chat_template(
    row["prompt"],
    tokenize=True,
    add_generation_prompt=True,
)
```

The token IDs, not a separately reconstructed string template, are supplied to vLLM.

## Sampling

One vLLM process runs on one GPU. Do not use tensor parallelism for this 0.6B policy.

Canonical sampling parameters are derived from `controlled_run/configs/grpo_qwen3_0_6b.yaml`:

```text
n = num_generations = 16
temperature = 0.8
top_p = 0.95
top_k = 0
repetition_penalty = 1.0
max_tokens = max_completion_length = 1024
```

Question-level seeds must be deterministic and shard-independent. Use a seed derived only from the frozen experiment seed and the original GSM8K row index, so running one full job or four independent shards produces the same per-question sampling seeds.

## Scoring

Each generated completion is scored for GSM8K final-answer correctness using the same extraction/equality semantics used by the controlled binary reward path. The raw completion text is always preserved.

Each per-question record contains at least:

```json
{
  "dataset_index": 0,
  "question_seed": 4200000,
  "question": "...",
  "gold": "...",
  "n_correct": 9,
  "n_rollouts": 16,
  "rollouts": [
    {
      "rollout": 0,
      "correct": true,
      "text": "..."
    }
  ]
}
```

Additional extraction metadata may be recorded if it comes directly from the shared scoring implementation.

## Outputs

Each shard writes a self-contained directory:

```text
OUTPUT/
  p0_raw.jsonl
  p0_run_manifest.json
  prompt_length_audit.json
```

`p0_run_manifest.json` records:

- mode: `canonical_p0`
- policy directory identity and `pi0_lineage_id`
- full `pi0_manifest`
- GSM8K resolved dataset SHA and split
- GRPO config SHA256 and the exact sampling fields copied from it
- prompt-length audit
- shard start/end indices and record count
- runtime metadata sufficient to identify vLLM/CUDA/GPU

The runner must write local results incrementally by question so a long shard can be inspected and resumed in a future extension, but the initial CLI does not need automatic resume support unless implementation complexity is negligible.

## Multi-GPU deep p0

Deep p0 uses independent one-GPU shards, for example:

```text
GPU0: [0, 330)
GPU1: [330, 660)
GPU2: [660, 990)
GPU3: [990, 1319)
```

Each shard uses the same policy, dataset revision, GRPO config, prompt construction, and per-question seed derivation. Shard outputs are merged only after manifest compatibility checks. Tensor parallelism is explicitly not the default strategy.

## Acceptance tests

Unit tests must establish that:

1. canonical sampling fields are copied from the GRPO config and not hard-coded to a second independent recipe;
2. the policy-local tokenizer is the tokenizer passed to prompt construction/model loading;
3. prompt rows are built through `build_gsm8k_rl_rows()` and the 512-token preflight is enforced;
4. shard slicing is deterministic, inclusive/exclusive, and bounds-checked;
5. question seeds depend on original dataset index rather than shard-local position;
6. lineage verification happens before sampling;
7. manifest output records the exact GRPO-derived sampling configuration and dataset/policy provenance;
8. the historical `probe/prompts.py` constants are not imported by the new controlled sampler.

A real A40 smoke test then runs a tiny held-out shard and must show successful vLLM generation from the exact frozen `pi_0` before any result is called canonical p0.

## Non-goals

- Do not change GRPO scientific hyperparameters.
- Do not change the historical Qwen2.5 probe pipeline.
- Do not use tensor parallelism for the 0.6B p0 runner.
- Do not treat the prior 30x16 historical-probe output as canonical evidence.
- Do not merge PR #5 as part of this work.
