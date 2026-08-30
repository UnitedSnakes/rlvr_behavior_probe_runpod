# Controlled Qwen3 RLVR — Canonical SFT Data Bundle Freeze

## Status

Frozen from the live A40 materialization on 2026-08-28 after the 16,384-token cutoff was selected from the extended tokenizer audit.

The materialization passed the portable bundle verifier and the controlled test suite at the then-current branch state (`131 passed, 14 warnings`).

## Scientific selection state

- base model: `Qwen/Qwen3-0.6B-Base`
- base model/tokenizer SHA: `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`
- SFT dataset: `open-r1/OpenR1-Math-220k`
- SFT dataset SHA: `e4e141ec9dea9f8326f4d347be56105859b2bd68`
- GSM8K dataset: `openai/gsm8k`
- GSM8K dataset SHA: `740312add88f781978c0658806c59bc2815b9866`
- source identity: `pinned_dataset_revision_plus_source_index`
- seed: `42`
- train size: `10000`
- validation size: `512`
- canonical formatted-token cutoff: `16384`

## Final audit counts

- `max_formatted_tokens`: `16384`
- `removed_too_long`: `1059`
- `eligible_after_filters`: `63909`
- `selected_total_count`: `10512`
- `train_count`: `10000`
- `validation_count`: `512`
- `removed_exact_duplicates`: `0`
- `removed_normalized_duplicates`: `0`
- `removed_near_duplicates`: `0`

The zero contamination-removal result was separately sanity-checked against all 8,792 GSM8K train+test references: exact/basic/aggressive reference sets each contained all 8,792 questions, OpenR1 produced zero real hits, and a known GSM8K positive was detected by the exact/normalized/near-duplicate machinery.

## Frozen artifact fingerprints

| Artifact | Bytes | SHA256 |
|---|---:|---|
| `generated/sft_10k_records.jsonl` | 171290329 | `886cfaec59674b3f37a51cd34f508981499c4815241dba9a8e380a83fa2707e8` |
| `generated/sft_val_512_records.jsonl` | 8249822 | `e7561133f680a4d0c481231e9bd5e15dea0a8b7838136d2f752a55167825f5da` |
| `manifests/contamination_audit.json` | 1089 | `b69af66c8e0ab5c080a47ac2b85a211b649e26f6f7bbae359a39acaaebf05ba1` |
| `manifests/sft_10k_manifest.jsonl` | 3240972 | `14c0076db8a75b246523b1142ffc6b600309b9a85d655af754ab63cc639a3838` |
| `manifests/sft_val_512_manifest.jsonl` | 166020 | `5d77f3ebe60b2e0d1d4c65ba0cc33ce39e0181ba78b179ea964b8b19f2e76778` |
| `manifests/source_revisions.json` | 732 | `903c1ac5a4aa6e9f529a16e61d4b4c6ed856898190ad27ed09adafdea1027f9e` |

## Gate semantics

Canonical SFT must pass `verify_canonical_sft_bundle` before model loading. The gate verifies bundle hashes/counts, rejects `audit_only=true`, requires the audit cutoff to equal the SFT config cutoff, checks selected train+validation count, and binds the exact supplied trainer artifacts to the bundle SHA256 values.

Any regenerated or edited artifact creates a different bundle and must not be silently treated as this frozen canonical dataset.

## Next infrastructure gate

Do not start canonical SFT until the A40 runtime passes the explicit `flash_attention_2` acceptance check and a one-A40 16k engineering smoke completes successfully. Canonical SFT then uses two A40 GPUs with global optimizer batch 64.
