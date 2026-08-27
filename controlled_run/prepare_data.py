from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from controlled_run.constants import BASE_MODEL, GSM8K_DATASET, SEED, SFT_DATASET
from controlled_run.data import build_sft_manifest, materialize_sft_records
from controlled_run.provenance import resolve_hf_revision, write_json


DEFAULT_MANIFESTS_DIR = Path("data/controlled_run/manifests")
DEFAULT_GENERATED_DIR = Path("data/controlled_run/generated")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def prepare_data(
    manifests_dir: Path = DEFAULT_MANIFESTS_DIR,
    generated_dir: Path = DEFAULT_GENERATED_DIR,
    target_size: int = 10_000,
    seed: int = SEED,
) -> dict:
    manifests_dir = Path(manifests_dir)
    generated_dir = Path(generated_dir)

    base_sha = resolve_hf_revision(BASE_MODEL, "main", "model")
    sft_dataset_sha = resolve_hf_revision(SFT_DATASET, "main", "dataset")
    gsm8k_dataset_sha = resolve_hf_revision(GSM8K_DATASET, "main", "dataset")

    source_revisions = {
        "base_model": {
            "repo_id": BASE_MODEL,
            "requested_revision": "main",
            "sha": base_sha,
        },
        "tokenizer": {
            "repo_id": BASE_MODEL,
            "requested_revision": "main",
            "sha": base_sha,
        },
        "sft_dataset": {
            "repo_id": SFT_DATASET,
            "requested_revision": "main",
            "sha": sft_dataset_sha,
        },
        "gsm8k_dataset": {
            "repo_id": GSM8K_DATASET,
            "requested_revision": "main",
            "sha": gsm8k_dataset_sha,
        },
        "seed": seed,
        "target_size": target_size,
    }

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=base_sha)
    openr1_rows = list(
        load_dataset(
            SFT_DATASET,
            "default",
            split="train",
            revision=sft_dataset_sha,
        )
    )
    gsm8k_train = list(
        load_dataset(
            GSM8K_DATASET,
            "main",
            split="train",
            revision=gsm8k_dataset_sha,
        )
    )
    gsm8k_test = list(
        load_dataset(
            GSM8K_DATASET,
            "main",
            split="test",
            revision=gsm8k_dataset_sha,
        )
    )

    manifest, audit = build_sft_manifest(
        openr1_rows,
        [*gsm8k_train, *gsm8k_test],
        tokenizer,
        target_size=target_size,
        seed=seed,
    )
    records = materialize_sft_records(manifest, openr1_rows)

    if len(manifest) != target_size or len(records) != target_size:
        raise RuntimeError(
            "Controlled SFT materialization produced an unexpected number of records: "
            f"manifest={len(manifest)}, records={len(records)}, expected={target_size}"
        )

    _write_jsonl(manifests_dir / "sft_10k_manifest.jsonl", manifest)
    write_json(manifests_dir / "contamination_audit.json", audit)
    write_json(manifests_dir / "source_revisions.json", source_revisions)
    _write_jsonl(generated_dir / "sft_10k_records.jsonl", records)

    return source_revisions


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build the deterministic contamination-audited controlled SFT dataset."
    )
    parser.add_argument(
        "--manifests-dir",
        type=Path,
        default=DEFAULT_MANIFESTS_DIR,
    )
    parser.add_argument(
        "--generated-dir",
        type=Path,
        default=DEFAULT_GENERATED_DIR,
    )
    parser.add_argument("--target-size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    revisions = prepare_data(
        manifests_dir=args.manifests_dir,
        generated_dir=args.generated_dir,
        target_size=args.target_size,
        seed=args.seed,
    )
    print(json.dumps(revisions, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
