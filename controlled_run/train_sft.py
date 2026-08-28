from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from controlled_run.checkpointing import freeze_pi0, load_pi0_manifest
from controlled_run.config import (
    load_config,
    validate_sft_config,
    validate_sft_runtime_batch,
)
from controlled_run.constants import BASE_MODEL
from controlled_run.data_bundle import verify_canonical_sft_bundle
from controlled_run.provenance import sha256_file, write_json


CANONICAL_SFT_RECORDS = 10_000
CANONICAL_SFT_VALIDATION_RECORDS = 512
DEFAULT_CONFIG = Path("controlled_run/configs/sft_qwen3_0_6b.yaml")
DEFAULT_RECORDS = Path("data/controlled_run/generated/sft_10k_records.jsonl")
DEFAULT_VALIDATION_RECORDS = Path("data/controlled_run/generated/sft_val_512_records.jsonl")
DEFAULT_SOURCE_REVISIONS = Path("data/controlled_run/manifests/source_revisions.json")
DEFAULT_SFT_MANIFEST = Path("data/controlled_run/manifests/sft_10k_manifest.jsonl")
DEFAULT_VALIDATION_MANIFEST = Path("data/controlled_run/manifests/sft_val_512_manifest.jsonl")
DEFAULT_OUTPUT_DIR = Path("controlled_run_outputs/sft")


def _load_sft_classes():
    try:
        from trl import SFTConfig, SFTTrainer
    except ImportError as error:
        raise RuntimeError(
            "TRL is required for controlled SFT training. Install the pinned A40 "
            "training runtime before running SFT."
        ) from error
    return SFTConfig, SFTTrainer


def _runtime_world_size() -> int:
    raw = os.environ.get("WORLD_SIZE", "1")
    try:
        world_size = int(raw)
    except ValueError as error:
        raise ValueError(f"WORLD_SIZE must be an integer, got {raw!r}") from error
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    return world_size


def _wait_for_everyone(trainer) -> None:
    accelerator = getattr(trainer, "accelerator", None)
    wait = getattr(accelerator, "wait_for_everyone", None)
    if callable(wait):
        wait()


def _is_world_process_zero(trainer) -> bool:
    check = getattr(trainer, "is_world_process_zero", None)
    if callable(check):
        return bool(check())
    return True


def build_sft_arguments(
    config: dict,
    output_dir: Path,
    max_steps: int | None = None,
):
    validate_sft_config(config)
    SFTConfig, _ = _load_sft_classes()

    canonical = max_steps is None
    kwargs = {
        "output_dir": str(Path(output_dir)),
        "num_train_epochs": config["num_train_epochs"],
        "max_length": config["max_length"],
        "bf16": config["bf16"],
        "gradient_checkpointing": config["gradient_checkpointing"],
        "packing": config["packing"],
        "packing_strategy": config["packing_strategy"],
        "completion_only_loss": config["completion_only_loss"],
        "learning_rate": config["learning_rate"],
        "lr_scheduler_type": config["lr_scheduler_type"],
        "warmup_ratio": config["warmup_ratio"],
        "weight_decay": config["weight_decay"],
        "per_device_train_batch_size": config["per_device_train_batch_size"],
        "per_device_eval_batch_size": config["per_device_train_batch_size"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "optim": config["optim"],
        "save_strategy": "epoch",
        "eval_strategy": "epoch" if canonical else "no",
        "load_best_model_at_end": False,
        "report_to": "none",
        "seed": config["seed"],
        "data_seed": config["seed"],
    }
    if max_steps is not None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive when supplied")
        kwargs["max_steps"] = int(max_steps)

    return SFTConfig(**kwargs)


def load_prompt_completion_jsonl(path: Path) -> Dataset:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Controlled SFT records file does not exist: {source}")

    rows: list[dict] = []
    with source.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {source}"
                ) from error

            if "prompt" not in row or "completion" not in row:
                raise ValueError(
                    f"SFT record line {line_number} must contain prompt and completion"
                )
            rows.append(row)

    if not rows:
        raise ValueError(f"Controlled SFT records file is empty: {source}")

    return Dataset.from_list(rows)


def _record_count(path: Path) -> int:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Controlled SFT records file does not exist: {source}")
    count = sum(
        1
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if count == 0:
        raise ValueError(f"Controlled SFT records file is empty: {source}")
    return count


def validate_record_count(path: Path, canonical: bool) -> int:
    count = _record_count(path)
    if canonical and count != CANONICAL_SFT_RECORDS:
        raise ValueError(
            "Canonical controlled SFT requires exactly 10000 records; "
            f"found {count} in {path}"
        )
    return count


def validate_validation_record_count(path: Path, canonical: bool) -> int:
    count = _record_count(path)
    if canonical and count != CANONICAL_SFT_VALIDATION_RECORDS:
        raise ValueError(
            "Canonical controlled SFT requires exactly 512 validation records; "
            f"found {count} in {path}"
        )
    return count


def _load_source_revisions(path: Path) -> dict:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing source revision manifest: {source}") from None

    for key in ("base_model", "sft_dataset"):
        entry = payload.get(key)
        if not isinstance(entry, dict) or not entry.get("repo_id") or not entry.get("sha"):
            raise ValueError(
                f"Source revision manifest must contain {key}.repo_id and {key}.sha"
            )
    return payload


def build_sft_lineage(
    source_revisions_path: Path,
    sft_manifest_path: Path,
    config_path: Path,
    validation_manifest_path: Path | None = None,
) -> dict:
    revisions = _load_source_revisions(source_revisions_path)
    lineage = {
        "base_model_sha": str(revisions["base_model"]["sha"]),
        "sft_dataset_sha": str(revisions["sft_dataset"]["sha"]),
        "sft_data_manifest_sha256": sha256_file(Path(sft_manifest_path)),
        "sft_config_sha256": sha256_file(Path(config_path)),
    }
    if validation_manifest_path is not None:
        lineage["sft_validation_manifest_sha256"] = sha256_file(
            Path(validation_manifest_path)
        )
    return lineage


def _write_run_manifest(
    destination: Path,
    *,
    mode: str,
    record_count: int,
    validation_record_count: int | None,
    smoke_steps: int | None,
    config: dict,
    lineage: dict,
    runtime_batch: dict,
) -> None:
    write_json(
        destination / "sft_run_manifest.json",
        {
            "mode": mode,
            "record_count": record_count,
            "validation_record_count": validation_record_count,
            "validation_role": "diagnostic_only_no_checkpoint_selection",
            "smoke_steps": smoke_steps,
            "config": config,
            "runtime_batch": runtime_batch,
            "lineage": lineage,
        },
    )


def run_sft(
    *,
    config_path: Path,
    records_path: Path,
    source_revisions_path: Path,
    sft_manifest_path: Path,
    output_dir: Path,
    validation_records_path: Path | None = None,
    validation_manifest_path: Path | None = None,
    smoke_steps: int | None = None,
) -> dict:
    canonical = smoke_steps is None
    config = load_config(Path(config_path))
    validate_sft_config(config)

    if canonical and (validation_records_path is None or validation_manifest_path is None):
        raise ValueError(
            "Canonical SFT requires the deterministic 512-example validation records "
            "and validation manifest"
        )

    if canonical:
        verify_canonical_sft_bundle(
            Path(source_revisions_path).parent,
            Path(records_path).parent,
            expected_max_formatted_tokens=int(config["max_length"]),
            supplied_artifacts={
                "generated/sft_10k_records.jsonl": Path(records_path),
                "generated/sft_val_512_records.jsonl": Path(validation_records_path),
                "manifests/source_revisions.json": Path(source_revisions_path),
                "manifests/sft_10k_manifest.jsonl": Path(sft_manifest_path),
                "manifests/sft_val_512_manifest.jsonl": Path(validation_manifest_path),
            },
        )

    record_count = validate_record_count(records_path, canonical=canonical)

    validation_record_count: int | None = None
    eval_dataset = None
    if validation_records_path is not None:
        validation_record_count = validate_validation_record_count(
            validation_records_path,
            canonical=canonical,
        )
        eval_dataset = load_prompt_completion_jsonl(validation_records_path)

    runtime_batch = validate_sft_runtime_batch(
        config,
        world_size=_runtime_world_size(),
        canonical=canonical,
    )
    revisions = _load_source_revisions(source_revisions_path)

    base_entry = revisions["base_model"]
    if config["model_name"] != base_entry["repo_id"]:
        raise ValueError(
            "SFT config model_name does not match pinned base-model repository: "
            f"{config['model_name']!r} != {base_entry['repo_id']!r}"
        )
    if config["model_name"] != BASE_MODEL:
        raise ValueError(
            f"Controlled SFT must use {BASE_MODEL!r}, got {config['model_name']!r}"
        )

    base_sha = str(base_entry["sha"])
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        revision=base_sha,
        dtype=torch.bfloat16,
        attn_implementation=config["attn_implementation"],
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        revision=base_sha,
    )
    train_dataset = load_prompt_completion_jsonl(records_path)

    destination = Path(output_dir)
    trainer_output = destination / "trainer"
    args = build_sft_arguments(
        config,
        trainer_output,
        max_steps=smoke_steps,
    )
    _, SFTTrainer = _load_sft_classes()
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    _wait_for_everyone(trainer)

    lineage = build_sft_lineage(
        source_revisions_path,
        sft_manifest_path,
        config_path,
        validation_manifest_path=validation_manifest_path,
    )
    mode = "canonical" if canonical else "smoke"
    is_world_zero = _is_world_process_zero(trainer)

    if is_world_zero:
        _write_run_manifest(
            destination,
            mode=mode,
            record_count=record_count,
            validation_record_count=validation_record_count,
            smoke_steps=smoke_steps,
            config=config,
            lineage=lineage,
            runtime_batch=runtime_batch,
        )

        if canonical:
            if "sft_validation_manifest_sha256" not in lineage:
                raise RuntimeError(
                    "Canonical SFT lineage is missing validation manifest SHA256"
                )
            freeze_pi0(
                trainer,
                tokenizer,
                destination / "pi_0",
                lineage,
            )
        else:
            smoke_final = destination / "smoke_final"
            trainer.save_model(str(smoke_final))
            tokenizer.save_pretrained(str(smoke_final))

    _wait_for_everyone(trainer)

    if canonical:
        manifest = load_pi0_manifest(destination / "pi_0")
        return {
            "mode": "canonical",
            "record_count": record_count,
            "validation_record_count": validation_record_count,
            "runtime_batch": runtime_batch,
            "pi0_manifest": manifest,
        }

    return {
        "mode": "smoke",
        "record_count": record_count,
        "validation_record_count": validation_record_count,
        "runtime_batch": runtime_batch,
        "smoke_steps": int(smoke_steps),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run controlled Qwen3 reasoning SFT and freeze exact pi_0."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument(
        "--validation-records",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--source-revisions",
        type=Path,
        default=DEFAULT_SOURCE_REVISIONS,
    )
    parser.add_argument(
        "--sft-manifest",
        type=Path,
        default=DEFAULT_SFT_MANIFEST,
    )
    parser.add_argument(
        "--validation-manifest",
        type=Path,
        default=None,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--smoke-steps", type=int, default=None)
    args = parser.parse_args(argv)

    canonical = args.smoke_steps is None
    validation_records_path = args.validation_records
    validation_manifest_path = args.validation_manifest
    if canonical:
        if validation_records_path is None:
            validation_records_path = DEFAULT_VALIDATION_RECORDS
        if validation_manifest_path is None:
            validation_manifest_path = DEFAULT_VALIDATION_MANIFEST

    result = run_sft(
        config_path=args.config,
        records_path=args.records,
        validation_records_path=validation_records_path,
        source_revisions_path=args.source_revisions,
        sft_manifest_path=args.sft_manifest,
        validation_manifest_path=validation_manifest_path,
        output_dir=args.output_dir,
        smoke_steps=args.smoke_steps,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
