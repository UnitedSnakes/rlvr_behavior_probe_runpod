from __future__ import annotations

import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from controlled_run.checkpointing import freeze_pi0
from controlled_run.config import load_config, validate_sft_config
from controlled_run.constants import BASE_MODEL
from controlled_run.provenance import sha256_file


CANONICAL_SFT_RECORDS = 10_000


def _load_sft_classes():
    try:
        from trl import SFTConfig, SFTTrainer
    except ImportError as error:
        raise RuntimeError(
            "TRL is required for controlled SFT training. Install "
            "controlled_run/requirements-a40.in in the isolated A40 environment."
        ) from error
    return SFTConfig, SFTTrainer


def build_sft_arguments(
    config: dict,
    output_dir: Path,
    max_steps: int | None = None,
):
    validate_sft_config(config)
    SFTConfig, _ = _load_sft_classes()

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
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "optim": config["optim"],
        "save_strategy": "epoch",
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


def validate_record_count(path: Path, canonical: bool) -> int:
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
    if canonical and count != CANONICAL_SFT_RECORDS:
        raise ValueError(
            "Canonical controlled SFT requires exactly 10000 records; "
            f"found {count} in {source}"
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
) -> dict:
    revisions = _load_source_revisions(source_revisions_path)
    return {
        "base_model_sha": str(revisions["base_model"]["sha"]),
        "sft_dataset_sha": str(revisions["sft_dataset"]["sha"]),
        "sft_data_manifest_sha256": sha256_file(Path(sft_manifest_path)),
        "sft_config_sha256": sha256_file(Path(config_path)),
    }


def run_sft(
    *,
    config_path: Path,
    records_path: Path,
    source_revisions_path: Path,
    sft_manifest_path: Path,
    output_dir: Path,
    smoke_steps: int | None = None,
) -> dict:
    canonical = smoke_steps is None
    record_count = validate_record_count(records_path, canonical=canonical)
    config = load_config(Path(config_path))
    validate_sft_config(config)
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
        processing_class=tokenizer,
    )
    trainer.train()

    lineage = build_sft_lineage(
        source_revisions_path,
        sft_manifest_path,
        config_path,
    )

    if canonical:
        manifest = freeze_pi0(
            trainer,
            tokenizer,
            destination / "pi_0",
            lineage,
        )
        return {
            "mode": "canonical",
            "record_count": record_count,
            "pi0_manifest": manifest,
        }

    smoke_final = destination / "smoke_final"
    trainer.save_model(str(smoke_final))
    tokenizer.save_pretrained(str(smoke_final))
    return {
        "mode": "smoke",
        "record_count": record_count,
        "smoke_steps": int(smoke_steps),
    }
