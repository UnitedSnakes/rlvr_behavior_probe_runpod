from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset

from controlled_run.config import validate_sft_config


def _load_sft_classes():
    try:
        from trl import SFTConfig, SFTTrainer
    except ImportError as error:
        raise RuntimeError(
            "TRL is required for controlled SFT training. Install "
            "controlled_run/requirements-a40.in in the isolated A40 environment."
        ) from error
    return SFTConfig, SFTTrainer


def build_sft_arguments(config: dict, output_dir: Path):
    validate_sft_config(config)
    SFTConfig, _ = _load_sft_classes()

    return SFTConfig(
        output_dir=str(Path(output_dir)),
        num_train_epochs=config["num_train_epochs"],
        max_length=config["max_length"],
        bf16=config["bf16"],
        gradient_checkpointing=config["gradient_checkpointing"],
        packing=config["packing"],
        packing_strategy=config["packing_strategy"],
        completion_only_loss=config["completion_only_loss"],
        learning_rate=config["learning_rate"],
        lr_scheduler_type=config["lr_scheduler_type"],
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        optim=config["optim"],
        save_strategy="epoch",
        report_to="none",
        seed=config["seed"],
        data_seed=config["seed"],
    )


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
