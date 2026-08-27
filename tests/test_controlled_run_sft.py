from __future__ import annotations

import json
from pathlib import Path

import pytest

import controlled_run.train_sft as train_sft
from controlled_run.config import load_config


ROOT = Path(__file__).resolve().parents[1]


class FakeSFTConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_build_sft_arguments_maps_exact_canonical_recipe(monkeypatch, tmp_path):
    monkeypatch.setattr(
        train_sft,
        "_load_sft_classes",
        lambda: (FakeSFTConfig, object),
    )
    config = load_config(ROOT / "controlled_run/configs/sft_qwen3_0_6b.yaml")

    args = train_sft.build_sft_arguments(config, tmp_path / "trainer")

    assert args.kwargs == {
        "output_dir": str(tmp_path / "trainer"),
        "num_train_epochs": 2,
        "max_length": 2048,
        "bf16": True,
        "gradient_checkpointing": True,
        "packing": True,
        "packing_strategy": "bfd",
        "completion_only_loss": True,
        "learning_rate": 2e-5,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "weight_decay": 0.01,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 8,
        "optim": "adamw_torch_fused",
        "save_strategy": "epoch",
        "report_to": "none",
        "seed": 42,
        "data_seed": 42,
    }


def test_load_prompt_completion_jsonl_preserves_conversational_columns(tmp_path):
    path = tmp_path / "records.jsonl"
    rows = [
        {
            "uuid": "u1",
            "prompt": [{"role": "user", "content": "1+1?"}],
            "completion": [{"role": "assistant", "content": "\\boxed{2}"}],
        },
        {
            "uuid": "u2",
            "prompt": [{"role": "user", "content": "2+2?"}],
            "completion": [{"role": "assistant", "content": "\\boxed{4}"}],
        },
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    dataset = train_sft.load_prompt_completion_jsonl(path)

    assert len(dataset) == 2
    assert set(dataset.column_names) == {"uuid", "prompt", "completion"}
    assert dataset[0]["uuid"] == "u1"
    assert dataset[0]["completion"][0]["content"] == "\\boxed{2}"


def test_load_prompt_completion_jsonl_rejects_missing_prompt_or_completion(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"uuid": "u1", "prompt": []}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="prompt.*completion"):
        train_sft.load_prompt_completion_jsonl(path)
