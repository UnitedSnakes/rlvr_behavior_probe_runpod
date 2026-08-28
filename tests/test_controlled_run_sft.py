from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import controlled_run.train_sft as train_sft
from controlled_run.config import load_config


ROOT = Path(__file__).resolve().parents[1]


class FakeSFTConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _write_record(path: Path, uuid: str) -> None:
    row = {
        "uuid": uuid,
        "prompt": [{"role": "user", "content": "1+1?"}],
        "completion": [{"role": "assistant", "content": "\\boxed{2}"}],
    }
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row) + "\n")


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
        "per_device_eval_batch_size": 8,
        "gradient_accumulation_steps": 8,
        "optim": "adamw_torch_fused",
        "save_strategy": "epoch",
        "eval_strategy": "epoch",
        "load_best_model_at_end": False,
        "report_to": "none",
        "seed": 42,
        "data_seed": 42,
    }


def test_build_sft_arguments_adds_max_steps_only_for_smoke(monkeypatch, tmp_path):
    monkeypatch.setattr(
        train_sft,
        "_load_sft_classes",
        lambda: (FakeSFTConfig, object),
    )
    config = load_config(ROOT / "controlled_run/configs/sft_qwen3_0_6b.yaml")

    args = train_sft.build_sft_arguments(config, tmp_path / "trainer", max_steps=2)

    assert args.kwargs["max_steps"] == 2
    assert args.kwargs["num_train_epochs"] == 2
    assert args.kwargs["eval_strategy"] == "no"
    assert args.kwargs["load_best_model_at_end"] is False


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


def test_validate_record_count_enforces_10000_only_for_canonical(tmp_path):
    records = tmp_path / "records.jsonl"
    _write_record(records, "u1")

    with pytest.raises(ValueError, match="exactly 10000"):
        train_sft.validate_record_count(records, canonical=True)

    assert train_sft.validate_record_count(records, canonical=False) == 1


def test_validate_validation_record_count_enforces_512_only_for_canonical(tmp_path):
    records = tmp_path / "validation.jsonl"
    _write_record(records, "u1")

    with pytest.raises(ValueError, match="exactly 512"):
        train_sft.validate_validation_record_count(records, canonical=True)

    assert train_sft.validate_validation_record_count(records, canonical=False) == 1


def test_build_sft_lineage_hashes_train_val_manifest_and_config(tmp_path):
    source_revisions = tmp_path / "source_revisions.json"
    source_revisions.write_text(
        json.dumps(
            {
                "base_model": {"repo_id": "Qwen/Qwen3-0.6B-Base", "sha": "base-sha"},
                "sft_dataset": {"repo_id": "open-r1/OpenR1-Math-220k", "sha": "sft-sha"},
            }
        ),
        encoding="utf-8",
    )
    data_manifest = tmp_path / "sft_10k_manifest.jsonl"
    data_manifest.write_text('{"source_index":1}\n', encoding="utf-8")
    validation_manifest = tmp_path / "sft_val_512_manifest.jsonl"
    validation_manifest.write_text('{"source_index":2}\n', encoding="utf-8")
    config_path = tmp_path / "sft.yaml"
    config_path.write_text("model_name: x\n", encoding="utf-8")

    lineage = train_sft.build_sft_lineage(
        source_revisions,
        data_manifest,
        config_path,
        validation_manifest_path=validation_manifest,
    )

    assert lineage["base_model_sha"] == "base-sha"
    assert lineage["sft_dataset_sha"] == "sft-sha"
    assert len(lineage["sft_data_manifest_sha256"]) == 64
    assert len(lineage["sft_validation_manifest_sha256"]) == 64
    assert len(lineage["sft_config_sha256"]) == 64


def test_run_sft_loads_exact_base_sha_and_freezes_pi0_only_in_canonical_mode(
    monkeypatch,
    tmp_path,
):
    config_path = ROOT / "controlled_run/configs/sft_qwen3_0_6b.yaml"
    records = tmp_path / "records.jsonl"
    _write_record(records, "u1")
    source_revisions = tmp_path / "source_revisions.json"
    source_revisions.write_text(
        json.dumps(
            {
                "base_model": {
                    "repo_id": "Qwen/Qwen3-0.6B-Base",
                    "sha": "base-sha",
                },
                "sft_dataset": {
                    "repo_id": "open-r1/OpenR1-Math-220k",
                    "sha": "sft-sha",
                },
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "sft_10k_manifest.jsonl"
    manifest_path.write_text('{"uuid":"u1"}\n', encoding="utf-8")

    model_calls = []
    tokenizer_calls = []

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, repo_id, **kwargs):
            model_calls.append((repo_id, kwargs))
            return "MODEL"

    class FakeTokenizer:
        def save_pretrained(self, output_dir):
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            (Path(output_dir) / "tokenizer.json").write_text("{}", encoding="utf-8")

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, repo_id, **kwargs):
            tokenizer_calls.append((repo_id, kwargs))
            return FakeTokenizer()

    trainer_instances = []

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.trained = False
            trainer_instances.append(self)

        def train(self):
            self.trained = True

        def save_model(self, output_dir):
            path = Path(output_dir)
            path.mkdir(parents=True, exist_ok=True)
            (path / "model.safetensors").write_bytes(b"weights")

    monkeypatch.setattr(train_sft, "AutoModelForCausalLM", FakeAutoModel)
    monkeypatch.setattr(train_sft, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(
        train_sft,
        "_load_sft_classes",
        lambda: (FakeSFTConfig, FakeTrainer),
    )

    smoke_dir = tmp_path / "smoke"
    result = train_sft.run_sft(
        config_path=config_path,
        records_path=records,
        source_revisions_path=source_revisions,
        sft_manifest_path=manifest_path,
        output_dir=smoke_dir,
        smoke_steps=2,
    )

    assert model_calls[0][0] == "Qwen/Qwen3-0.6B-Base"
    assert model_calls[0][1]["revision"] == "base-sha"
    assert model_calls[0][1]["dtype"] is torch.bfloat16
    assert model_calls[0][1]["attn_implementation"] == "flash_attention_2"
    assert tokenizer_calls[0] == (
        "Qwen/Qwen3-0.6B-Base",
        {"revision": "base-sha"},
    )
    assert trainer_instances[0].trained is True
    assert trainer_instances[0].kwargs["eval_dataset"] is None
    assert result["mode"] == "smoke"
    assert (smoke_dir / "smoke_final" / "model.safetensors").exists()
    assert not (smoke_dir / "pi_0").exists()
