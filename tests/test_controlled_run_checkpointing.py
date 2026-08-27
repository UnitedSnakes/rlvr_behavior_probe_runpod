from __future__ import annotations

import json
from pathlib import Path

import pytest

from controlled_run.checkpointing import freeze_pi0, load_pi0_manifest


LINEAGE = {
    "base_model_sha": "base-sha",
    "sft_dataset_sha": "dataset-sha",
    "sft_data_manifest_sha256": "data-manifest-sha",
    "sft_config_sha256": "config-sha",
}


class FakeTrainer:
    def __init__(self):
        self.saved_to = []

    def save_model(self, output_dir):
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "model.safetensors").write_bytes(b"exact-sft-final-weights")
        (path / "config.json").write_text('{"model":"qwen3"}\n', encoding="utf-8")
        self.saved_to.append(path)


class FakeTokenizer:
    def save_pretrained(self, output_dir):
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer.json").write_text('{"tokenizer":"qwen3"}\n', encoding="utf-8")


def test_freeze_pi0_saves_dedicated_policy_and_self_verifying_manifest(tmp_path):
    trainer = FakeTrainer()
    tokenizer = FakeTokenizer()
    pi0_dir = tmp_path / "pi_0"

    manifest = freeze_pi0(trainer, tokenizer, pi0_dir, LINEAGE)

    assert trainer.saved_to == [pi0_dir]
    assert manifest["policy_name"] == "pi_0"
    assert {key: manifest[key] for key in LINEAGE} == LINEAGE
    assert set(manifest["files"]) == {
        "config.json",
        "model.safetensors",
        "tokenizer.json",
    }
    assert (pi0_dir / "pi0_manifest.json").exists()

    on_disk = json.loads(
        (pi0_dir / "pi0_manifest.json").read_text(encoding="utf-8")
    )
    assert on_disk == manifest
    assert load_pi0_manifest(pi0_dir) == manifest


def test_load_pi0_manifest_rejects_changed_weight(tmp_path):
    pi0_dir = tmp_path / "pi_0"
    freeze_pi0(FakeTrainer(), FakeTokenizer(), pi0_dir, LINEAGE)
    (pi0_dir / "model.safetensors").write_bytes(b"mutated")

    with pytest.raises(ValueError, match="fingerprint.*model.safetensors"):
        load_pi0_manifest(pi0_dir)


def test_load_pi0_manifest_rejects_untracked_extra_file(tmp_path):
    pi0_dir = tmp_path / "pi_0"
    freeze_pi0(FakeTrainer(), FakeTokenizer(), pi0_dir, LINEAGE)
    (pi0_dir / "unexpected.bin").write_bytes(b"extra")

    with pytest.raises(ValueError, match="unexpected.bin"):
        load_pi0_manifest(pi0_dir)


def test_freeze_pi0_refuses_nonempty_destination(tmp_path):
    pi0_dir = tmp_path / "pi_0"
    pi0_dir.mkdir()
    (pi0_dir / "stale.txt").write_text("stale", encoding="utf-8")

    with pytest.raises(FileExistsError, match="pi_0.*not empty"):
        freeze_pi0(FakeTrainer(), FakeTokenizer(), pi0_dir, LINEAGE)
