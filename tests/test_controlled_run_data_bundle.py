from __future__ import annotations

import json
from pathlib import Path

import pytest

from controlled_run.data_bundle import verify_data_bundle, write_data_bundle_manifest


def _write_jsonl(path: Path, count: int, prefix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps({"id": f"{prefix}-{i}"}) + "\n" for i in range(count)),
        encoding="utf-8",
    )


def _make_bundle_files(tmp_path: Path) -> tuple[Path, Path]:
    manifests = tmp_path / "manifests"
    generated = tmp_path / "generated"
    manifests.mkdir()
    generated.mkdir()

    _write_jsonl(manifests / "sft_10k_manifest.jsonl", 2, "train-manifest")
    _write_jsonl(manifests / "sft_val_512_manifest.jsonl", 1, "val-manifest")
    _write_jsonl(generated / "sft_10k_records.jsonl", 2, "train-record")
    _write_jsonl(generated / "sft_val_512_records.jsonl", 1, "val-record")

    (manifests / "contamination_audit.json").write_text(
        json.dumps({"train_count": 2, "validation_count": 1}) + "\n",
        encoding="utf-8",
    )
    (manifests / "source_revisions.json").write_text(
        json.dumps(
            {
                "base_model": {"repo_id": "model", "sha": "model-sha"},
                "sft_dataset": {"repo_id": "sft", "sha": "sft-sha"},
                "gsm8k_dataset": {"repo_id": "gsm8k", "sha": "gsm8k-sha"},
                "seed": 42,
                "target_size": 2,
                "validation_size": 1,
                "source_identity": "pinned_dataset_revision_plus_source_index",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return manifests, generated


def test_write_and_verify_data_bundle(tmp_path):
    manifests, generated = _make_bundle_files(tmp_path)

    written = write_data_bundle_manifest(manifests, generated)
    verified = verify_data_bundle(manifests, generated)

    assert written == verified
    assert verified["schema_version"] == 1
    assert verified["target_size"] == 2
    assert verified["validation_size"] == 1
    assert verified["source_identity"] == "pinned_dataset_revision_plus_source_index"
    assert verified["source_revisions"]["sft_dataset"]["sha"] == "sft-sha"
    assert set(verified["artifacts"]) == {
        "manifests/sft_10k_manifest.jsonl",
        "manifests/sft_val_512_manifest.jsonl",
        "manifests/contamination_audit.json",
        "manifests/source_revisions.json",
        "generated/sft_10k_records.jsonl",
        "generated/sft_val_512_records.jsonl",
    }
    assert all(len(item["sha256"]) == 64 for item in verified["artifacts"].values())


def test_verify_data_bundle_rejects_changed_artifact(tmp_path):
    manifests, generated = _make_bundle_files(tmp_path)
    write_data_bundle_manifest(manifests, generated)
    with (generated / "sft_10k_records.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps({"id": "tampered"}) + "\n")

    with pytest.raises(ValueError, match="SHA256 mismatch.*sft_10k_records"):
        verify_data_bundle(manifests, generated)


def test_verify_data_bundle_rejects_missing_artifact(tmp_path):
    manifests, generated = _make_bundle_files(tmp_path)
    write_data_bundle_manifest(manifests, generated)
    (manifests / "contamination_audit.json").unlink()

    with pytest.raises(FileNotFoundError, match="contamination_audit"):
        verify_data_bundle(manifests, generated)


def test_verify_data_bundle_rejects_count_mismatch_even_with_updated_hash(tmp_path):
    manifests, generated = _make_bundle_files(tmp_path)
    _write_jsonl(generated / "sft_val_512_records.jsonl", 2, "wrong-val-record")
    write_data_bundle_manifest(manifests, generated)

    with pytest.raises(ValueError, match="validation records.*expected 1.*found 2"):
        verify_data_bundle(manifests, generated)
