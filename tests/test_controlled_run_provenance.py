from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import controlled_run.provenance as provenance
from controlled_run.provenance import (
    directory_fingerprint,
    resolve_hf_revision,
    sha256_file,
    verify_directory_fingerprint,
    write_json,
)


def test_resolve_model_revision_returns_server_sha(monkeypatch):
    class FakeApi:
        def model_info(self, repo_id, revision):
            assert repo_id == "owner/model"
            assert revision == "main"
            return SimpleNamespace(sha="model-sha")

    monkeypatch.setattr(provenance, "HfApi", FakeApi)

    assert resolve_hf_revision("owner/model", "main", "model") == "model-sha"


def test_resolve_dataset_revision_uses_dataset_info(monkeypatch):
    class FakeApi:
        def dataset_info(self, repo_id, revision):
            assert repo_id == "owner/data"
            assert revision == "refs/convert/parquet"
            return SimpleNamespace(sha="dataset-sha")

    monkeypatch.setattr(provenance, "HfApi", FakeApi)

    assert (
        resolve_hf_revision("owner/data", "refs/convert/parquet", "dataset")
        == "dataset-sha"
    )


def test_resolve_revision_rejects_unknown_repo_type():
    with pytest.raises(ValueError, match="repo_type"):
        resolve_hf_revision("owner/thing", "main", "space")


def test_sha256_file_and_directory_fingerprint_are_stable(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "b.txt").write_text("bravo", encoding="utf-8")
    (tmp_path / "nested" / "a.txt").write_text("alpha", encoding="utf-8")

    fingerprint = directory_fingerprint(tmp_path)

    assert list(fingerprint) == ["b.txt", "nested/a.txt"]
    assert fingerprint["b.txt"] == sha256_file(tmp_path / "b.txt")
    assert fingerprint["nested/a.txt"] == sha256_file(tmp_path / "nested/a.txt")


def test_directory_fingerprint_can_exclude_manifest(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "pi0_manifest.json").write_text("{}", encoding="utf-8")

    fingerprint = directory_fingerprint(tmp_path, exclude={"pi0_manifest.json"})

    assert set(fingerprint) == {"model.safetensors"}


def test_verify_directory_fingerprint_detects_changed_weight(tmp_path):
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"abc")
    fingerprint = directory_fingerprint(tmp_path)

    verify_directory_fingerprint(tmp_path, fingerprint)
    weight.write_bytes(b"changed")

    with pytest.raises(ValueError, match="fingerprint.*model.safetensors"):
        verify_directory_fingerprint(tmp_path, fingerprint)


def test_verify_directory_fingerprint_detects_extra_file(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"abc")
    fingerprint = directory_fingerprint(tmp_path)
    (tmp_path / "unexpected.txt").write_text("extra", encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected.txt"):
        verify_directory_fingerprint(tmp_path, fingerprint)


def test_write_json_is_stable_and_creates_parent_directories(tmp_path):
    path = tmp_path / "nested" / "manifest.json"

    write_json(path, {"z": 1, "a": {"d": 4, "b": 2}})

    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert text.index('"a"') < text.index('"z"')
    assert json.loads(text) == {"a": {"b": 2, "d": 4}, "z": 1}
