from __future__ import annotations

import json
from pathlib import Path

import pytest

from controlled_run.prepare_analysis_inputs import (
    get_artifact_bundle,
    load_artifact_registry,
    provision_artifact_bundle,
    resolve_bundle_revision,
)


PINNED_SHA = "a" * 40


def _write_registry(
    path: Path,
    *,
    expected_revision_sha: str | None = PINNED_SHA,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bundles": {
                    "reference": {
                        "repo_type": "model",
                        "repo_id": "owner/repo",
                        "revision": "main",
                        "expected_revision_sha": expected_revision_sha,
                        "files": [
                            {
                                "remote_path": "signal_ledger/rank0.jsonl",
                                "role": "ledger_rank0",
                            },
                            {
                                "remote_path": "p0/rollouts.jsonl",
                                "role": "p0_rollouts",
                            },
                        ],
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_repository_registry_records_current_canonical_reference_paths():
    registry = load_artifact_registry()
    bundle = get_artifact_bundle(
        registry,
        "canonical_grpo_seed42_h1_reference",
    )

    assert bundle["repo_type"] == "model"
    assert (
        bundle["repo_id"]
        == "HKReporter/rlvr-behavior-probe-grpo-canonical-seed42-2026-09-02"
    )
    paths = {item["remote_path"] for item in bundle["files"]}
    assert "grpo_run_manifest.json" in paths
    assert (
        "signal_ledger/signal_ledger_20260902T042720Z_rank0.jsonl"
        in paths
    )
    assert (
        "signal_ledger/signal_ledger_20260902T042720Z_rank1.jsonl"
        in paths
    )
    assert (
        "p0_train_k32_top_p1_canonical/rollouts_shard0of2.jsonl"
        in paths
    )
    assert (
        "p0_train_k32_top_p1_canonical/rollouts_shard1of2.jsonl"
        in paths
    )


def test_resolve_bundle_revision_returns_immutable_sha(tmp_path):
    registry_path = tmp_path / "registry.json"
    _write_registry(registry_path)
    registry = load_artifact_registry(registry_path)
    bundle = get_artifact_bundle(registry, "reference")

    resolved = resolve_bundle_revision(
        bundle,
        revision_resolver=lambda repo_id, revision, repo_type: PINNED_SHA,
    )

    assert resolved == PINNED_SHA


def test_unpinned_bundle_fails_closed_before_download(tmp_path):
    registry_path = tmp_path / "registry.json"
    _write_registry(registry_path, expected_revision_sha=None)
    calls: list[dict] = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        raise AssertionError("download must not run for an unpinned bundle")

    with pytest.raises(RuntimeError, match="not pinned"):
        provision_artifact_bundle(
            bundle_name="reference",
            registry_path=registry_path,
            output_root=tmp_path / "out",
            revision_resolver=lambda repo_id, revision, repo_type: PINNED_SHA,
            download_file=fake_download,
        )

    assert calls == []


def test_revision_mismatch_fails_closed_before_download(tmp_path):
    registry_path = tmp_path / "registry.json"
    _write_registry(registry_path, expected_revision_sha=PINNED_SHA)
    other_sha = "b" * 40
    calls: list[dict] = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        raise AssertionError("download must not run on revision mismatch")

    with pytest.raises(RuntimeError, match="revision mismatch"):
        provision_artifact_bundle(
            bundle_name="reference",
            registry_path=registry_path,
            output_root=tmp_path / "out",
            revision_resolver=lambda repo_id, revision, repo_type: other_sha,
            download_file=fake_download,
        )

    assert calls == []


def test_pinned_bundle_downloads_exact_files_and_writes_hash_manifest(tmp_path):
    registry_path = tmp_path / "registry.json"
    _write_registry(registry_path)
    cache = tmp_path / "cache"

    payloads = {
        "signal_ledger/rank0.jsonl": b'{"rank": 0}\n',
        "p0/rollouts.jsonl": b'{"dataset_index": 0}\n',
    }

    def fake_download(*, repo_id, filename, repo_type, revision):
        assert repo_id == "owner/repo"
        assert repo_type == "model"
        assert revision == PINNED_SHA
        path = cache / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payloads[filename])
        return path

    result = provision_artifact_bundle(
        bundle_name="reference",
        registry_path=registry_path,
        output_root=tmp_path / "out",
        revision_resolver=lambda repo_id, revision, repo_type: PINNED_SHA,
        download_file=fake_download,
    )

    root = tmp_path / "out" / "reference"
    assert result["status"] == "READY"
    assert result["resolved_revision_sha"] == PINNED_SHA
    assert (root / "signal_ledger/rank0.jsonl").read_bytes() == payloads[
        "signal_ledger/rank0.jsonl"
    ]
    assert (root / "p0/rollouts.jsonl").read_bytes() == payloads[
        "p0/rollouts.jsonl"
    ]

    manifest = json.loads(
        (root / "provision_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["resolved_revision_sha"] == PINNED_SHA
    assert set(manifest["files"]) == set(payloads)
    assert all(
        len(item["sha256"]) == 64
        for item in manifest["files"].values()
    )


def test_registry_rejects_parent_traversal(tmp_path):
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "bundles": {
                    "bad": {
                        "repo_type": "model",
                        "repo_id": "owner/repo",
                        "revision": "main",
                        "expected_revision_sha": PINNED_SHA,
                        "files": [
                            {
                                "remote_path": "../escape.jsonl",
                                "role": "bad",
                            }
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="safe relative path"):
        load_artifact_registry(registry_path)
