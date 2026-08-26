from datetime import datetime, timezone
from pathlib import Path

import pytest

import probe.results_upload as results_upload
from probe.results_upload import build_remote_run_path, format_run_started_at


class FakeApi:
    def __init__(self, token):
        self.token = token
        self.upload_calls = []
        self.repo_files = []
        self.upload_error = None

    def list_repo_files(self, *, repo_id, repo_type):
        assert repo_type == "dataset"
        return list(self.repo_files)

    def upload_folder(self, **kwargs):
        if self.upload_error is not None:
            raise self.upload_error
        self.upload_calls.append(kwargs)


def test_format_run_started_at_uses_utc_z_suffix():
    started_at = datetime(2026, 8, 25, 23, 53, 12, tzinfo=timezone.utc)

    assert format_run_started_at(started_at) == "2026-08-25T23:53:12Z"


def test_build_remote_run_path_uses_start_time_and_result_dir_basename():
    started_at = datetime(2026, 8, 25, 23, 53, 12, tzinfo=timezone.utc)
    result_dir = Path("/workspace/experiments/results_sft256_vllm")

    assert build_remote_run_path(result_dir, started_at) == (
        "runs/20260825T235312Z-results_sft256_vllm"
    )


def test_upload_requires_hf_token(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        results_upload.upload_result_dir(
            tmp_path,
            "UnitedSnakes/rlvr-behavior-probe-results",
            "runs/20260825T235312Z-results",
        )


def test_upload_uses_dataset_repo_and_expected_remote_path(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "test-token")
    fake_api = FakeApi(token="test-token")
    monkeypatch.setattr(results_upload, "HfApi", lambda token: fake_api)

    results_upload.upload_result_dir(
        tmp_path,
        "UnitedSnakes/rlvr-behavior-probe-results",
        "runs/20260825T235312Z-results",
    )

    assert fake_api.token == "test-token"
    assert fake_api.upload_calls == [
        {
            "folder_path": str(tmp_path),
            "repo_id": "UnitedSnakes/rlvr-behavior-probe-results",
            "repo_type": "dataset",
            "path_in_repo": "runs/20260825T235312Z-results",
        }
    ]


def test_upload_rejects_existing_remote_run(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "test-token")
    fake_api = FakeApi(token="test-token")
    fake_api.repo_files = [
        "runs/20260825T235312Z-results/run_config.json",
    ]
    monkeypatch.setattr(results_upload, "HfApi", lambda token: fake_api)

    with pytest.raises(FileExistsError, match="already exists"):
        results_upload.upload_result_dir(
            tmp_path,
            "UnitedSnakes/rlvr-behavior-probe-results",
            "runs/20260825T235312Z-results",
        )

    assert fake_api.upload_calls == []


def test_upload_api_errors_propagate(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "test-token")
    fake_api = FakeApi(token="test-token")
    fake_api.upload_error = RuntimeError("network down")
    monkeypatch.setattr(results_upload, "HfApi", lambda token: fake_api)

    with pytest.raises(RuntimeError, match="network down"):
        results_upload.upload_result_dir(
            tmp_path,
            "UnitedSnakes/rlvr-behavior-probe-results",
            "runs/20260825T235312Z-results",
        )
