from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi


def _as_utc(run_started_at: datetime) -> datetime:
    if run_started_at.tzinfo is None:
        raise ValueError("run_started_at must be timezone-aware")
    return run_started_at.astimezone(timezone.utc)


def format_run_started_at(run_started_at: datetime) -> str:
    utc_time = _as_utc(run_started_at).replace(microsecond=0)
    return utc_time.isoformat().replace("+00:00", "Z")


def build_remote_run_path(result_dir: Path, run_started_at: datetime) -> str:
    utc_time = _as_utc(run_started_at).replace(microsecond=0)
    timestamp = utc_time.strftime("%Y%m%dT%H%M%SZ")
    return f"runs/{timestamp}-{result_dir.name}"


def upload_result_dir(result_dir: Path, repo_id: str, remote_path: str) -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required when --upload-repo is requested")

    api = HfApi(token=token)
    existing_files = api.list_repo_files(
        repo_id=repo_id,
        repo_type="dataset",
    )

    normalized_remote_path = remote_path.rstrip("/")
    remote_prefix = normalized_remote_path + "/"
    collision = any(
        file_path == normalized_remote_path or file_path.startswith(remote_prefix)
        for file_path in existing_files
    )
    if collision:
        raise FileExistsError(
            f"Remote run path already exists: {normalized_remote_path}"
        )

    api.upload_folder(
        folder_path=str(result_dir),
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo=normalized_remote_path,
    )
