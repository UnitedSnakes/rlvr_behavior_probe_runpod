from __future__ import annotations

import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi


_CHUNK_SIZE = 1024 * 1024


def resolve_hf_revision(repo_id: str, revision: str, repo_type: str) -> str:
    api = HfApi()

    if repo_type == "model":
        info = api.model_info(repo_id, revision=revision)
    elif repo_type == "dataset":
        info = api.dataset_info(repo_id, revision=revision)
    else:
        raise ValueError(
            f"repo_type must be 'model' or 'dataset', got {repo_type!r}"
        )

    sha = getattr(info, "sha", None)
    if not sha:
        raise RuntimeError(
            f"Hugging Face did not return an immutable SHA for "
            f"{repo_type} {repo_id!r} at revision {revision!r}"
        )
    return str(sha)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while True:
            chunk = file.read(_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def directory_fingerprint(
    path: Path,
    exclude: set[str] | None = None,
) -> dict[str, str]:
    root = Path(path)
    excluded = set(exclude or ())
    files: dict[str, str] = {}

    for candidate in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(root).as_posix()
        if relative in excluded:
            continue
        files[relative] = sha256_file(candidate)

    return files


def verify_directory_fingerprint(
    path: Path,
    expected: dict[str, str],
    exclude: set[str] | None = None,
) -> None:
    actual = directory_fingerprint(path, exclude=exclude)
    expected_paths = set(expected)
    actual_paths = set(actual)

    missing = sorted(expected_paths - actual_paths)
    if missing:
        raise ValueError(
            f"Directory fingerprint mismatch: missing file {missing[0]}"
        )

    extra = sorted(actual_paths - expected_paths)
    if extra:
        raise ValueError(
            f"Directory fingerprint mismatch: unexpected file {extra[0]}"
        )

    for relative in sorted(expected):
        if actual[relative] != expected[relative]:
            raise ValueError(
                f"Directory fingerprint mismatch for {relative}: "
                f"expected {expected[relative]}, got {actual[relative]}"
            )


def write_json(path: Path, payload: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
