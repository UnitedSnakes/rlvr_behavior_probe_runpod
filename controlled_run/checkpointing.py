from __future__ import annotations

import json
from pathlib import Path

from controlled_run.provenance import (
    directory_fingerprint,
    verify_directory_fingerprint,
    write_json,
)


PI0_MANIFEST_NAME = "pi0_manifest.json"
_REQUIRED_LINEAGE_KEYS = {
    "base_model_sha",
    "sft_dataset_sha",
    "sft_data_manifest_sha256",
    "sft_validation_manifest_sha256",
    "sft_config_sha256",
}


def _validate_lineage(lineage: dict) -> None:
    missing = sorted(_REQUIRED_LINEAGE_KEYS - set(lineage))
    if missing:
        raise ValueError(
            "pi_0 lineage is missing required field " + ", ".join(missing)
        )
    for key in sorted(_REQUIRED_LINEAGE_KEYS):
        if not isinstance(lineage[key], str) or not lineage[key]:
            raise ValueError(f"pi_0 lineage field {key!r} must be a non-empty string")


def _ensure_empty_destination(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"pi_0 destination is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def freeze_pi0(
    source_trainer,
    tokenizer,
    pi0_dir: Path,
    lineage: dict,
) -> dict:
    destination = Path(pi0_dir)
    _validate_lineage(lineage)
    _ensure_empty_destination(destination)

    source_trainer.save_model(str(destination))
    tokenizer.save_pretrained(str(destination))

    files = directory_fingerprint(
        destination,
        exclude={PI0_MANIFEST_NAME},
    )
    if not files:
        raise RuntimeError("pi_0 save produced no model/tokenizer files")

    manifest = {
        "policy_name": "pi_0",
        **{key: lineage[key] for key in sorted(_REQUIRED_LINEAGE_KEYS)},
        "files": files,
    }
    write_json(destination / PI0_MANIFEST_NAME, manifest)
    return manifest


def load_pi0_manifest(pi0_dir: Path) -> dict:
    directory = Path(pi0_dir)
    manifest_path = directory / PI0_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing pi_0 manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("policy_name") != "pi_0":
        raise ValueError(
            f"Expected policy_name='pi_0' in {manifest_path}, "
            f"got {manifest.get('policy_name')!r}"
        )
    _validate_lineage(manifest)

    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"pi_0 manifest has no file fingerprint map: {manifest_path}")

    verify_directory_fingerprint(
        directory,
        files,
        exclude={PI0_MANIFEST_NAME},
    )
    return manifest
