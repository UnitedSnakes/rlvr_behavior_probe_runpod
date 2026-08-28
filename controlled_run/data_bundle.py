from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCHEMA_VERSION = 1
ARTIFACTS = (
    ("manifests/sft_10k_manifest.jsonl", "train manifest"),
    ("manifests/sft_val_512_manifest.jsonl", "validation manifest"),
    ("manifests/contamination_audit.json", "contamination audit"),
    ("manifests/source_revisions.json", "source revisions"),
    ("generated/sft_10k_records.jsonl", "train records"),
    ("generated/sft_val_512_records.jsonl", "validation records"),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing controlled data bundle file: {path}") from None
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in controlled data bundle file: {path}")
    return payload


def _count_jsonl(path: Path) -> int:
    try:
        return sum(
            1
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing controlled data bundle file: {path}") from None


def _artifact_path(relative_path: str, manifests_dir: Path, generated_dir: Path) -> Path:
    prefix, name = relative_path.split("/", 1)
    if prefix == "manifests":
        return manifests_dir / name
    if prefix == "generated":
        return generated_dir / name
    raise ValueError(f"Unsupported controlled data bundle path: {relative_path}")


def _validate_source_revisions(source_revisions: dict) -> tuple[int, int, str]:
    try:
        target_size = int(source_revisions["target_size"])
        validation_size = int(source_revisions["validation_size"])
        source_identity = str(source_revisions["source_identity"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "source_revisions.json must contain target_size, validation_size, "
            "and source_identity"
        ) from error
    if target_size <= 0 or validation_size <= 0:
        raise ValueError("Controlled data bundle train/validation sizes must be positive")
    if source_identity != "pinned_dataset_revision_plus_source_index":
        raise ValueError(
            "Controlled data bundle requires pinned_dataset_revision_plus_source_index identity"
        )
    return target_size, validation_size, source_identity


def write_data_bundle_manifest(manifests_dir: Path, generated_dir: Path) -> dict:
    manifests_dir = Path(manifests_dir)
    generated_dir = Path(generated_dir)
    source_revisions = _json(manifests_dir / "source_revisions.json")
    target_size, validation_size, source_identity = _validate_source_revisions(
        source_revisions
    )

    artifacts: dict[str, dict] = {}
    for relative_path, _label in ARTIFACTS:
        path = _artifact_path(relative_path, manifests_dir, generated_dir)
        if not path.is_file():
            raise FileNotFoundError(f"Missing controlled data bundle file: {path}")
        artifacts[relative_path] = {
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_identity": source_identity,
        "target_size": target_size,
        "validation_size": validation_size,
        "source_revisions": source_revisions,
        "artifacts": artifacts,
    }
    destination = manifests_dir / "data_bundle_manifest.json"
    destination.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def verify_data_bundle(manifests_dir: Path, generated_dir: Path) -> dict:
    manifests_dir = Path(manifests_dir)
    generated_dir = Path(generated_dir)
    bundle_path = manifests_dir / "data_bundle_manifest.json"
    bundle = _json(bundle_path)

    if bundle.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported controlled data bundle schema: {bundle.get('schema_version')!r}"
        )

    source_revisions = _json(manifests_dir / "source_revisions.json")
    target_size, validation_size, source_identity = _validate_source_revisions(
        source_revisions
    )
    if bundle.get("source_revisions") != source_revisions:
        raise ValueError("Controlled data bundle source revisions do not match manifest")
    if bundle.get("target_size") != target_size:
        raise ValueError("Controlled data bundle target_size does not match source revisions")
    if bundle.get("validation_size") != validation_size:
        raise ValueError("Controlled data bundle validation_size does not match source revisions")
    if bundle.get("source_identity") != source_identity:
        raise ValueError("Controlled data bundle source identity does not match source revisions")

    artifact_manifest = bundle.get("artifacts")
    if not isinstance(artifact_manifest, dict):
        raise ValueError("Controlled data bundle manifest is missing artifacts")

    for relative_path, _label in ARTIFACTS:
        path = _artifact_path(relative_path, manifests_dir, generated_dir)
        if not path.is_file():
            raise FileNotFoundError(f"Missing controlled data bundle file: {path}")
        entry = artifact_manifest.get(relative_path)
        if not isinstance(entry, dict) or not entry.get("sha256"):
            raise ValueError(f"Controlled data bundle manifest is missing {relative_path}")
        actual_sha = _sha256_file(path)
        if actual_sha != entry["sha256"]:
            raise ValueError(
                f"SHA256 mismatch for controlled data bundle artifact {relative_path}: "
                f"expected {entry['sha256']}, got {actual_sha}"
            )

    expected_counts = (
        (manifests_dir / "sft_10k_manifest.jsonl", target_size, "train manifest"),
        (
            manifests_dir / "sft_val_512_manifest.jsonl",
            validation_size,
            "validation manifest",
        ),
        (generated_dir / "sft_10k_records.jsonl", target_size, "train records"),
        (
            generated_dir / "sft_val_512_records.jsonl",
            validation_size,
            "validation records",
        ),
    )
    for path, expected, label in expected_counts:
        actual = _count_jsonl(path)
        if actual != expected:
            raise ValueError(
                f"Controlled data bundle {label} count mismatch: "
                f"expected {expected}, found {actual}"
            )

    audit = _json(manifests_dir / "contamination_audit.json")
    if audit.get("train_count") != target_size:
        raise ValueError("Controlled data bundle audit train_count does not match target_size")
    if audit.get("validation_count") != validation_size:
        raise ValueError(
            "Controlled data bundle audit validation_count does not match validation_size"
        )

    return bundle


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Verify a portable controlled SFT data bundle.")
    parser.add_argument(
        "--manifests-dir",
        type=Path,
        default=Path("data/controlled_run/manifests"),
    )
    parser.add_argument(
        "--generated-dir",
        type=Path,
        default=Path("data/controlled_run/generated"),
    )
    args = parser.parse_args(argv)
    verified = verify_data_bundle(args.manifests_dir, args.generated_dir)
    print(json.dumps(verified, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
