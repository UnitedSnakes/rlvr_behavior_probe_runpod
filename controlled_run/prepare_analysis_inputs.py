from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Callable

from huggingface_hub import hf_hub_download

from controlled_run.provenance import resolve_hf_revision, sha256_file, write_json


DEFAULT_REGISTRY = Path(__file__).with_name("artifact_registry.json")
DEFAULT_OUTPUT_ROOT = Path("controlled_run_outputs/reference_inputs")
PROVISION_MANIFEST_NAME = "provision_manifest.json"


def _safe_relative_path(raw: str) -> Path:
    value = str(raw)
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"artifact path must be a safe relative path, got {raw!r}")
    return path


def load_artifact_registry(path: Path = DEFAULT_REGISTRY) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("artifact registry schema_version must equal 1")

    bundles = payload.get("bundles")
    if not isinstance(bundles, dict) or not bundles:
        raise ValueError("artifact registry must contain a non-empty bundles mapping")

    for name, bundle in bundles.items():
        if not isinstance(bundle, dict):
            raise ValueError(f"artifact bundle {name!r} must be a mapping")
        if bundle.get("repo_type") not in {"model", "dataset"}:
            raise ValueError(
                f"artifact bundle {name!r} repo_type must be 'model' or 'dataset'"
            )
        if not isinstance(bundle.get("repo_id"), str) or not bundle["repo_id"]:
            raise ValueError(f"artifact bundle {name!r} must define repo_id")
        if not isinstance(bundle.get("revision"), str) or not bundle["revision"]:
            raise ValueError(f"artifact bundle {name!r} must define revision")

        expected_sha = bundle.get("expected_revision_sha")
        if expected_sha is not None:
            if (
                not isinstance(expected_sha, str)
                or len(expected_sha) != 40
                or any(char not in "0123456789abcdef" for char in expected_sha.lower())
            ):
                raise ValueError(
                    f"artifact bundle {name!r} expected_revision_sha must be "
                    "a 40-character Git SHA or null"
                )

        files = bundle.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError(f"artifact bundle {name!r} must define a non-empty files list")

        seen_paths: set[str] = set()
        seen_roles: set[str] = set()
        for item in files:
            if not isinstance(item, dict):
                raise ValueError(f"artifact bundle {name!r} file entries must be mappings")
            remote_path = item.get("remote_path")
            role = item.get("role")
            if not isinstance(remote_path, str):
                raise ValueError(f"artifact bundle {name!r} file is missing remote_path")
            _safe_relative_path(remote_path)
            if remote_path in seen_paths:
                raise ValueError(
                    f"artifact bundle {name!r} repeats remote_path {remote_path!r}"
                )
            seen_paths.add(remote_path)

            if not isinstance(role, str) or not role:
                raise ValueError(
                    f"artifact bundle {name!r} file {remote_path!r} is missing role"
                )
            if role in seen_roles:
                raise ValueError(f"artifact bundle {name!r} repeats role {role!r}")
            seen_roles.add(role)

    return payload


def get_artifact_bundle(registry: dict, bundle_name: str) -> dict:
    bundles = registry["bundles"]
    if bundle_name not in bundles:
        raise KeyError(
            f"unknown artifact bundle {bundle_name!r}; "
            f"available={sorted(bundles)}"
        )
    return dict(bundles[bundle_name])


def resolve_bundle_revision(
    bundle: dict,
    *,
    revision_resolver: Callable[[str, str, str], str] = resolve_hf_revision,
) -> str:
    resolved = revision_resolver(
        str(bundle["repo_id"]),
        str(bundle["revision"]),
        str(bundle["repo_type"]),
    )
    if not isinstance(resolved, str) or len(resolved) != 40:
        raise RuntimeError(
            f"resolved Hugging Face revision must be a 40-character SHA, got {resolved!r}"
        )
    return resolved


def _download_hf_file(
    *,
    repo_id: str,
    filename: str,
    repo_type: str,
    revision: str,
) -> Path:
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            revision=revision,
            token=os.environ.get("HF_TOKEN") or None,
        )
    )


def provision_artifact_bundle(
    *,
    bundle_name: str,
    registry_path: Path = DEFAULT_REGISTRY,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    allow_unpinned: bool = False,
    force: bool = False,
    revision_resolver: Callable[[str, str, str], str] = resolve_hf_revision,
    download_file: Callable[..., Path] = _download_hf_file,
) -> dict:
    registry = load_artifact_registry(registry_path)
    bundle = get_artifact_bundle(registry, bundle_name)
    resolved_sha = resolve_bundle_revision(
        bundle,
        revision_resolver=revision_resolver,
    )
    expected_sha = bundle.get("expected_revision_sha")

    if expected_sha is None and not allow_unpinned:
        raise RuntimeError(
            f"artifact bundle {bundle_name!r} is not pinned: "
            "expected_revision_sha is null. Run with --resolve-only, record the "
            "resolved SHA in controlled_run/artifact_registry.json, commit it, "
            "and then provision the bundle."
        )
    if expected_sha is not None and resolved_sha != expected_sha:
        raise RuntimeError(
            f"artifact bundle {bundle_name!r} revision mismatch: "
            f"expected {expected_sha}, resolved {resolved_sha}"
        )

    destination = Path(output_root) / bundle_name
    manifest_path = destination / PROVISION_MANIFEST_NAME

    if manifest_path.exists() and not force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_sha = existing.get("resolved_revision_sha")
        if existing_sha != resolved_sha:
            raise RuntimeError(
                f"existing provisioned bundle uses revision {existing_sha}; "
                f"current registry resolves to {resolved_sha}. Use --force only "
                "after confirming the provenance change."
            )

    destination.mkdir(parents=True, exist_ok=True)
    file_records: dict[str, dict[str, object]] = {}

    for item in bundle["files"]:
        remote_path = str(item["remote_path"])
        relative = _safe_relative_path(remote_path)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)

        source = Path(
            download_file(
                repo_id=str(bundle["repo_id"]),
                filename=remote_path,
                repo_type=str(bundle["repo_type"]),
                revision=resolved_sha,
            )
        )
        if not source.is_file():
            raise FileNotFoundError(
                f"Hugging Face download did not produce expected file {remote_path!r}: "
                f"{source}"
            )
        shutil.copy2(source, target)

        file_records[remote_path] = {
            "role": str(item["role"]),
            "local_path": target.relative_to(destination).as_posix(),
            "sha256": sha256_file(target),
            "bytes": target.stat().st_size,
        }

    manifest = {
        "schema_version": 1,
        "bundle_name": bundle_name,
        "repo_type": str(bundle["repo_type"]),
        "repo_id": str(bundle["repo_id"]),
        "requested_revision": str(bundle["revision"]),
        "expected_revision_sha": expected_sha,
        "resolved_revision_sha": resolved_sha,
        "files": file_records,
    }
    write_json(manifest_path, manifest)
    return {
        "status": "READY",
        "bundle_name": bundle_name,
        "output_dir": str(destination),
        "resolved_revision_sha": resolved_sha,
        "files": len(file_records),
        "manifest": str(manifest_path),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve and provision pinned Hugging Face reference artifacts used "
            "by controlled analyses."
        )
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
    )
    parser.add_argument(
        "--bundle",
        default="canonical_grpo_seed42_h1_reference",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--resolve-only",
        action="store_true",
        help="Print the immutable HF SHA without downloading files.",
    )
    parser.add_argument(
        "--allow-unpinned",
        action="store_true",
        help=(
            "Permit provisioning when expected_revision_sha is null. "
            "Not recommended for controlled scientific inputs."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow replacing a previously provisioned bundle after provenance review.",
    )
    args = parser.parse_args(argv)

    registry = load_artifact_registry(args.registry)
    bundle = get_artifact_bundle(registry, args.bundle)

    if args.resolve_only:
        resolved = resolve_bundle_revision(bundle)
        print(
            json.dumps(
                {
                    "bundle_name": args.bundle,
                    "repo_type": bundle["repo_type"],
                    "repo_id": bundle["repo_id"],
                    "requested_revision": bundle["revision"],
                    "expected_revision_sha": bundle.get("expected_revision_sha"),
                    "resolved_revision_sha": resolved,
                    "pinned": bundle.get("expected_revision_sha") == resolved,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    result = provision_artifact_bundle(
        bundle_name=args.bundle,
        registry_path=args.registry,
        output_root=args.output_root,
        allow_unpinned=args.allow_unpinned,
        force=args.force,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
