from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

from huggingface_hub import HfApi


BUNDLE_DATE = "2026-09-05"
DEFAULT_MODEL_REPO = (
    "HKReporter/rlvr-behavior-probe-maxrl-canonical-seed42-2026-09-05"
)
DEFAULT_ANALYSIS_REPO = (
    "HKReporter/rlvr-behavior-probe-maxrl-analysis-seed42-2026-09-05"
)

TRAINING_EXECUTION_COMMIT = "981475795538eee391c7e86aa022ee609b539770"
SEQUENTIAL_EVALUATOR_COMMIT = "1c26b1f0f3c5f6ea1187fd00318587388a891272"
PI0_LINEAGE_ID = (
    "f89fc90226a67a6a3c7374f9c13abadfcecda88f397ab812fa4130f1f425605b"
)

MODEL_ROOT = Path("controlled_run_outputs/maxrl_canonical_seed42")
ANALYSIS_FOLDERS = {
    Path("controlled_run_outputs/maxrl_snapshot_eval_train256_k16_cbank"):
        "fixed_panel/maxrl_snapshot_eval_train256_k16_cbank",
    Path("controlled_run_outputs/maxrl_snapshot_crossfit"):
        "derived/maxrl_snapshot_crossfit",
    Path("controlled_run_outputs/maxrl_ledger_crossfit_signal"):
        "derived/maxrl_ledger_crossfit_signal",
    Path("controlled_run_outputs/maxrl_grpo_objective_comparison"):
        "derived/maxrl_grpo_objective_comparison",
}
DIAGNOSTIC_FOLDERS = (
    Path("controlled_run_outputs/_slow_evaluator_partial_for_parity_check"),
    Path("controlled_run_outputs/_batched_parity_pi005"),
    Path("controlled_run_outputs/_parity_fail_batched_pi005"),
)
PROVENANCE_FILES = (
    Path("docs/superpowers/specs/2026-09-03-maxrl-objective-intervention-amendment.md"),
    Path("docs/superpowers/specs/2026-09-03-maxrl-finite-g-signal-shape-pre-h1-outcome-addendum.md"),
    Path("docs/superpowers/checkpoints/2026-09-03-maxrl-h1-shakedown-postoutcome-gate.md"),
    Path("docs/superpowers/checkpoints/2026-09-04-maxrl-canonical-structural-pass.md"),
    Path("docs/superpowers/specs/2026-09-04-maxrl-canonical-fixed-panel-preoutcome-addendum.md"),
    Path("docs/superpowers/checkpoints/2026-09-04-maxrl-cbank-batching-parity-fail.md"),
    Path("docs/superpowers/checkpoints/2026-09-05-maxrl-h2-h3-postoutcome-gate.md"),
)
TRACKED_RESULT_FOLDERS = {
    Path("controlled_run_outputs/maxrl_snapshot_crossfit"):
        Path("analyses/canonical_maxrl_snapshot_crossfit"),
    Path("controlled_run_outputs/maxrl_ledger_crossfit_signal"):
        Path("analyses/canonical_maxrl_ledger_crossfit_signal"),
    Path("controlled_run_outputs/maxrl_grpo_objective_comparison"):
        Path("analyses/canonical_maxrl_grpo_objective_comparison"),
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=root,
        text=True,
    ).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(Path(root).rglob("*")):
        if path.is_file():
            yield path


def _entry(
    *,
    local_path: Path,
    remote_repo: str,
    repo_type: str,
    remote_path: str,
) -> dict:
    stat = local_path.stat()
    return {
        "local_path": str(local_path),
        "remote_repo": remote_repo,
        "repo_type": repo_type,
        "remote_path": remote_path,
        "size_bytes": int(stat.st_size),
        "sha256": sha256_file(local_path),
    }


def collect_backup_entries(
    root: Path,
    *,
    model_repo: str,
    analysis_repo: str,
) -> list[dict]:
    root = Path(root)
    model_root = root / MODEL_ROOT
    if not model_root.is_dir():
        raise FileNotFoundError(model_root)

    for source in ANALYSIS_FOLDERS:
        path = root / source
        if not path.is_dir():
            raise FileNotFoundError(path)

    entries: list[dict] = []

    for path in _iter_files(model_root):
        entries.append(
            _entry(
                local_path=path,
                remote_repo=model_repo,
                repo_type="model",
                remote_path=path.relative_to(model_root).as_posix(),
            )
        )

    for relative_source, remote_prefix in ANALYSIS_FOLDERS.items():
        source = root / relative_source
        for path in _iter_files(source):
            entries.append(
                _entry(
                    local_path=path,
                    remote_repo=analysis_repo,
                    repo_type="dataset",
                    remote_path=(
                        Path(remote_prefix) / path.relative_to(source)
                    ).as_posix(),
                )
            )

    seen_diagnostics: set[Path] = set()
    for relative_source in DIAGNOSTIC_FOLDERS:
        source = root / relative_source
        if not source.is_dir():
            continue
        resolved = source.resolve()
        if resolved in seen_diagnostics:
            continue
        seen_diagnostics.add(resolved)
        for path in _iter_files(source):
            entries.append(
                _entry(
                    local_path=path,
                    remote_repo=analysis_repo,
                    repo_type="dataset",
                    remote_path=(
                        Path("diagnostics") / source.name / path.relative_to(source)
                    ).as_posix(),
                )
            )

    for relative_path in PROVENANCE_FILES:
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        entries.append(
            _entry(
                local_path=path,
                remote_repo=analysis_repo,
                repo_type="dataset",
                remote_path=(Path("provenance") / path.name).as_posix(),
            )
        )

    reference_manifest = (
        root
        / "controlled_run_outputs/reference_inputs/"
        "canonical_grpo_seed42_h1_reference/provision_manifest.json"
    )
    if reference_manifest.is_file():
        entries.append(
            _entry(
                local_path=reference_manifest,
                remote_repo=analysis_repo,
                repo_type="dataset",
                remote_path=(
                    "provenance/canonical_grpo_seed42_reference_provision_manifest.json"
                ),
            )
        )

    for path in sorted((root / "controlled_run_outputs").glob("*maxrl*.log")):
        if path.is_file():
            entries.append(
                _entry(
                    local_path=path,
                    remote_repo=analysis_repo,
                    repo_type="dataset",
                    remote_path=(Path("logs") / path.name).as_posix(),
                )
            )

    keys = {
        (entry["remote_repo"], entry["repo_type"], entry["remote_path"])
        for entry in entries
    }
    if len(keys) != len(entries):
        raise ValueError("backup mapping contains duplicate remote paths")
    return entries


def prepare_tracked_results(root: Path) -> list[str]:
    root = Path(root)
    written: list[str] = []
    for relative_source, relative_destination in TRACKED_RESULT_FOLDERS.items():
        source = root / relative_source
        destination = root / relative_destination
        if not source.is_dir():
            raise FileNotFoundError(source)
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True, exist_ok=True)
        for path in _iter_files(source):
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            written.append(str(target.relative_to(root)))
    return written


def _write_manifest(
    root: Path,
    *,
    entries: list[dict],
    model_repo: str,
    analysis_repo: str,
) -> Path:
    bundle_dir = root / "hf_bundles/2026-09-05-canonical-maxrl-seed42"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    path = bundle_dir / "backup_manifest.json"

    payload = {
        "bundle_date": BUNDLE_DATE,
        "status": "local_hash_manifest_complete_remote_upload_not_yet_verified",
        "repository": "UnitedSnakes/rlvr_behavior_probe_runpod",
        "packaging_git_head": _git(root, "rev-parse", "HEAD"),
        "git_status_short": _git(root, "status", "--short"),
        "training_execution_commit": TRAINING_EXECUTION_COMMIT,
        "sequential_evaluator_implementation_commit": SEQUENTIAL_EVALUATOR_COMMIT,
        "pi0_lineage_id": PI0_LINEAGE_ID,
        "model_repo": model_repo,
        "analysis_repo": analysis_repo,
        "file_count": len(entries),
        "total_bytes": sum(int(entry["size_bytes"]) for entry in entries),
        "files": entries,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _upload_folder(
    api: HfApi,
    *,
    source: Path,
    repo_id: str,
    repo_type: str,
    path_in_repo: str | None = None,
) -> None:
    kwargs = {
        "folder_path": str(source),
        "repo_id": repo_id,
        "repo_type": repo_type,
    }
    if path_in_repo:
        kwargs["path_in_repo"] = path_in_repo
    api.upload_folder(**kwargs)


def _upload(
    root: Path,
    *,
    model_repo: str,
    analysis_repo: str,
    manifest_path: Path,
) -> tuple[str, str]:
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    api.whoami()

    api.create_repo(
        repo_id=model_repo,
        repo_type="model",
        private=True,
        exist_ok=True,
    )
    api.create_repo(
        repo_id=analysis_repo,
        repo_type="dataset",
        private=True,
        exist_ok=True,
    )

    _upload_folder(
        api,
        source=root / MODEL_ROOT,
        repo_id=model_repo,
        repo_type="model",
    )

    for relative_source, remote_prefix in ANALYSIS_FOLDERS.items():
        _upload_folder(
            api,
            source=root / relative_source,
            repo_id=analysis_repo,
            repo_type="dataset",
            path_in_repo=remote_prefix,
        )

    seen_diagnostics: set[Path] = set()
    for relative_source in DIAGNOSTIC_FOLDERS:
        source = root / relative_source
        if not source.is_dir():
            continue
        resolved = source.resolve()
        if resolved in seen_diagnostics:
            continue
        seen_diagnostics.add(resolved)
        _upload_folder(
            api,
            source=source,
            repo_id=analysis_repo,
            repo_type="dataset",
            path_in_repo=f"diagnostics/{source.name}",
        )

    for relative_path in PROVENANCE_FILES:
        path = root / relative_path
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=f"provenance/{path.name}",
            repo_id=analysis_repo,
            repo_type="dataset",
        )

    reference_manifest = (
        root
        / "controlled_run_outputs/reference_inputs/"
        "canonical_grpo_seed42_h1_reference/provision_manifest.json"
    )
    if reference_manifest.is_file():
        api.upload_file(
            path_or_fileobj=str(reference_manifest),
            path_in_repo=(
                "provenance/canonical_grpo_seed42_reference_provision_manifest.json"
            ),
            repo_id=analysis_repo,
            repo_type="dataset",
        )

    for path in sorted((root / "controlled_run_outputs").glob("*maxrl*.log")):
        if path.is_file():
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=f"logs/{path.name}",
                repo_id=analysis_repo,
                repo_type="dataset",
            )

    api.upload_file(
        path_or_fileobj=str(manifest_path),
        path_in_repo="backup_manifest.json",
        repo_id=analysis_repo,
        repo_type="dataset",
    )

    model_info = api.repo_info(repo_id=model_repo, repo_type="model")
    analysis_info = api.repo_info(repo_id=analysis_repo, repo_type="dataset")
    return str(model_info.sha), str(analysis_info.sha)


def _verify_remote(
    *,
    entries: list[dict],
    model_repo: str,
    analysis_repo: str,
) -> dict:
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    verified = 0
    sha_verified = 0
    size_verified = 0
    failures: list[str] = []

    grouped = {
        (model_repo, "model"): [
            entry for entry in entries
            if entry["remote_repo"] == model_repo and entry["repo_type"] == "model"
        ],
        (analysis_repo, "dataset"): [
            entry for entry in entries
            if entry["remote_repo"] == analysis_repo
            and entry["repo_type"] == "dataset"
        ],
    }

    for (repo_id, repo_type), expected in grouped.items():
        info = api.repo_info(
            repo_id=repo_id,
            repo_type=repo_type,
            files_metadata=True,
        )
        remote = {
            sibling.rfilename: sibling
            for sibling in info.siblings
        }
        for entry in expected:
            remote_path = entry["remote_path"]
            sibling = remote.get(remote_path)
            if sibling is None:
                failures.append(f"{repo_id}: missing {remote_path}")
                continue

            remote_size = getattr(sibling, "size", None)
            lfs = getattr(sibling, "lfs", None)
            if remote_size is None and lfs is not None:
                remote_size = getattr(lfs, "size", None)
                if remote_size is None and isinstance(lfs, dict):
                    remote_size = lfs.get("size")
            if remote_size is not None:
                if int(remote_size) != int(entry["size_bytes"]):
                    failures.append(
                        f"{repo_id}: size mismatch {remote_path}: "
                        f"{remote_size} != {entry['size_bytes']}"
                    )
                    continue
                size_verified += 1

            remote_sha = None
            if lfs is not None:
                remote_sha = getattr(lfs, "sha256", None)
                if remote_sha is None and isinstance(lfs, dict):
                    remote_sha = lfs.get("sha256")
            if remote_sha:
                if str(remote_sha) != entry["sha256"]:
                    failures.append(
                        f"{repo_id}: LFS sha256 mismatch {remote_path}"
                    )
                    continue
                sha_verified += 1

            verified += 1

    if failures:
        raise RuntimeError("remote verification failed:\n" + "\n".join(failures[:50]))

    return {
        "expected_files": len(entries),
        "verified_present": verified,
        "size_verified": size_verified,
        "lfs_sha256_verified": sha_verified,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Package and back up the canonical MaxRL seed42 result."
    )
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--analysis-repo", default=DEFAULT_ANALYSIS_REPO)
    parser.add_argument(
        "--prepare-git-results",
        action="store_true",
        help="Copy lightweight derived outputs from controlled_run_outputs into tracked analyses/ directories.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Create private Hugging Face repos, upload all mapped artifacts, and verify the remote copy.",
    )
    args = parser.parse_args(argv)

    root = _repo_root()

    if args.prepare_git_results:
        written = prepare_tracked_results(root)
        print(f"prepared tracked lightweight results: {len(written)} files")

    print("hashing backup set...")
    entries = collect_backup_entries(
        root,
        model_repo=args.model_repo,
        analysis_repo=args.analysis_repo,
    )
    manifest_path = _write_manifest(
        root,
        entries=entries,
        model_repo=args.model_repo,
        analysis_repo=args.analysis_repo,
    )

    print(f"backup files={len(entries)}")
    print(f"backup bytes={sum(int(e['size_bytes']) for e in entries)}")
    print(f"manifest={manifest_path.relative_to(root)}")
    print(f"model_repo={args.model_repo}")
    print(f"analysis_repo={args.analysis_repo}")

    if not args.upload:
        print("LOCAL BACKUP MANIFEST: PASS")
        print("REMOTE UPLOAD: NOT REQUESTED")
        return

    model_sha, analysis_sha = _upload(
        root,
        model_repo=args.model_repo,
        analysis_repo=args.analysis_repo,
        manifest_path=manifest_path,
    )
    verification = _verify_remote(
        entries=entries,
        model_repo=args.model_repo,
        analysis_repo=args.analysis_repo,
    )

    upload_record = {
        "status": "REMOTE_BACKUP_VERIFIED",
        "bundle_date": BUNDLE_DATE,
        "model_repo": args.model_repo,
        "model_repo_commit": model_sha,
        "analysis_repo": args.analysis_repo,
        "analysis_repo_commit": analysis_sha,
        "verification": verification,
        "local_manifest": str(manifest_path.relative_to(root)),
    }
    record_path = manifest_path.parent / "upload_record.json"
    record_path.write_text(
        json.dumps(upload_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(upload_record, indent=2, sort_keys=True))
    print("MAXRL REMOTE BACKUP: VERIFIED")


if __name__ == "__main__":
    main()
