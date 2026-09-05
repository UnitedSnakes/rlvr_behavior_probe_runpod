from pathlib import Path

from controlled_run import backup_maxrl_seed42 as backup


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_prepare_tracked_results_copies_all_files(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "controlled_run_outputs/maxrl_snapshot_crossfit"
    _write(source / "a.csv", b"a")
    _write(source / "nested/b.json", b"b")

    monkeypatch.setattr(
        backup,
        "TRACKED_RESULT_FOLDERS",
        {
            Path("controlled_run_outputs/maxrl_snapshot_crossfit"):
                Path("analyses/canonical_maxrl_snapshot_crossfit")
        },
    )

    written = backup.prepare_tracked_results(tmp_path)

    assert sorted(written) == [
        "analyses/canonical_maxrl_snapshot_crossfit/a.csv",
        "analyses/canonical_maxrl_snapshot_crossfit/nested/b.json",
    ]
    assert (
        tmp_path / "analyses/canonical_maxrl_snapshot_crossfit/a.csv"
    ).read_bytes() == b"a"


def test_collect_backup_entries_maps_model_and_analysis_paths(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "controlled_run_outputs/maxrl_canonical_seed42/pi_005/model.safetensors",
        b"model",
    )
    _write(
        tmp_path / "controlled_run_outputs/maxrl_snapshot_eval_train256_k16_cbank/pi_005/snapshot_raw.jsonl",
        b"raw\n",
    )

    monkeypatch.setattr(
        backup,
        "ANALYSIS_FOLDERS",
        {
            Path("controlled_run_outputs/maxrl_snapshot_eval_train256_k16_cbank"):
                "fixed_panel/maxrl_snapshot_eval_train256_k16_cbank",
        },
    )
    monkeypatch.setattr(backup, "DIAGNOSTIC_FOLDERS", ())
    monkeypatch.setattr(backup, "PROVENANCE_FILES", ())

    entries = backup.collect_backup_entries(
        tmp_path,
        model_repo="owner/model",
        analysis_repo="owner/data",
    )

    simplified = {
        (entry["remote_repo"], entry["repo_type"], entry["remote_path"])
        for entry in entries
    }
    assert simplified == {
        ("owner/model", "model", "pi_005/model.safetensors"),
        (
            "owner/data",
            "dataset",
            "fixed_panel/maxrl_snapshot_eval_train256_k16_cbank/pi_005/snapshot_raw.jsonl",
        ),
    }
    assert all(len(entry["sha256"]) == 64 for entry in entries)
