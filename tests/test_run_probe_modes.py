import json
import os
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import run_probe


def test_upload_repo_defaults_to_none(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_probe.py"])

    args = run_probe.parse_args()

    assert args.upload_repo is None


def test_no_upload_records_start_time_and_never_calls_uploader(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_probe.py", "--only-rl", "--result-dir", str(tmp_path)],
    )
    started_at = datetime(2026, 8, 25, 23, 53, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(run_probe, "utc_now", lambda: started_at)
    monkeypatch.setattr(run_probe, "set_seed", lambda seed: None)
    monkeypatch.setattr(run_probe, "resolve_device", lambda device: "cuda")
    monkeypatch.setattr(run_probe, "resolve_dtype", lambda dtype: "bfloat16")
    monkeypatch.setattr(
        run_probe,
        "prepare_questions",
        lambda path, questions, seed: [
            {"qid": 0, "question": "1+1?", "gold": "2"}
        ],
    )
    monkeypatch.setattr(run_probe, "run_one_checkpoint", lambda **kwargs: None)

    upload_calls = []
    monkeypatch.setattr(
        run_probe,
        "upload_result_dir",
        lambda **kwargs: upload_calls.append(kwargs),
    )

    run_probe.main()

    config = json.loads((tmp_path / "run_config.json").read_text())
    assert config["run_started_at"] == "2026-08-25T23:53:12Z"
    assert config["upload_repo"] is None
    assert config["upload_path"] is None
    assert upload_calls == []


def _patch_upload_run(monkeypatch, tmp_path, mode="--only-rl"):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_probe.py",
            mode,
            "--result-dir",
            str(tmp_path),
            "--upload-repo",
            "UnitedSnakes/rlvr-behavior-probe-results",
        ],
    )
    started_at = datetime(2026, 8, 25, 23, 53, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(run_probe, "utc_now", lambda: started_at)
    monkeypatch.setattr(run_probe, "set_seed", lambda seed: None)
    monkeypatch.setattr(run_probe, "resolve_device", lambda device: "cuda")
    monkeypatch.setattr(run_probe, "resolve_dtype", lambda dtype: "bfloat16")
    monkeypatch.setattr(
        run_probe,
        "prepare_questions",
        lambda path, questions, seed: [
            {"qid": 0, "question": "1+1?", "gold": "2"}
        ],
    )
    monkeypatch.setattr(
        run_probe,
        "resolve_checkpoint_revision",
        lambda model_name, revision: "resolved-sft",
    )


def test_requested_upload_runs_after_config_is_written(monkeypatch, tmp_path):
    _patch_upload_run(monkeypatch, tmp_path)

    def fake_run_one_checkpoint(**kwargs):
        kwargs["out_path"].write_text("local result\n", encoding="utf-8")

    monkeypatch.setattr(run_probe, "run_one_checkpoint", fake_run_one_checkpoint)
    upload_calls = []

    def fake_upload_result_dir(result_dir, repo_id, remote_path):
        assert (result_dir / "run_config.json").exists()
        assert (result_dir / "rl_raw.jsonl").exists()
        upload_calls.append((result_dir, repo_id, remote_path))

    monkeypatch.setattr(run_probe, "upload_result_dir", fake_upload_result_dir)

    run_probe.main()

    expected_path = f"runs/20260825T235312Z-{tmp_path.name}"
    config = json.loads((tmp_path / "run_config.json").read_text())
    assert config["upload_path"] == expected_path
    assert upload_calls == [
        (
            tmp_path,
            "UnitedSnakes/rlvr-behavior-probe-results",
            expected_path,
        )
    ]


def test_upload_failure_preserves_local_results_and_fails_run(monkeypatch, tmp_path):
    _patch_upload_run(monkeypatch, tmp_path)

    def fake_run_one_checkpoint(**kwargs):
        kwargs["out_path"].write_text("local result\n", encoding="utf-8")

    monkeypatch.setattr(run_probe, "run_one_checkpoint", fake_run_one_checkpoint)

    def fail_upload(result_dir, repo_id, remote_path):
        raise RuntimeError("network down")

    monkeypatch.setattr(run_probe, "upload_result_dir", fail_upload)

    with pytest.raises(RuntimeError, match="local results.*backup failed"):
        run_probe.main()

    assert (tmp_path / "rl_raw.jsonl").exists()
    assert (tmp_path / "run_config.json").exists()


@pytest.mark.parametrize(
    ("mode", "expected_file", "absent_file"),
    [
        ("--only-sft", "sft_raw.jsonl", "rl_raw.jsonl"),
        ("--only-rl", "rl_raw.jsonl", "sft_raw.jsonl"),
    ],
)
def test_single_checkpoint_mode_can_upload_without_paired_output(
    monkeypatch,
    tmp_path,
    mode,
    expected_file,
    absent_file,
):
    _patch_upload_run(monkeypatch, tmp_path, mode)

    def fake_run_one_checkpoint(**kwargs):
        kwargs["out_path"].write_text("local result\n", encoding="utf-8")

    monkeypatch.setattr(run_probe, "run_one_checkpoint", fake_run_one_checkpoint)
    upload_calls = []

    def fake_upload_result_dir(result_dir, repo_id, remote_path):
        assert (result_dir / expected_file).exists()
        assert not (result_dir / absent_file).exists()
        upload_calls.append(remote_path)

    monkeypatch.setattr(run_probe, "upload_result_dir", fake_upload_result_dir)

    run_probe.main()

    assert upload_calls == [f"runs/20260825T235312Z-{tmp_path.name}"]


def test_only_sft_and_only_rl_are_mutually_exclusive(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_probe.py", "--only-sft", "--only-rl"],
    )

    with pytest.raises(SystemExit) as error:
        run_probe.parse_args()

    assert error.value.code == 2


def test_only_rl_skips_sft_resolution_and_sampling(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_probe.py",
            "--only-rl",
            "--result-dir",
            str(tmp_path),
        ],
    )

    monkeypatch.setattr(run_probe, "set_seed", lambda seed: None)
    monkeypatch.setattr(run_probe, "resolve_device", lambda device: "cuda")
    monkeypatch.setattr(run_probe, "resolve_dtype", lambda dtype: "bfloat16")
    monkeypatch.setattr(
        run_probe,
        "prepare_questions",
        lambda path, questions, seed: [
            {"qid": 0, "question": "1+1?", "gold": "2"}
        ],
    )

    resolved_models = []

    def fake_resolve_checkpoint_revision(model_name, revision):
        resolved_models.append((model_name, revision))
        return "resolved-sft"

    monkeypatch.setattr(
        run_probe,
        "resolve_checkpoint_revision",
        fake_resolve_checkpoint_revision,
    )

    runs = []

    def fake_run_one_checkpoint(**kwargs):
        runs.append(SimpleNamespace(**kwargs))

    monkeypatch.setattr(run_probe, "run_one_checkpoint", fake_run_one_checkpoint)

    run_probe.main()

    assert resolved_models == []
    assert [run.alias for run in runs] == ["rl"]
    assert runs[0].revision == "main"
    assert runs[0].out_path == tmp_path / "rl_raw.jsonl"


def test_vllm_runtime_sets_spawn_without_global_cuda_seed(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_probe.py",
            "--engine",
            "vllm",
            "--only-rl",
            "--result-dir",
            str(tmp_path),
        ],
    )
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)

    seed_calls = []
    monkeypatch.setattr(run_probe, "set_seed", lambda seed: seed_calls.append(seed))
    monkeypatch.setattr(run_probe, "resolve_device", lambda device: "cuda")
    monkeypatch.setattr(run_probe, "resolve_dtype", lambda dtype: "bfloat16")
    monkeypatch.setattr(
        run_probe,
        "prepare_questions",
        lambda path, questions, seed: [
            {"qid": 0, "question": "1+1?", "gold": "2"}
        ],
    )
    monkeypatch.setattr(run_probe, "run_one_checkpoint", lambda **kwargs: None)

    run_probe.main()

    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    assert seed_calls == []


def test_vllm_checkpoint_run_does_not_touch_parent_cuda_cache(monkeypatch, tmp_path):
    args = SimpleNamespace(
        engine="vllm",
        resume=False,
        seed=42,
        rollouts=1,
        batch_rollouts=4,
        max_new_tokens=128,
        temperature=1.0,
        top_p=0.95,
    )

    class FakeSampler:
        def sample(self, **kwargs):
            return ["The answer is \\boxed{2}."]

    monkeypatch.setattr(run_probe, "build_sampler", lambda **kwargs: FakeSampler())
    monkeypatch.setattr(run_probe, "append_jsonl", lambda path, row: None)

    cache_calls = []
    monkeypatch.setattr(run_probe, "empty_device_cache", lambda: cache_calls.append(True))

    run_probe.run_one_checkpoint(
        alias="rl",
        model_name="example/model",
        revision="main",
        questions=[{"qid": 0, "question": "1+1?", "gold": "2"}],
        out_path=tmp_path / "rl_raw.jsonl",
        args=args,
        device="cuda",
        dtype="bfloat16",
    )

    assert cache_calls == []
