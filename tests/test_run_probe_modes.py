import sys
from types import SimpleNamespace

import pytest

import run_probe


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
