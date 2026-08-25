import os
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
