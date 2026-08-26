import sys
from types import SimpleNamespace

import run_probe


def test_sampling_cli_defaults_are_canonical(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_probe.py"])

    args = run_probe.parse_args()

    assert args.temperature == 1.0
    assert args.top_p == 0.95
    assert args.top_k == 0
    assert args.repetition_penalty == 1.0


def test_sampling_cli_allows_explicit_hf_like_override(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_probe.py",
            "--top-k",
            "20",
            "--repetition-penalty",
            "1.1",
        ],
    )

    args = run_probe.parse_args()

    assert args.top_k == 20
    assert args.repetition_penalty == 1.1


def test_run_one_checkpoint_forwards_full_sampling_policy(monkeypatch, tmp_path):
    sample_calls = []

    class FakeSampler:
        def sample(self, **kwargs):
            sample_calls.append(kwargs)
            return ["The answer is \\boxed{2}."]

    monkeypatch.setattr(run_probe, "build_sampler", lambda **kwargs: FakeSampler())
    monkeypatch.setattr(run_probe, "append_jsonl", lambda path, row: None)

    args = SimpleNamespace(
        engine="vllm",
        resume=False,
        seed=42,
        rollouts=1,
        batch_rollouts=4,
        max_new_tokens=128,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        repetition_penalty=1.1,
    )

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

    assert sample_calls == [
        {
            "question": "1+1?",
            "n": 1,
            "batch_rollouts": 4,
            "max_new_tokens": 128,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "repetition_penalty": 1.1,
            "seed": 4200000,
        }
    ]
