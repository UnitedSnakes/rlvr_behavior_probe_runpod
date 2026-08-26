import json
import sys

import run_probe


def test_run_config_records_runtime_metadata(monkeypatch, tmp_path):
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
        "collect_runtime_metadata",
        lambda engine, device: {
            "platform": "cuda",
            "implementation": "vllm-cuda" if engine == "vllm" else "transformers",
        },
    )
    monkeypatch.setattr(
        run_probe,
        "prepare_questions",
        lambda path, questions, seed: [
            {"qid": 0, "question": "1+1?", "gold": "2"}
        ],
    )
    monkeypatch.setattr(run_probe, "run_one_checkpoint", lambda **kwargs: None)

    run_probe.main()

    config = json.loads((tmp_path / "run_config.json").read_text())
    assert config["runtime"] == {
        "platform": "cuda",
        "implementation": "transformers",
    }
