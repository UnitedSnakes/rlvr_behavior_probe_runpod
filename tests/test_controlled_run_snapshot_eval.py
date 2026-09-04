from __future__ import annotations

import json
from pathlib import Path

import pytest

from controlled_run.config import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "controlled_run/configs/grpo_qwen3_0_6b.yaml"


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _make_canonical_snapshot(tmp_path: Path, pct: int = 25) -> Path:
    run_dir = tmp_path / "grpo_canonical"
    lineage = "canonical-lineage"
    step = 934
    _write_json(
        run_dir / "grpo_run_manifest.json",
        {
            "mode": "canonical",
            "scientific_use": True,
            "pi0_lineage_id": lineage,
            "gsm8k_dataset_sha": "gsm8k-sha",
        },
    )
    _write_json(
        run_dir / "policy_snapshot_schedule.json",
        {
            "max_steps": 3736,
            "percentage_to_step": {str(pct): step},
            "pi0_lineage_id": lineage,
        },
    )
    policy_dir = run_dir / f"pi_{pct:03d}"
    policy_dir.mkdir(parents=True)
    _write_json(
        policy_dir / "policy_metadata.json",
        {
            "actual_step": step,
            "target_percentage": pct,
            "pi0_lineage_id": lineage,
        },
    )
    (policy_dir / "config.json").write_text("{}", encoding="utf-8")
    return run_dir


def test_resolve_snapshot_policy_accepts_canonical_maxrl_manifest(tmp_path):
    import controlled_run.eval_snapshot as eval_snapshot

    run_dir = tmp_path / "maxrl_canonical"
    lineage = "canonical-lineage"
    pct = 25
    step = 934
    _write_json(
        run_dir / "maxrl_run_manifest.json",
        {
            "mode": "canonical",
            "scientific_use": True,
            "pi0_lineage_id": lineage,
            "gsm8k_dataset_sha": "gsm8k-sha",
            "objective": {
                "objective_family": "MaxRL",
                "advantage_estimator": "practical_maxrl",
            },
        },
    )
    _write_json(
        run_dir / "policy_snapshot_schedule.json",
        {
            "max_steps": 3736,
            "percentage_to_step": {str(pct): step},
            "pi0_lineage_id": lineage,
        },
    )
    policy_dir = run_dir / f"pi_{pct:03d}"
    policy_dir.mkdir(parents=True)
    _write_json(
        policy_dir / "policy_metadata.json",
        {
            "actual_step": step,
            "target_percentage": pct,
            "pi0_lineage_id": lineage,
        },
    )
    (policy_dir / "config.json").write_text("{}", encoding="utf-8")

    resolved = eval_snapshot.resolve_snapshot_policy(run_dir, pct)

    assert resolved["policy_dir"] == policy_dir
    assert resolved["canonical_manifest_filename"] == "maxrl_run_manifest.json"
    assert resolved["objective_family"] == "MaxRL"


def test_resolve_snapshot_policy_fail_closes_on_lineage_and_schedule(tmp_path):
    import controlled_run.eval_snapshot as eval_snapshot

    run_dir = _make_canonical_snapshot(tmp_path)
    resolved = eval_snapshot.resolve_snapshot_policy(run_dir, 25)

    assert resolved["policy_dir"] == run_dir / "pi_025"
    assert resolved["actual_step"] == 934
    assert resolved["target_percentage"] == 25
    assert resolved["pi0_lineage_id"] == "canonical-lineage"

    metadata_path = run_dir / "pi_025" / "policy_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["pi0_lineage_id"] = "wrong-lineage"
    _write_json(metadata_path, metadata)
    with pytest.raises(ValueError, match="lineage"):
        eval_snapshot.resolve_snapshot_policy(run_dir, 25)

    metadata["pi0_lineage_id"] = "canonical-lineage"
    metadata["actual_step"] = 999
    _write_json(metadata_path, metadata)
    with pytest.raises(ValueError, match="schedule"):
        eval_snapshot.resolve_snapshot_policy(run_dir, 25)


def test_resolve_panel_fixes_train256_and_requires_explicit_heldout_indices():
    import controlled_run.eval_snapshot as eval_snapshot

    train = eval_snapshot.resolve_panel("train256")
    assert train == {"name": "train256", "split": "train", "indices": list(range(256))}

    heldout = eval_snapshot.resolve_panel("heldout", [9, 2, 7])
    assert heldout == {"name": "heldout", "split": "test", "indices": [2, 7, 9]}

    with pytest.raises(ValueError, match="explicit"):
        eval_snapshot.resolve_panel("heldout")
    with pytest.raises(ValueError, match="duplicates"):
        eval_snapshot.resolve_panel("heldout", [2, 2])
    with pytest.raises(ValueError, match="panel"):
        eval_snapshot.resolve_panel("other", [1])


def test_score_snapshot_completions_uses_token_id_termination_and_canonical_reward(monkeypatch):
    import controlled_run.eval_snapshot as eval_snapshot

    monkeypatch.setattr(
        eval_snapshot,
        "gsm8k_binary_reward",
        lambda completions, answer: [1.0, 1.0, 0.0],
    )

    scored = eval_snapshot.score_snapshot_completions(
        texts=["correct terminated", "correct capped", "wrong terminated"],
        token_ids=[[10, 151643], [20, 21], [30, 151643]],
        finish_reasons=["stop", "length", "stop"],
        answer="42",
        terminal_token_ids=(151643,),
    )

    assert scored["n_rollouts"] == 3
    assert scored["n_correct"] == 2
    assert scored["n_terminated"] == 2
    assert scored["n_reward"] == 1
    assert scored["p_correct"] == pytest.approx(2 / 3)
    assert scored["p_terminated"] == pytest.approx(2 / 3)
    assert scored["p_reward"] == pytest.approx(1 / 3)
    assert scored["rollouts"] == [
        {
            "rollout": 0,
            "correct": True,
            "terminated": True,
            "canonical_reward": 1,
            "completion_length": 2,
            "finish_reason": "stop",
            "token_ids": [10, 151643],
            "text": "correct terminated",
        },
        {
            "rollout": 1,
            "correct": True,
            "terminated": False,
            "canonical_reward": 0,
            "completion_length": 2,
            "finish_reason": "length",
            "token_ids": [20, 21],
            "text": "correct capped",
        },
        {
            "rollout": 2,
            "correct": False,
            "terminated": True,
            "canonical_reward": 0,
            "completion_length": 2,
            "finish_reason": "stop",
            "token_ids": [30, 151643],
            "text": "wrong terminated",
        },
    ]


def test_build_snapshot_manifest_records_policy_panel_sampling_and_dataset_provenance(tmp_path):
    import controlled_run.eval_snapshot as eval_snapshot

    config_path = tmp_path / "grpo.yaml"
    config_path.write_text("seed: 42\n", encoding="utf-8")

    manifest = eval_snapshot.build_snapshot_manifest(
        canonical_run_dir=tmp_path / "grpo_canonical",
        snapshot={
            "policy_dir": tmp_path / "grpo_canonical/pi_025",
            "actual_step": 934,
            "target_percentage": 25,
            "pi0_lineage_id": "canonical-lineage",
        },
        panel={"name": "train256", "split": "train", "indices": list(range(256))},
        dataset_sha="gsm8k-sha",
        dataset_config="main",
        config_path=config_path,
        sampling_settings={"num_generations": 32, "temperature": 0.8},
        prompt_audit={"count": 256, "limit": 512},
        runtime={"gpu_name": "NVIDIA A40"},
        request_batch_size=32,
    )

    assert manifest["mode"] == "canonical_snapshot_eval"
    assert manifest["snapshot"]["target_percentage"] == 25
    assert manifest["snapshot"]["actual_step"] == 934
    assert manifest["pi0_lineage_id"] == "canonical-lineage"
    assert manifest["panel"]["name"] == "train256"
    assert manifest["panel"]["split"] == "train"
    assert manifest["panel"]["indices"] == list(range(256))
    assert manifest["dataset"] == {
        "name": "openai/gsm8k",
        "config": "main",
        "split": "train",
        "sha": "gsm8k-sha",
    }
    assert manifest["sampling"]["num_generations"] == 32
    assert manifest["request_batch_size"] == 32
    assert manifest["grpo_config_sha256"] == eval_snapshot.sha256_file(config_path)


def test_snapshot_sampling_batches_requests_without_changing_per_question_seeds(
    monkeypatch,
    tmp_path,
):
    import controlled_run.eval_snapshot as eval_snapshot

    config = load_config(CONFIG_PATH)
    settings = eval_snapshot.canonical_sampling_settings(
        config,
        num_generations_override=2,
    )
    captured = {"calls": []}

    class FakeTokenizer:
        eos_token_id = 151643
        pad_token_id = 151643

        def apply_chat_template(self, prompt, tokenize, add_generation_prompt):
            assert tokenize is True
            assert add_generation_prompt is True
            return [101, int(prompt[-1]["content"])]

    class FakeTokensPrompt:
        def __init__(self, prompt_token_ids):
            self.prompt_token_ids = prompt_token_ids

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeOutput:
        def __init__(self, text, token_ids, finish_reason):
            self.text = text
            self.token_ids = token_ids
            self.finish_reason = finish_reason

    class FakeRequestOutput:
        def __init__(self, marker):
            self.outputs = [
                FakeOutput(f"correct-{marker}-0", [1, 151643], "stop"),
                FakeOutput(f"correct-{marker}-1", [2, 151643], "stop"),
            ]

    class FakeLLM:
        def generate(self, prompts, sampling_params, use_tqdm):
            assert use_tqdm is False
            captured["calls"].append(
                {
                    "prompts": [item.prompt_token_ids for item in prompts],
                    "params": [item.kwargs for item in sampling_params],
                }
            )
            return [
                FakeRequestOutput(prompt.prompt_token_ids[-1])
                for prompt in prompts
            ]

    monkeypatch.setattr(
        eval_snapshot,
        "gsm8k_binary_reward",
        lambda texts, answer: [1.0, 1.0],
    )

    output_path = tmp_path / "snapshot_raw.jsonl"
    eval_snapshot.sample_snapshot_rows(
        llm=FakeLLM(),
        tokenizer=FakeTokenizer(),
        indexed_rows=[
            (
                17,
                {
                    "prompt": [{"role": "user", "content": "17"}],
                    "answer": "7",
                },
            ),
            (
                18,
                {
                    "prompt": [{"role": "user", "content": "18"}],
                    "answer": "8",
                },
            ),
            (
                19,
                {
                    "prompt": [{"role": "user", "content": "19"}],
                    "answer": "9",
                },
            ),
        ],
        settings=settings,
        output_path=output_path,
        sampling_params_cls=FakeSamplingParams,
        tokens_prompt_cls=FakeTokensPrompt,
        request_batch_size=2,
    )

    assert len(captured["calls"]) == 2
    assert captured["calls"][0]["prompts"] == [[101, 17], [101, 18]]
    assert captured["calls"][1]["prompts"] == [[101, 19]]

    params = [
        item
        for call in captured["calls"]
        for item in call["params"]
    ]
    assert [item["seed"] for item in params] == [
        4_275_017,
        4_275_018,
        4_275_019,
    ]
    for item in params:
        assert item["n"] == 2
        assert item["temperature"] == config["temperature"]
        assert item["top_p"] == config["top_p"]
        assert item["top_k"] == config["top_k"]
        assert item["repetition_penalty"] == config["repetition_penalty"]
        assert item["max_tokens"] == config["max_completion_length"]

    records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["dataset_index"] for row in records] == [17, 18, 19]
    assert [row["question_seed"] for row in records] == [
        4_275_017,
        4_275_018,
        4_275_019,
    ]
    assert all(row["n_rollouts"] == 2 for row in records)


def test_snapshot_sampling_rejects_nonpositive_request_batch_size(tmp_path):
    import controlled_run.eval_snapshot as eval_snapshot

    with pytest.raises(ValueError, match="request_batch_size"):
        eval_snapshot.sample_snapshot_rows(
            llm=object(),
            tokenizer=object(),
            indexed_rows=[],
            settings={},
            output_path=tmp_path / "out.jsonl",
            sampling_params_cls=object,
            tokens_prompt_cls=object,
            request_batch_size=0,
        )
