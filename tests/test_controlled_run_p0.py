from __future__ import annotations

import json
from pathlib import Path

import pytest

from controlled_run.config import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "controlled_run/configs/grpo_qwen3_0_6b.yaml"


def test_canonical_sampling_settings_are_derived_from_grpo_config():
    import controlled_run.sample_p0 as sample_p0

    config = load_config(CONFIG_PATH)
    settings = sample_p0.canonical_sampling_settings(config)

    assert settings == {
        "num_generations": config["num_generations"],
        "temperature": config["temperature"],
        "top_p": config["top_p"],
        "top_k": config["top_k"],
        "repetition_penalty": config["repetition_penalty"],
        "max_completion_length": config["max_completion_length"],
        "max_prompt_tokens": config["max_prompt_tokens"],
        "vllm_max_model_length": config["vllm_max_model_length"],
        "seed": config["seed"],
    }


def test_canonical_sampling_settings_accepts_num_generations_override():
    import controlled_run.sample_p0 as sample_p0

    config = load_config(CONFIG_PATH)
    settings = sample_p0.canonical_sampling_settings(
        config, num_generations_override=1024
    )

    assert settings["num_generations"] == 1024
    assert settings["temperature"] == config["temperature"]


def test_select_indexed_rows_sorts_dedupes_and_bounds_checks():
    import controlled_run.sample_p0 as sample_p0

    rows = [{"x": index} for index in range(10)]

    assert sample_p0.select_indexed_rows(rows, [7, 2, 5]) == [
        (2, {"x": 2}),
        (5, {"x": 5}),
        (7, {"x": 7}),
    ]

    with pytest.raises(ValueError, match="non-empty"):
        sample_p0.select_indexed_rows(rows, [])
    with pytest.raises(ValueError, match="duplicates"):
        sample_p0.select_indexed_rows(rows, [3, 3])
    with pytest.raises(ValueError, match="out of range"):
        sample_p0.select_indexed_rows(rows, [10])
    with pytest.raises(ValueError, match="out of range"):
        sample_p0.select_indexed_rows(rows, [-1])


def test_run_p0_rejects_dataset_indices_combined_with_start_end(tmp_path):
    import controlled_run.sample_p0 as sample_p0

    with pytest.raises(ValueError, match="cannot be combined"):
        sample_p0.run_p0(
            policy_dir=tmp_path / "does-not-exist",
            output_dir=tmp_path / "out",
            start_index=5,
            dataset_indices=[1, 2, 3],
        )


def test_slice_shard_is_inclusive_exclusive_and_bounds_checked():
    import controlled_run.sample_p0 as sample_p0

    rows = [{"x": index} for index in range(6)]
    assert sample_p0.slice_shard(rows, 2, 5) == [
        (2, {"x": 2}),
        (3, {"x": 3}),
        (4, {"x": 4}),
    ]
    assert sample_p0.slice_shard(rows, 0, None)[-1][0] == 5

    with pytest.raises(ValueError, match="start_index"):
        sample_p0.slice_shard(rows, -1, 2)
    with pytest.raises(ValueError, match="end_index"):
        sample_p0.slice_shard(rows, 2, 7)
    with pytest.raises(ValueError, match="greater than"):
        sample_p0.slice_shard(rows, 4, 4)


def test_question_seed_depends_on_original_dataset_index():
    import controlled_run.sample_p0 as sample_p0

    assert sample_p0.question_seed(42, 17) == 4_200_017
    assert sample_p0.question_seed(42, 17) == sample_p0.question_seed(42, 17)
    assert sample_p0.question_seed(42, 18) != sample_p0.question_seed(42, 17)


def test_prepare_p0_rows_reuses_controlled_prompt_builder_and_preflight(monkeypatch):
    import controlled_run.sample_p0 as sample_p0

    calls = {}

    def fake_builder(raw_dataset):
        calls["builder_input"] = raw_dataset
        return [{"prompt": [{"role": "system", "content": "controlled"}], "answer": "7"}]

    def fake_audit(rows, tokenizer, max_tokens):
        calls["audit"] = (rows, tokenizer, max_tokens)
        return {"count": 1, "max_tokens": 12, "p95_tokens": 12.0, "limit": max_tokens}

    monkeypatch.setattr(sample_p0, "build_gsm8k_rl_rows", fake_builder)
    monkeypatch.setattr(sample_p0, "assert_prompt_token_limit", fake_audit)

    tokenizer = object()
    rows, audit = sample_p0.prepare_p0_rows([{"raw": True}], tokenizer, 512)

    assert calls["builder_input"] == [{"raw": True}]
    assert calls["audit"] == (rows, tokenizer, 512)
    assert audit["limit"] == 512


def test_build_p0_manifest_records_exact_provenance(tmp_path):
    import controlled_run.sample_p0 as sample_p0

    config_path = tmp_path / "grpo.yaml"
    config_path.write_text("seed: 42\n", encoding="utf-8")
    pi0_manifest = {"policy_name": "pi_0", "files": {"model.safetensors": "abc"}}
    settings = {"temperature": 0.8, "num_generations": 16}
    prompt_audit = {"count": 30, "limit": 512}

    manifest = sample_p0.build_p0_manifest(
        policy_dir=tmp_path / "pi_0",
        pi0_manifest=pi0_manifest,
        pi0_lineage_id="lineage-sha",
        dataset_sha="gsm8k-sha",
        dataset_config="main",
        config_path=config_path,
        sampling_settings=settings,
        prompt_audit=prompt_audit,
        start_index=10,
        end_index=40,
        record_count=30,
        runtime={"gpu_name": "NVIDIA A40"},
    )

    assert manifest["mode"] == "canonical_p0"
    assert manifest["pi0_lineage_id"] == "lineage-sha"
    assert manifest["pi0_manifest"] == pi0_manifest
    assert manifest["dataset"] == {
        "name": "openai/gsm8k",
        "config": "main",
        "split": "test",
        "sha": "gsm8k-sha",
    }
    assert manifest["sampling"] == settings
    assert manifest["prompt_length_audit"] == prompt_audit
    assert manifest["shard"] == {"start_index": 10, "end_index": 40, "record_count": 30}
    assert manifest["grpo_config_sha256"] == sample_p0.sha256_file(config_path)


def test_build_p0_manifest_records_explicit_indices_when_provided(tmp_path):
    import controlled_run.sample_p0 as sample_p0

    config_path = tmp_path / "grpo.yaml"
    config_path.write_text("seed: 42\n", encoding="utf-8")

    manifest = sample_p0.build_p0_manifest(
        policy_dir=tmp_path / "pi_0",
        pi0_manifest={"policy_name": "pi_0", "files": {}},
        pi0_lineage_id="lineage-sha",
        dataset_sha="gsm8k-sha",
        dataset_config="main",
        config_path=config_path,
        sampling_settings={"num_generations": 1024},
        prompt_audit={"count": 1, "limit": 512},
        start_index=2,
        end_index=8,
        record_count=3,
        runtime={},
        indices=[2, 5, 7],
    )

    assert manifest["shard"] == {
        "start_index": 2,
        "end_index": 8,
        "record_count": 3,
        "indices": [2, 5, 7],
    }


def test_sample_indexed_rows_uses_policy_tokenizer_tokens_and_shared_reward(monkeypatch, tmp_path):
    import controlled_run.sample_p0 as sample_p0

    captured = {"prompts": [], "params": [], "reward": []}

    class FakeTokenizer:
        def apply_chat_template(self, prompt, tokenize, add_generation_prompt):
            assert tokenize is True
            assert add_generation_prompt is True
            assert prompt[0]["content"] == "controlled"
            return {
                "input_ids": [101, 202, 303],
                "attention_mask": [1, 1, 1],
            }

    class FakeTokensPrompt:
        def __init__(self, prompt_token_ids):
            captured["prompts"].append(prompt_token_ids)
            self.prompt_token_ids = prompt_token_ids

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            captured["params"].append(kwargs)
            self.kwargs = kwargs

    class FakeOutput:
        def __init__(self, text):
            self.text = text

    class FakeRequestOutput:
        def __init__(self):
            self.outputs = [FakeOutput("answer one"), FakeOutput("answer two")]

    class FakeLLM:
        def generate(self, prompts, sampling_params, use_tqdm):
            assert len(prompts) == 1
            assert use_tqdm is False
            return [FakeRequestOutput()]

    def fake_reward(completions, answer, **kwargs):
        del kwargs
        captured["reward"].append((list(completions), answer))
        return [1.0, 0.0]

    monkeypatch.setattr(sample_p0, "gsm8k_binary_reward", fake_reward)

    rows = [
        (
            17,
            {
                "prompt": [{"role": "system", "content": "controlled"}],
                "answer": "7",
            },
        )
    ]
    settings = {
        "num_generations": 2,
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 0,
        "repetition_penalty": 1.0,
        "max_completion_length": 1024,
        "seed": 42,
    }
    output_path = tmp_path / "p0_raw.jsonl"

    sample_p0.sample_indexed_rows(
        llm=FakeLLM(),
        tokenizer=FakeTokenizer(),
        indexed_rows=rows,
        settings=settings,
        output_path=output_path,
        sampling_params_cls=FakeSamplingParams,
        tokens_prompt_cls=FakeTokensPrompt,
    )

    assert captured["prompts"] == [[101, 202, 303]]
    assert captured["params"][0] == {
        "n": 2,
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 0,
        "repetition_penalty": 1.0,
        "max_tokens": 1024,
        "seed": 4_200_017,
    }
    assert captured["reward"] == [(["answer one", "answer two"], "7")]

    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["dataset_index"] == 17
    assert records[0]["question_seed"] == 4_200_017
    assert records[0]["n_correct"] == 1
    assert records[0]["n_rollouts"] == 2
    assert records[0]["rollouts"] == [
        {"rollout": 0, "correct": True, "text": "answer one"},
        {"rollout": 1, "correct": False, "text": "answer two"},
    ]


def test_controlled_p0_sampler_never_imports_historical_prompt_constants():
    source_path = ROOT / "controlled_run/sample_p0.py"
    if not source_path.exists():
        pytest.fail("controlled_run/sample_p0.py does not exist yet")
    source = source_path.read_text(encoding="utf-8")
    assert "probe.prompts" not in source
    assert "TOKENIZER_NAME" not in source
