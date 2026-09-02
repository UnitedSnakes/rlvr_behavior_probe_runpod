from __future__ import annotations

import math

import pytest

import diagnose_p0_signal_budget as p0diag


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeLLM:
    def __init__(self):
        self.calls = []

    def generate(self, prompts, sampling_params, use_tqdm):
        self.calls.append(
            {
                "prompts": list(prompts),
                "sampling_params": list(sampling_params),
                "use_tqdm": use_tqdm,
            }
        )
        return [f"call-{len(self.calls)}-{i}" for i in range(len(prompts))]


class FakeOutput:
    def __init__(self, text: str, token_ids: list[int], finish_reason: str = "stop"):
        self.text = text
        self.token_ids = token_ids
        self.finish_reason = finish_reason


def test_half_seed_uses_disjoint_preregistered_offsets():
    assert p0diag.half_seed(42, 17, "A") == 4_200_017
    assert p0diag.half_seed(42, 17, "B") == 4_250_017
    assert p0diag.half_seed(42, 18, "A") != p0diag.half_seed(42, 17, "A")
    with pytest.raises(ValueError, match="half"):
        p0diag.half_seed(42, 17, "C")


def test_generate_independent_halves_uses_two_calls_and_n16():
    llm = FakeLLM()
    prompts = ["p0", "p1"]
    dataset_indices = [7, 13]
    settings = {
        "temperature": 0.8,
        "top_p": 1.0,
        "top_k": 0,
        "repetition_penalty": 1.0,
        "max_completion_length": 2048,
        "seed": 42,
    }

    a, b = p0diag.generate_independent_halves(
        llm=llm,
        prompts=prompts,
        dataset_indices=dataset_indices,
        settings=settings,
        sampling_params_cls=FakeSamplingParams,
        half_size=16,
        use_tqdm=False,
    )

    assert a == ["call-1-0", "call-1-1"]
    assert b == ["call-2-0", "call-2-1"]
    assert len(llm.calls) == 2

    for call in llm.calls:
        assert call["prompts"] == prompts
        assert call["use_tqdm"] is False
        assert [x.kwargs["n"] for x in call["sampling_params"]] == [16, 16]
        assert all(x.kwargs["top_p"] == 1.0 for x in call["sampling_params"])

    assert [x.kwargs["seed"] for x in llm.calls[0]["sampling_params"]] == [
        4_200_007,
        4_200_013,
    ]
    assert [x.kwargs["seed"] for x in llm.calls[1]["sampling_params"]] == [
        4_250_007,
        4_250_013,
    ]


def test_score_half_uses_token_termination_and_canonical_reward(monkeypatch):
    monkeypatch.setattr(
        p0diag,
        "gsm8k_binary_reward",
        lambda completions, answer: [1.0, 1.0, 0.0, 0.0],
    )

    outputs = [
        FakeOutput("correct terminated", [10, 151643]),
        FakeOutput("correct truncated", [10, 11], finish_reason="length"),
        FakeOutput("wrong terminated", [12, 151643]),
        FakeOutput("wrong truncated", [12, 13], finish_reason="length"),
    ]

    rollouts = p0diag.score_half_outputs(
        outputs,
        answer="#### 7",
        terminal_token_ids=(151643,),
        half_label="A",
        store_token_ids=False,
    )

    assert [r["correct"] for r in rollouts] == [True, True, False, False]
    assert [r["terminated"] for r in rollouts] == [True, False, True, False]
    assert [r["canonical_reward"] for r in rollouts] == [1, 0, 0, 0]
    assert [r["capped"] for r in rollouts] == [False, True, False, True]
    assert all(r["half"] == "A" for r in rollouts)
    assert all("token_ids" not in r for r in rollouts)


def test_combine_halves_defines_p0_from_terminated_correct_and_g16_live():
    half_a = [
        {"correct": True, "terminated": True, "canonical_reward": 1, "capped": False},
        {"correct": True, "terminated": False, "canonical_reward": 0, "capped": True},
    ] + [
        {"correct": False, "terminated": True, "canonical_reward": 0, "capped": False}
        for _ in range(14)
    ]
    half_b = [
        {"correct": True, "terminated": True, "canonical_reward": 1, "capped": False}
        for _ in range(2)
    ] + [
        {"correct": False, "terminated": True, "canonical_reward": 0, "capped": False}
        for _ in range(14)
    ]

    record = p0diag.combine_halves(
        dataset_index=9,
        question="q",
        gold="#### 7",
        half_a=half_a,
        half_b=half_b,
        group_size=16,
    )

    assert record["p0_A"] == pytest.approx(1 / 16)
    assert record["p0_B"] == pytest.approx(2 / 16)
    assert record["p0"] == pytest.approx(3 / 32)
    assert record["correctness_p0"] == pytest.approx(4 / 32)
    assert record["termination_rate"] == pytest.approx(31 / 32)
    assert record["cap_rate"] == pytest.approx(1 / 32)
    assert record["empirical_live_A"] is True
    assert record["empirical_live_B"] is True
    assert record["model_live_probability_G16"] == pytest.approx(
        1 - (3 / 32) ** 16 - (1 - 3 / 32) ** 16
    )
    assert record["group_size"] == 16
    assert record["n_rollouts"] == 32


def test_model_live_probability_is_explicitly_g16_not_k32():
    p = 0.2
    observed = p0diag.model_live_probability(p, group_size=16)
    assert observed == pytest.approx(1 - p**16 - (1 - p) ** 16)
    assert not math.isclose(observed, 1 - p**32 - (1 - p) ** 32)
