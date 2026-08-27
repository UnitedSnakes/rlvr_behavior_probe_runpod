from __future__ import annotations

import pytest

from controlled_run.rewards import completion_text, gsm8k_binary_reward


def test_completion_text_accepts_string_and_chat_completion():
    assert completion_text("plain answer") == "plain answer"
    assert completion_text(
        [{"role": "assistant", "content": "reasoning \\boxed{2}"}]
    ) == "reasoning \\boxed{2}"


def test_completion_text_rejects_ambiguous_chat_completion():
    with pytest.raises(ValueError, match="assistant completion"):
        completion_text(
            [
                {"role": "assistant", "content": "first"},
                {"role": "assistant", "content": "second"},
            ]
        )


def test_binary_reward_scores_numeric_correctness_only():
    completions = [
        [{"role": "assistant", "content": "work... \\boxed{12}"}],
        [{"role": "assistant", "content": "work... \\boxed{13}"}],
        [{"role": "assistant", "content": "no numeric answer"}],
    ]

    assert gsm8k_binary_reward(
        completions,
        answer=["12", "12", "12"],
    ) == [1.0, 0.0, 0.0]


def test_binary_reward_reuses_numeric_scorer_for_fraction_equivalence():
    completions = [
        "The final answer is 1/2.",
        "The final answer is 0.5.",
    ]

    assert gsm8k_binary_reward(completions, answer=["0.5", "1/2"]) == [1.0, 1.0]


def test_binary_reward_broadcasts_one_gold_answer_to_group():
    completions = ["\\boxed{7}", "\\boxed{8}"]

    assert gsm8k_binary_reward(completions, answer="7") == [1.0, 0.0]


def test_binary_reward_rejects_answer_length_mismatch():
    with pytest.raises(ValueError, match="answer count"):
        gsm8k_binary_reward(["\\boxed{1}", "\\boxed{2}"], answer=["1"])
