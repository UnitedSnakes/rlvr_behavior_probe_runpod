from __future__ import annotations

import pytest

from controlled_run.rewards import (
    completion_text,
    gsm8k_binary_reward,
    is_truncated_completion,
    make_gsm8k_terminated_binary_reward,
    resolve_terminal_token_ids,
)


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


def test_is_truncated_completion_matches_trl_predicate():
    # TRL: a completion is truncated iff ids[-1] not in [eos_token_id, pad_token_id]
    assert is_truncated_completion([1, 2, 151643], {151643, 151645}) is False
    assert is_truncated_completion([1, 2, 999], {151643, 151645}) is True
    # An empty completion never terminated.
    assert is_truncated_completion([], {151643}) is True


def test_terminated_reward_zeroes_correct_but_truncated_completions():
    reward = make_gsm8k_terminated_binary_reward({151643})
    completions = [
        [{"role": "assistant", "content": "work... \\boxed{12}"}],  # correct, stopped
        [{"role": "assistant", "content": "work... \\boxed{12}"}],  # correct, truncated
        [{"role": "assistant", "content": "work... \\boxed{13}"}],  # wrong, stopped
    ]
    completion_ids = [[7, 151643], [7, 8], [7, 151643]]

    assert reward(
        completions,
        answer=["12", "12", "12"],
        completion_ids=completion_ids,
    ) == [1.0, 0.0, 0.0]


def test_terminated_reward_exposes_canonical_name():
    reward = make_gsm8k_terminated_binary_reward({151643})
    assert reward.__name__ == "binary_terminated_final_answer_correctness"


def test_terminated_reward_requires_completion_ids():
    reward = make_gsm8k_terminated_binary_reward({151643})
    with pytest.raises(ValueError, match="completion_ids"):
        reward(["\\boxed{12}"], answer=["12"])


def test_terminated_reward_rejects_mismatched_completion_ids():
    reward = make_gsm8k_terminated_binary_reward({151643})
    with pytest.raises(ValueError, match="does not match"):
        reward(
            ["\\boxed{12}", "\\boxed{12}"],
            answer=["12", "12"],
            completion_ids=[[151643]],
        )


def test_terminated_reward_factory_rejects_empty_terminal_set():
    with pytest.raises(ValueError, match="at least one token id"):
        make_gsm8k_terminated_binary_reward(set())


def test_resolve_terminal_token_ids_uses_eos_and_pad():
    class Tok:
        eos_token_id = 151643
        pad_token_id = 151643

    assert resolve_terminal_token_ids(Tok()) == (151643,)

    class TokDistinct:
        eos_token_id = 151643
        pad_token_id = 151645

    assert resolve_terminal_token_ids(TokDistinct()) == (151643, 151645)


def test_resolve_terminal_token_ids_rejects_missing_terminals():
    class Tok:
        eos_token_id = None
        pad_token_id = None

    with pytest.raises(ValueError, match="truncation"):
        resolve_terminal_token_ids(Tok())
