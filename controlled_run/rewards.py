from __future__ import annotations

from collections.abc import Sequence

from probe.scoring import _to_number, extract_numeric_answer, numeric_equal


def completion_text(completion) -> str:
    if isinstance(completion, str):
        return completion

    if isinstance(completion, dict):
        if completion.get("role") == "assistant" and isinstance(
            completion.get("content"), str
        ):
            return completion["content"]
        raise ValueError("Expected an assistant completion with string content")

    if isinstance(completion, Sequence):
        assistant_messages = [
            message
            for message in completion
            if isinstance(message, dict)
            and message.get("role") == "assistant"
            and isinstance(message.get("content"), str)
        ]
        if len(assistant_messages) != 1:
            raise ValueError(
                "Expected exactly one assistant completion with string content"
            )
        return assistant_messages[0]["content"]

    raise ValueError(
        "Expected completion to be a string or conversational assistant completion"
    )


def _answers_for_group(answer, count: int) -> list[str]:
    if isinstance(answer, str):
        return [answer] * count

    if not isinstance(answer, Sequence):
        raise ValueError("answer must be a string or a sequence of gold answers")

    answers = list(answer)
    if len(answers) != count:
        raise ValueError(
            f"Completion count {count} does not match answer count {len(answers)}"
        )
    return [str(value) for value in answers]


def gsm8k_binary_reward(completions, answer, **kwargs) -> list[float]:
    del kwargs
    completions = list(completions)
    answers = _answers_for_group(answer, len(completions))

    rewards: list[float] = []
    for completion, gold_text in zip(completions, answers):
        gold_value = _to_number(gold_text)
        if gold_value is None:
            raise ValueError(f"Could not parse GSM8K gold answer: {gold_text!r}")

        pred_value, _, _ = extract_numeric_answer(completion_text(completion))
        rewards.append(1.0 if numeric_equal(pred_value, gold_value) else 0.0)

    return rewards
