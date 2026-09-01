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


def is_truncated_completion(completion_ids, terminal_token_ids) -> bool:
    """Return True when a completion did not end on a terminal token.

    This mirrors the predicate TRL applies internally when
    ``mask_truncated_completions`` is enabled (``grpo_trainer.py``: a
    completion is truncated iff ``ids[-1] not in [eos_token_id,
    pad_token_id]``). Reusing the same rule keeps the reward definition and
    TRL's own truncation accounting from drifting apart.

    Truncation is decided from token ids, never from decoded text: vLLM omits
    special tokens from text output by default.
    """
    ids = list(completion_ids)
    if not ids:
        return True
    return ids[-1] not in set(terminal_token_ids)


def make_gsm8k_terminated_binary_reward(terminal_token_ids, *, ledger_recorder=None):
    """Build the canonical post-2026-08-30 GRPO reward.

    r(x, z) = 1 if z terminated within the completion budget and the extracted
    final numeric answer is correct, else 0.

    See docs/superpowers/specs/2026-08-30-grpo-truncation-policy-amendment.md.
    A truncated completion scores 0 even when a correct answer is parseable
    from the truncated text; that cost is accepted deliberately.
    """
    terminal_token_ids = frozenset(terminal_token_ids)
    if not terminal_token_ids:
        raise ValueError("terminal_token_ids must contain at least one token id")

    def gsm8k_terminated_binary_reward(
        completions, answer, completion_ids=None, **kwargs
    ) -> list[float]:
        completions = list(completions)
        correctness = gsm8k_binary_reward(completions, answer)

        if completion_ids is None:
            raise ValueError(
                "gsm8k_terminated_binary_reward requires completion_ids; TRL "
                "supplies them to synchronous reward functions"
            )

        completion_ids = list(completion_ids)
        if len(completion_ids) != len(completions):
            raise ValueError(
                f"completion_ids count {len(completion_ids)} does not match "
                f"completion count {len(completions)}"
            )

        terminated = [
            not is_truncated_completion(ids, terminal_token_ids)
            for ids in completion_ids
        ]
        rewards = [
            score if did_terminate else 0.0
            for score, did_terminate in zip(correctness, terminated, strict=True)
        ]

        if ledger_recorder is not None:
            dataset_indices = kwargs.get("dataset_index")
            if dataset_indices is None:
                raise ValueError(
                    "Signal ledger requires dataset_index to reach the reward function"
                )
            ledger_recorder.capture(
                dataset_indices=dataset_indices,
                correctness=correctness,
                terminated=terminated,
                rewards=rewards,
                completion_lengths=[len(ids) for ids in completion_ids],
            )

        return rewards

    gsm8k_terminated_binary_reward.__name__ = "binary_terminated_final_answer_correctness"
    return gsm8k_terminated_binary_reward


def resolve_terminal_token_ids(tokenizer) -> tuple[int, ...]:
    """Terminal token ids used to decide whether a completion truncated.

    Must match TRL's own rule (eos plus pad) so the reward definition and
    TRL's clipped_ratio metric cannot drift apart.
    """
    ids = [
        getattr(tokenizer, "eos_token_id", None),
        getattr(tokenizer, "pad_token_id", None),
    ]
    resolved = tuple(sorted({int(i) for i in ids if i is not None}))
    if not resolved:
        raise ValueError(
            "Tokenizer exposes neither eos_token_id nor pad_token_id; "
            "cannot determine completion truncation"
        )
    return resolved
