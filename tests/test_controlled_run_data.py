from __future__ import annotations

import hashlib

import pytest

from controlled_run.constants import CONTROLLED_SYSTEM_PROMPT
from controlled_run.data import (
    NearDuplicateIndex,
    build_sft_manifest,
    materialize_sft_records,
    normalize_problem,
    select_verified_trace,
    stable_candidate_key,
    word_shingles,
)


class FakeTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
    ):
        assert tokenize is True
        assert add_generation_prompt is False
        assert messages[0] == {
            "role": "system",
            "content": CONTROLLED_SYSTEM_PROMPT,
        }
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"

        completion = messages[2]["content"]
        if "LONG_TRACE" in completion:
            return list(range(2049))
        return list(range(100))


def make_row(
    uuid: str,
    problem: str,
    *,
    source: str = "synthetic",
    generations: list[str] | None = None,
    complete: list[bool] | None = None,
    verified: list[bool] | None = None,
):
    generations = generations or [f"reasoning for {uuid} \\boxed{{1}}"]
    complete = complete if complete is not None else [True] * len(generations)
    verified = verified if verified is not None else [True] * len(generations)
    return {
        "uuid": uuid,
        "problem": problem,
        "source": source,
        "generations": generations,
        "is_reasoning_complete": complete,
        "correctness_math_verify": verified,
    }


def test_select_verified_trace_uses_lowest_complete_math_verified_generation():
    row = make_row(
        "u1",
        "What is 2+2?",
        generations=["bad", "reasoning ... \\boxed{4}", "also correct"],
        complete=[True, True, True],
        verified=[False, True, True],
    )

    assert select_verified_trace(row) == (1, "reasoning ... \\boxed{4}")


def test_select_verified_trace_skips_incomplete_or_unverified_rows():
    incomplete = make_row(
        "u1",
        "q",
        generations=["trace"],
        complete=[False],
        verified=[True],
    )
    unverified = make_row(
        "u2",
        "q",
        generations=["trace"],
        complete=[True],
        verified=[False],
    )

    assert select_verified_trace(incomplete) is None
    assert select_verified_trace(unverified) is None


def test_select_verified_trace_rejects_misaligned_sequence_fields():
    row = make_row(
        "broken",
        "q",
        generations=["a", "b"],
        complete=[True],
        verified=[True, True],
    )

    with pytest.raises(ValueError, match="broken.*sequence lengths"):
        select_verified_trace(row)


def test_normalize_problem_has_basic_and_aggressive_modes():
    assert normalize_problem("How many? 1,000") == normalize_problem(
        "  HOW many?   1,000  "
    )
    assert normalize_problem(
        "How many? 1,000", aggressive=True
    ) == normalize_problem("How many 1000", aggressive=True)


def test_word_shingles_uses_aggressive_normalized_words():
    shingles = word_shingles("Alpha, beta gamma delta epsilon zeta", n=5)

    assert shingles == {
        "alpha beta gamma delta epsilon",
        "beta gamma delta epsilon zeta",
    }


def test_word_shingles_keeps_short_nonempty_text_comparable():
    assert word_shingles("one two three", n=5) == {"one two three"}
    assert word_shingles("   ", n=5) == set()


def test_near_duplicate_index_finds_default_threshold_match():
    reference = " ".join(f"word{i}" for i in range(20))
    candidate_words = [f"word{i}" for i in range(20)]
    candidate_words[-1] = "replacement"
    candidate = " ".join(candidate_words)

    index = NearDuplicateIndex([reference], n=5, threshold=0.80)

    assert index.find_match(candidate) == 0


def test_near_duplicate_index_rejects_below_threshold_candidate():
    reference_words = [f"word{i}" for i in range(20)]
    candidate_words = list(reference_words)
    candidate_words[10] = "replacement"

    index = NearDuplicateIndex(
        [" ".join(reference_words)],
        n=5,
        threshold=0.80,
    )

    assert index.find_match(" ".join(candidate_words)) is None


def test_near_duplicate_index_breaks_ties_by_reference_order():
    text = "alpha beta gamma delta epsilon zeta"
    index = NearDuplicateIndex([text, text], n=5, threshold=0.80)

    assert index.find_match(text) == 0


def test_stable_candidate_key_is_seeded_sha256():
    expected = hashlib.sha256(b"42:u1").hexdigest()
    assert stable_candidate_key("u1", 42) == expected
    assert stable_candidate_key("u1", 7) != expected


def test_build_sft_manifest_is_order_invariant_and_records_filter_counts():
    openr1_rows = [
        make_row("u1", "safe problem one"),
        make_row("u2", "GSM duplicate question"),
        make_row("u3", "safe problem three"),
        make_row(
            "u4",
            "safe but too long",
            generations=["LONG_TRACE \\boxed{1}"],
        ),
        make_row(
            "u5",
            "safe but not verified",
            verified=[False],
        ),
        make_row("u6", "safe problem six"),
        make_row("u7", "safe problem seven"),
    ]
    gsm8k_rows = [{"question": "GSM duplicate question"}]
    tokenizer = FakeTokenizer()

    manifest_a, audit_a = build_sft_manifest(
        openr1_rows,
        gsm8k_rows,
        tokenizer,
        target_size=3,
        seed=42,
    )
    manifest_b, audit_b = build_sft_manifest(
        list(reversed(openr1_rows)),
        gsm8k_rows,
        tokenizer,
        target_size=3,
        seed=42,
    )

    eligible = ["u1", "u3", "u6", "u7"]
    expected = sorted(eligible, key=lambda uuid: stable_candidate_key(uuid, 42))[:3]

    assert [row["uuid"] for row in manifest_a] == expected
    assert manifest_a == manifest_b
    assert audit_a == audit_b
    assert audit_a["candidate_count"] == 7
    assert audit_a["removed_exact_duplicates"] == 1
    assert audit_a["removed_normalized_duplicates"] == 0
    assert audit_a["removed_near_duplicates"] == 0
    assert audit_a["removed_too_long"] == 1
    assert audit_a["removed_no_verified_trace"] == 1
    assert audit_a["eligible_after_filters"] == 4
    assert audit_a["final_count"] == 3

    assert set(manifest_a[0]) == {
        "uuid",
        "generation_index",
        "source",
        "problem_sha256",
        "completion_sha256",
        "formatted_token_count",
    }


def test_build_sft_manifest_counts_normalized_and_near_duplicates_separately():
    reference = " ".join(f"word{i}" for i in range(20))
    near_words = [f"word{i}" for i in range(20)]
    near_words[-1] = "replacement"
    near = " ".join(near_words)

    openr1_rows = [
        make_row("normalized", "HOW many? 1,000"),
        make_row("near", near),
        make_row("safe", "entirely unrelated safe problem"),
    ]
    gsm8k_rows = [
        {"question": "How many 1000"},
        {"question": reference},
    ]

    manifest, audit = build_sft_manifest(
        openr1_rows,
        gsm8k_rows,
        FakeTokenizer(),
        target_size=1,
        seed=42,
    )

    assert [row["uuid"] for row in manifest] == ["safe"]
    assert audit["removed_exact_duplicates"] == 0
    assert audit["removed_normalized_duplicates"] == 1
    assert audit["removed_near_duplicates"] == 1


def test_build_sft_manifest_fails_if_not_enough_examples_survive():
    rows = [make_row("u1", "only safe row")]

    with pytest.raises(ValueError, match="requested 2.*only 1"):
        build_sft_manifest(
            rows,
            [],
            FakeTokenizer(),
            target_size=2,
            seed=42,
        )


def test_materialize_sft_records_reconstructs_prompt_and_selected_completion():
    source_rows = [
        make_row(
            "u1",
            "What is 2+2?",
            generations=["wrong", "correct reasoning \\boxed{4}"],
            complete=[True, True],
            verified=[False, True],
        )
    ]
    manifest, _ = build_sft_manifest(
        source_rows,
        [],
        FakeTokenizer(),
        target_size=1,
        seed=42,
    )

    records = materialize_sft_records(manifest, source_rows)

    assert records == [
        {
            "uuid": "u1",
            "prompt": [
                {"role": "system", "content": CONTROLLED_SYSTEM_PROMPT},
                {"role": "user", "content": "What is 2+2?"},
            ],
            "completion": [
                {
                    "role": "assistant",
                    "content": "correct reasoning \\boxed{4}",
                }
            ],
        }
    ]
