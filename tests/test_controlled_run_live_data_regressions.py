from __future__ import annotations

import pytest

from controlled_run.data import build_sft_manifest, materialize_sft_records


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
        return list(range(100))


def make_row(source_index: int, uuid: str, problem: str) -> dict:
    return {
        "source_index": source_index,
        "uuid": uuid,
        "problem": problem,
        "source": "synthetic",
        "generations": [f"reasoning for row {source_index} \\boxed{{1}}"],
        "is_reasoning_complete": [True],
        "correctness_math_verify": [True],
    }


def test_source_index_allows_repeated_missing_like_uuid_metadata():
    rows = [
        make_row(10, "NaN", "safe problem alpha"),
        make_row(11, "NaN", "safe problem beta"),
        make_row(12, "duplicate-upstream-uuid", "safe problem gamma"),
        make_row(13, "duplicate-upstream-uuid", "safe problem delta"),
    ]

    manifest, _ = build_sft_manifest(
        rows,
        [],
        FakeTokenizer(),
        target_size=4,
        seed=42,
    )
    records = materialize_sft_records(manifest, rows)

    assert {item["source_index"] for item in manifest} == {10, 11, 12, 13}
    assert {item["source_index"] for item in records} == {10, 11, 12, 13}
    assert [item["uuid"] for item in records].count("NaN") == 2
    assert [item["uuid"] for item in records].count("duplicate-upstream-uuid") == 2


def test_duplicate_source_index_is_rejected_even_when_uuid_differs():
    rows = [
        make_row(7, "u1", "safe problem one"),
        make_row(7, "u2", "safe problem two"),
    ]

    with pytest.raises(ValueError, match="Duplicate OpenR1 source_index: 7"):
        build_sft_manifest(
            rows,
            [],
            FakeTokenizer(),
            target_size=1,
            seed=42,
        )
