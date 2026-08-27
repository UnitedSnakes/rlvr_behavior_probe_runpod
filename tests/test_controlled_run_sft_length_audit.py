from __future__ import annotations

import pytest

from controlled_run.data import build_sft_manifest


class LengthTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
    ):
        assert tokenize is True
        assert add_generation_prompt is False
        completion = messages[2]["content"]
        length = int(completion.split("LEN=")[1].split()[0])
        return list(range(length))


def _row(index: int, length: int) -> dict:
    return {
        "uuid": f"u{index}",
        "problem": f"unrelated synthetic problem {index}",
        "source": "synthetic",
        "generations": [f"LEN={length} verified reasoning \\boxed{{1}}"],
        "is_reasoning_complete": [True],
        "correctness_math_verify": [True],
    }


def test_sft_audit_reports_pre_length_filter_distribution_and_tail_fractions():
    lengths = [100, 500, 1000, 2000, 2048, 2049, 4096, 4097, 8192, 8193]
    rows = [_row(index, length) for index, length in enumerate(lengths)]

    manifest, audit = build_sft_manifest(
        rows,
        [],
        LengthTokenizer(),
        target_size=5,
        seed=42,
    )

    assert len(manifest) == 5
    assert audit["pre_length_filter_count"] == 10
    assert audit["removed_too_long"] == 5
    assert audit["formatted_token_percentiles"] == {
        "p50": pytest.approx(2048.5),
        "p75": pytest.approx(4096.25),
        "p90": pytest.approx(8192.1),
        "p95": pytest.approx(8192.55),
        "p99": pytest.approx(8192.91),
    }
    assert audit["formatted_token_tail_fractions"] == {
        "gt_2048": pytest.approx(0.5),
        "gt_4096": pytest.approx(0.3),
        "gt_8192": pytest.approx(0.1),
    }


def test_sft_length_audit_uses_contamination_clean_verified_denominator():
    safe = _row(0, 100)
    duplicate = _row(1, 9000)
    duplicate["problem"] = "heldout duplicate"
    unverified = _row(2, 9000)
    unverified["correctness_math_verify"] = [False]

    _, audit = build_sft_manifest(
        [safe, duplicate, unverified],
        [{"question": "heldout duplicate"}],
        LengthTokenizer(),
        target_size=1,
        seed=42,
    )

    assert audit["pre_length_filter_count"] == 1
    assert audit["formatted_token_percentiles"]["p99"] == pytest.approx(100.0)
    assert audit["formatted_token_tail_fractions"]["gt_2048"] == 0.0
