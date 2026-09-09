from __future__ import annotations

from analyses.strict_extractor_robustness import (
    replace_delta_c_with_strict,
    score_rollouts,
)
from probe.scoring import (
    extract_numeric_answer,
    extract_numeric_answer_strict,
)


def test_strict_extractor_preserves_explicit_answer_forms():
    cases = [
        ("reasoning... \\boxed{12}", 12.0, "boxed"),
        ("The final answer is 12.", 12.0, "final_phrase"),
        ("work... #### 12", 12.0, "final_phrase"),
        ("work... boxed: 12", 12.0, "boxed"),
    ]

    for text, expected, method in cases:
        strict = extract_numeric_answer_strict(text)
        current = extract_numeric_answer(text)
        assert strict[0] == expected
        assert strict[2] == method
        assert current == strict


def test_strict_extractor_disables_generic_last_number_fallback():
    text = "The computation has several steps and ends with the number 12."

    current = extract_numeric_answer(text)
    strict = extract_numeric_answer_strict(text)

    assert current == (12.0, "12", "last_number")
    assert strict == (None, None, "none")


def test_rollout_summary_measures_fallback_dependence_without_label_drift():
    summary, examples = score_rollouts(
        [
            {"rollout": 0, "text": "work... 12", "correct": True},
            {"rollout": 1, "text": "work... \\boxed{12}", "correct": True},
            {"rollout": 2, "text": "work... \\boxed{13}", "correct": False},
        ],
        gold=12.0,
        source="synthetic",
        dataset_index=7,
    )

    assert summary["n_rollouts"] == 3
    assert summary["current_n_correct"] == 2
    assert summary["strict_n_correct"] == 1
    assert summary["n_disagree"] == 1
    assert summary["n_last_number"] == 1
    assert summary["n_last_number_correct"] == 1
    assert summary["scorer_drift_count"] == 0
    assert len(examples) == 1
    assert examples[0]["current_method"] == "last_number"
    assert examples[0]["strict_method"] == "none"


def test_replace_delta_c_keeps_frozen_direction_and_uses_opposite_baseline_half():
    rows = [
        {
            "snapshot_pct": "25",
            "dataset_index": "0",
            "direction": "A-bin/B-base",
            "delta_C": "0.10",
        },
        {
            "snapshot_pct": "25",
            "dataset_index": "0",
            "direction": "B-bin/A-base",
            "delta_C": "0.20",
        },
    ]
    strict = replace_delta_c_with_strict(
        rows,
        p0_strict_halves={0: {"A": 0.25, "B": 0.50}},
        snapshot_strict={25: {0: 0.75}},
    )

    assert strict[0]["delta_C_current"] == 0.10
    assert strict[0]["baseline_C_strict"] == 0.50
    assert strict[0]["snapshot_C_strict"] == 0.75
    assert strict[0]["delta_C"] == 0.25

    assert strict[1]["delta_C_current"] == 0.20
    assert strict[1]["baseline_C_strict"] == 0.25
    assert strict[1]["delta_C"] == 0.50
