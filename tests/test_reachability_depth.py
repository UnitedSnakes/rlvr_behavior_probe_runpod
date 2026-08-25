import math

import numpy as np

from analyses.reachability_depth import (
    build_per_question_rows,
    simulate_depth_curve,
    zero_success_upper_bound,
)


def make_row(qid, correctness, alias="sft"):
    return {
        "model_alias": alias,
        "qid": qid,
        "question": f"question {qid}",
        "gold": str(qid),
        "n_correct": sum(correctness),
        "n_rollouts": len(correctness),
        "rollouts": [
            {"rollout": index, "correct": bool(correct)}
            for index, correct in enumerate(correctness)
        ],
    }


def test_zero_success_upper_bound_matches_exact_binomial_bound():
    bound = zero_success_upper_bound(256)
    expected = 1.0 - 0.05 ** (1.0 / 256)
    assert math.isclose(bound, expected, rel_tol=0, abs_tol=1e-12)


def test_per_question_rows_flag_persistent_unobserved_sft_success():
    sft_rows = [
        make_row(0, [False, False, False, False]),
        make_row(1, [False, True, False, False]),
    ]
    rl_rows = [
        make_row(0, [True, False], alias="rl"),
        make_row(1, [True, True], alias="rl"),
    ]

    rows = build_per_question_rows(sft_rows, rl_rows)
    by_qid = {row["qid"]: row for row in rows}

    assert by_qid[0]["sft_successes_full"] == 0
    assert by_qid[0]["rl_successes"] == 1
    assert by_qid[0]["persistent_unobserved_sft_success"] is True
    assert by_qid[0]["sft_zero_success_95pct_upper_bound"] > 0

    assert by_qid[1]["sft_successes_full"] == 1
    assert by_qid[1]["persistent_unobserved_sft_success"] is False
    assert np.isnan(by_qid[1]["sft_zero_success_95pct_upper_bound"])


def test_depth_curve_tracks_coverage_and_rl_success_unresolved_questions():
    sft_rows = [
        make_row(0, [False, False, False, False]),
        make_row(1, [True, True, True, True]),
        make_row(2, [True, False, False, False]),
    ]
    rl_rows = [
        make_row(0, [True, False], alias="rl"),
        make_row(1, [True, False], alias="rl"),
        make_row(2, [True, False], alias="rl"),
    ]

    curve = simulate_depth_curve(
        sft_rows,
        rl_rows,
        k_values=[1, 2, 4],
        repeats=4000,
        seed=7,
    )
    by_k = {row["k"]: row for row in curve}

    # At full depth, qid 0 is the only SFT-uncovered question.
    assert by_k[4]["sft_covered_mean"] == 2.0
    assert by_k[4]["rl_success_unresolved_mean"] == 1.0

    # At K=1, qid 2 is unresolved with probability 3/4.
    assert abs(by_k[1]["rl_success_unresolved_mean"] - 1.75) < 0.04
    assert abs(by_k[1]["sft_covered_mean"] - 1.25) < 0.04

    coverage = [by_k[k]["sft_covered_mean"] for k in [1, 2, 4]]
    unresolved = [by_k[k]["rl_success_unresolved_mean"] for k in [1, 2, 4]]

    assert coverage == sorted(coverage)
    assert unresolved == sorted(unresolved, reverse=True)
