from __future__ import annotations

import math

from analyses import exposure_split_transfer as est


def _rollout(*, reward: int, n_tokens: int) -> dict:
    return {
        "canonical_reward": reward,
        "terminated": True,
        "correct": bool(reward),
        "n_tokens": n_tokens,
    }


def _p0_record(index: int, *, p_a: int, p_b: int, len_a: int, len_b: int) -> dict:
    # K=4 fixture: p_a/p_b are success counts in each half.
    a = [_rollout(reward=int(i < p_a), n_tokens=len_a) for i in range(4)]
    b = [_rollout(reward=int(i < p_b), n_tokens=len_b) for i in range(4)]
    all_rows = a + b
    return {
        "dataset_index": index,
        "question": f"question {index}",
        "half_size": 4,
        "n_rollouts": 8,
        "p0_A": p_a / 4,
        "p0_B": p_b / 4,
        "p0": (p_a + p_b) / 8,
        "correctness_p0": sum(r["correct"] for r in all_rows) / 8,
        "termination_rate": 1.0,
        "rollouts_A": a,
        "rollouts_B": b,
    }


def test_balance_rows_use_opposite_crossfit_half_for_baseline_covariates() -> None:
    p0 = [
        _p0_record(0, p_a=1, p_b=0, len_a=10, len_b=100),
        _p0_record(1, p_a=1, p_b=1, len_a=20, len_b=200),
        _p0_record(2, p_a=1, p_b=2, len_a=30, len_b=300),
        _p0_record(3, p_a=1, p_b=3, len_a=40, len_b=400),
    ]
    exposure_steps = {0: 0, 1: 1, 2: 2, 3: 3}
    prompt_tokens = {0: 11, 1: 22, 2: 33, 3: 44}

    rows = est.build_balance_rows(
        p0,
        exposure_steps=exposure_steps,
        prompt_token_counts=prompt_tokens,
    )

    a_rows = [
        row for row in rows
        if row["direction"] == "A-bin/B-base" and row["bin"] == "(0,.25]"
    ]
    assert len(a_rows) == 4
    assert [row["baseline_p0"] for row in a_rows] == [0.0, 0.25, 0.5, 0.75]
    assert [row["baseline_p0_completion_length"] for row in a_rows] == [100.0, 200.0, 300.0, 400.0]
    assert [row["prompt_token_count"] for row in a_rows] == [11, 22, 33, 44]


def test_balance_summary_reports_correlations_and_uniformity_before_split() -> None:
    rows = [
        {
            "direction": "A-bin/B-base",
            "bin": "(0,.25]",
            "dataset_index": i,
            "exposure_step": i,
            "baseline_p0": float(i),
            "baseline_p0_completion_length": float(10 * i),
            "prompt_token_count": float(100 - 10 * i),
        }
        for i in range(4)
    ]

    summary = est.summarize_balance(rows, total_steps=4)
    assert len(summary) == 1
    row = summary[0]
    assert row["n_questions"] == 4
    assert math.isclose(row["corr_exposure_baseline_p0"], 1.0)
    assert math.isclose(row["corr_exposure_p0_completion_length"], 1.0)
    assert math.isclose(row["corr_exposure_prompt_tokens"], -1.0)
    # Midpoint-normalized exposure positions are .125, .375, .625, .875.
    assert math.isclose(row["uniform_ks_D"], 0.125)
    assert math.isclose(row["exposure_q1_fraction"], 0.25)
    assert math.isclose(row["exposure_q2_fraction"], 0.25)
    assert math.isclose(row["exposure_q3_fraction"], 0.25)
    assert math.isclose(row["exposure_q4_fraction"], 0.25)


def test_transfer_ratio_criterion_is_frozen_before_outcomes() -> None:
    assert est.OWN_EXPOSURE_RATIO_MAX == 0.25
    assert est.TRANSFER_RATIO_MIN == 0.75

    transfer = est.classify_transfer_ratio(delta_c_exposed=0.10, delta_c_unexposed=0.09)
    assert math.isclose(transfer["ratio"], 0.9)
    assert transfer["classification"] == "transfer_dominant"

    own = est.classify_transfer_ratio(delta_c_exposed=0.10, delta_c_unexposed=0.01)
    assert math.isclose(own["ratio"], 0.1)
    assert own["classification"] == "own_exposure_dominant"

    mixed = est.classify_transfer_ratio(delta_c_exposed=0.10, delta_c_unexposed=0.05)
    assert math.isclose(mixed["ratio"], 0.5)
    assert mixed["classification"] == "mixed"

    undefined = est.classify_transfer_ratio(delta_c_exposed=0.0, delta_c_unexposed=0.05)
    assert undefined["ratio"] is None
    assert undefined["classification"] == "not_classifiable"
