from __future__ import annotations

import math

import pytest

from analyses.maxrl_h1_matched_prefix import (
    aggregate_objective_by_p0,
    assert_matched_prompt_schedule,
    finite_g_prediction_rows,
)


def _group(step: int, dataset_index: int, k: int, mass: float) -> dict:
    return {
        "generation_global_step": step,
        "dataset_index": dataset_index,
        "group_successes": k,
        "group_size": 16,
        "k_over_G": k / 16,
        "active_group": 0 < k < 16,
        "group_total_abs_advantage": mass,
        "actual_is_ratio_sum": 16.0,
        "actual_is_ratio_sq_sum": 16.0,
        "actual_is_ratio_count": 16,
        "actual_is_capped_token_mass": 0.0,
        "exploratory_dapo_is_abs_mass": mass,
    }


def test_finite_g_prediction_matches_actual_trl_group_scaling():
    rows = {row["K"]: row for row in finite_g_prediction_rows()}

    assert rows[0]["grpo_group_abs_advantage_mass"] == 0.0
    assert rows[0]["maxrl_group_abs_advantage_mass"] == 0.0
    assert rows[16]["grpo_group_abs_advantage_mass"] == 0.0
    assert rows[16]["maxrl_group_abs_advantage_mass"] == 0.0

    assert rows[1]["maxrl_group_abs_advantage_mass"] == 30.0
    assert rows[4]["maxrl_group_abs_advantage_mass"] == 24.0
    assert rows[8]["maxrl_group_abs_advantage_mass"] == 16.0
    assert rows[12]["maxrl_group_abs_advantage_mass"] == 8.0
    assert rows[15]["maxrl_group_abs_advantage_mass"] == 2.0

    assert math.isclose(
        rows[1]["maxrl_over_grpo_mass"],
        4.0016,
        rel_tol=0.0,
        abs_tol=1e-6,
    )
    assert rows[8]["maxrl_over_grpo_mass"] > 1.0
    assert rows[9]["maxrl_over_grpo_mass"] < 1.0


def test_matched_prompt_schedule_requires_exact_ordered_keys():
    grpo = [_group(0, 10, 1, 1.0), _group(0, 20, 2, 1.0)]
    maxrl = [_group(0, 10, 1, 1.0), _group(0, 20, 2, 1.0)]

    assert assert_matched_prompt_schedule(grpo, maxrl) == [(0, 10), (0, 20)]

    with pytest.raises(ValueError, match="prompt schedule differs"):
        assert_matched_prompt_schedule(
            grpo,
            [_group(0, 10, 1, 1.0), _group(0, 21, 2, 1.0)],
        )


def test_signal_weighted_mean_p0_moves_left_when_low_p0_mass_increases():
    p0_records = [
        {"dataset_index": 0, "p0_A": 0.125, "p0_B": 0.1875},
        {"dataset_index": 1, "p0_A": 0.75, "p0_B": 0.8125},
    ]
    panel = [0, 1]

    _, _, grpo_weighted = aggregate_objective_by_p0(
        objective="GRPO",
        p0_records=p0_records,
        groups=[
            _group(0, 0, 1, 1.0),
            _group(0, 1, 12, 1.0),
        ],
        panel_indices=panel,
    )
    _, _, maxrl_weighted = aggregate_objective_by_p0(
        objective="MaxRL",
        p0_records=p0_records,
        groups=[
            _group(0, 0, 1, 4.0),
            _group(0, 1, 12, 0.5),
        ],
        panel_indices=panel,
    )

    g = {
        row["direction"]: row["signal_weighted_mean_p0"]
        for row in grpo_weighted
    }
    m = {
        row["direction"]: row["signal_weighted_mean_p0"]
        for row in maxrl_weighted
    }

    assert m["A-bin"] < g["A-bin"]
    assert m["B-bin"] < g["B-bin"]


def test_k0_groups_are_counted_but_contribute_zero_mass():
    p0_records = [
        {"dataset_index": 0, "p0_A": 0.125, "p0_B": 0.125},
    ]
    directional, _, _ = aggregate_objective_by_p0(
        objective="MaxRL",
        p0_records=p0_records,
        groups=[
            _group(0, 0, 0, 0.0),
            _group(1, 0, 1, 30.0),
        ],
        panel_indices=[0],
    )

    row = next(
        row
        for row in directional
        if row["direction"] == "A-bin" and row["bin"] == "(0,.25]"
    )
    assert row["zero_group_fraction"] == 0.5
    assert row["active_group_fraction"] == 0.5
    assert row["cumulative_abs_advantage_per_panel_question"] == 30.0
