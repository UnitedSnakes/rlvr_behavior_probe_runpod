from __future__ import annotations

import pytest

from analyses.maxrl_objective_comparison import build_objective_comparison


def _row(signal: float, delta_c: float) -> dict:
    return {
        "cumulative_abs_advantage_per_panel_question": signal,
        "exploratory_dapo_is_abs_mass_per_panel_question": signal * 10,
        "actual_is_ess_fraction": 0.998,
        "delta_R": delta_c + 0.1,
        "delta_T": delta_c + 0.2,
        "delta_C": delta_c,
    }


def test_build_objective_comparison_keeps_signal_and_behavior_separate():
    key = (100, "(0,.25]")
    rows = build_objective_comparison(
        {key: _row(2.0, 0.10)},
        {key: _row(6.0, 0.12)},
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["maxrl_over_grpo_signal"] == pytest.approx(3.0)
    assert row["maxrl_minus_grpo_delta_C"] == pytest.approx(0.02)
    assert row["maxrl_minus_grpo_delta_T"] == pytest.approx(0.02)
    assert row["maxrl_minus_grpo_delta_R"] == pytest.approx(0.02)


def test_build_objective_comparison_fails_closed_on_key_mismatch():
    with pytest.raises(ValueError, match="key mismatch"):
        build_objective_comparison(
            {(100, "(0,.25]"): _row(1.0, 0.1)},
            {(95, "(0,.25]"): _row(1.0, 0.1)},
        )
