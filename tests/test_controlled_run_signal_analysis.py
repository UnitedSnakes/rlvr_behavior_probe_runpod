from __future__ import annotations

import importlib
import math

import numpy as np


def test_sequence_mean_logprob_diff_summary_tracks_top_p_band():
    analysis = importlib.import_module("controlled_run.signal_analysis")
    rows = [
        {"raw_log_rho": -5.0, "completion_length": 100},
        {"raw_log_rho": -2.0, "completion_length": 100},
        {"raw_log_rho": 0.2, "completion_length": 100},
    ]

    summary = analysis.summarize_sequence_mean_logprob_diff(rows, top_p=0.95)

    assert summary["n"] == 3
    assert math.isclose(summary["top_p_log_lower_bound"], math.log(0.95), rel_tol=0, abs_tol=1e-12)
    assert math.isclose(summary["mean"], (-0.05 - 0.02 + 0.002) / 3, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(summary["fraction_in_top_p_band"], 2 / 3, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(summary["fraction_positive"], 1 / 3, rel_tol=0, abs_tol=1e-12)


def test_joint_regression_recovers_length_and_difficulty_with_group_cluster_se():
    analysis = importlib.import_module("controlled_run.signal_analysis")
    rows = []
    # Six prompt groups, four rollouts each. The response is exactly linear in
    # length and k/G so the OLS coefficients should be recovered regardless of
    # the repeated group-level difficulty covariate.
    for group in range(6):
        successes = group + 2
        group_size = 16
        for rollout in range(4):
            length = 100 + 10 * group + rollout
            difficulty = successes / group_size
            raw_log_rho = 1.25 - 0.02 * length + 3.5 * difficulty
            rows.append(
                {
                    "generation_global_step": group,
                    "dataset_index": 1000 + group,
                    "completion_length": length,
                    "group_successes": successes,
                    "group_size": group_size,
                    "raw_log_rho": raw_log_rho,
                }
            )

    result = analysis.fit_length_difficulty_regression(rows)

    assert result["n"] == 24
    assert result["n_clusters"] == 6
    assert np.allclose(
        [result["intercept"], result["length_coef"], result["success_fraction_coef"]],
        [1.25, -0.02, 3.5],
        atol=1e-10,
    )
    assert math.isclose(result["r_squared"], 1.0, rel_tol=0, abs_tol=1e-12)
    assert result["cluster_se_length"] >= 0.0
    assert result["cluster_se_success_fraction"] >= 0.0
