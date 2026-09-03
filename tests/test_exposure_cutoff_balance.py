from __future__ import annotations

import math

from analyses import exposure_cutoff_balance as ecb


def _rows():
    return [
        {
            "direction": "A-bin/B-base",
            "bin": "(0,.25]",
            "dataset_index": i,
            "exposure_step": i,
            "baseline_p0": float(i),
            "baseline_p0_completion_length": float(10 * (i + 1)),
            "prompt_token_count": float(100 + 10 * i),
        }
        for i in range(4)
    ]


def test_cutoff_balance_uses_exact_snapshot_boundary_and_pooled_smd() -> None:
    out = ecb.summarize_cutoff_balance(
        _rows(),
        snapshot_schedule={25: 2},
        target_pcts=(25,),
    )
    assert len(out) == 1
    row = out[0]
    assert row["snapshot_pct"] == 25
    assert row["snapshot_step"] == 2
    assert row["direction"] == "A-bin/B-base"
    assert row["bin"] == "(0,.25]"
    assert row["n_exposed"] == 2
    assert row["n_unexposed"] == 2

    # exposed iff exposure_step < snapshot_step, so exposed={0,1}, unexposed={2,3}
    assert math.isclose(row["baseline_p0_mean_exposed"], 0.5)
    assert math.isclose(row["baseline_p0_mean_unexposed"], 2.5)
    assert math.isclose(row["baseline_p0_diff_exposed_minus_unexposed"], -2.0)
    assert math.isclose(
        row["baseline_p0_smd_exposed_minus_unexposed"],
        -2.0 / math.sqrt(0.5),
    )

    assert math.isclose(row["p0_completion_length_diff_exposed_minus_unexposed"], -20.0)
    assert math.isclose(row["prompt_tokens_diff_exposed_minus_unexposed"], -20.0)


def test_cutoff_balance_smd_is_na_when_pooled_sd_is_zero_or_group_too_small() -> None:
    rows = [
        {
            "direction": "A-bin/B-base",
            "bin": "(0,.25]",
            "dataset_index": 0,
            "exposure_step": 0,
            "baseline_p0": 0.1,
            "baseline_p0_completion_length": 10.0,
            "prompt_token_count": 20.0,
        },
        {
            "direction": "A-bin/B-base",
            "bin": "(0,.25]",
            "dataset_index": 1,
            "exposure_step": 2,
            "baseline_p0": 0.1,
            "baseline_p0_completion_length": 10.0,
            "prompt_token_count": 20.0,
        },
    ]
    out = ecb.summarize_cutoff_balance(rows, snapshot_schedule={25: 1}, target_pcts=(25,))
    row = out[0]
    assert row["n_exposed"] == 1
    assert row["n_unexposed"] == 1
    assert row["baseline_p0_smd_exposed_minus_unexposed"] is None
    assert row["p0_completion_length_smd_exposed_minus_unexposed"] is None
    assert row["prompt_tokens_smd_exposed_minus_unexposed"] is None


def test_primary_gap_criterion_is_frozen_before_outcome_deblinding() -> None:
    assert math.isclose(ecb.DELTA_C_EXPOSED_MIN, 0.02)
    assert math.isclose(ecb.GAP_SMALL, 0.04)
    assert math.isclose(ecb.GAP_LARGE, 0.08)

    transfer_like = ecb.classify_delta_c_gap(delta_c_exposed=0.10, delta_c_unexposed=0.09)
    assert math.isclose(transfer_like["gap_unexposed_minus_exposed"], -0.01)
    assert transfer_like["classification"] == "transfer_compatible"

    mixed = ecb.classify_delta_c_gap(delta_c_exposed=0.10, delta_c_unexposed=0.04)
    assert math.isclose(mixed["gap_unexposed_minus_exposed"], -0.06)
    assert mixed["classification"] == "mixed_or_uncertain"

    own = ecb.classify_delta_c_gap(delta_c_exposed=0.10, delta_c_unexposed=0.01)
    assert math.isclose(own["gap_unexposed_minus_exposed"], -0.09)
    assert own["classification"] == "own_exposure_candidate"

    reverse = ecb.classify_delta_c_gap(delta_c_exposed=0.10, delta_c_unexposed=0.15)
    assert reverse["classification"] == "unexposed_higher"

    too_small = ecb.classify_delta_c_gap(delta_c_exposed=0.019, delta_c_unexposed=0.018)
    assert too_small["classification"] == "not_classifiable"
    assert too_small["ratio_unexposed_over_exposed"] is None


def test_ratio_is_secondary_reference_not_primary_classifier() -> None:
    result = ecb.classify_delta_c_gap(delta_c_exposed=0.10, delta_c_unexposed=0.04)
    assert math.isclose(result["ratio_unexposed_over_exposed"], 0.4)
    assert result["classification"] == "mixed_or_uncertain"
