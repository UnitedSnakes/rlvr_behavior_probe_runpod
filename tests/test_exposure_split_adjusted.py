from __future__ import annotations

import math

from analyses import exposure_split_adjusted as esa


def _synthetic_rows(*, treatment_effect: float) -> list[dict]:
    rows: list[dict] = []
    exposed_indices = {0, 1, 2, 6, 7, 9}
    for i in range(12):
        exposed = i in exposed_indices
        p0 = 0.02 * i
        p0_len = 100.0 + 11.0 * i + 3.0 * (i % 3)
        prompt = 30.0 + 2.0 * (i % 5) + float(i // 4)
        base = 0.04 + 0.30 * p0 - 0.0002 * p0_len + 0.001 * prompt
        effect = treatment_effect if exposed else 0.0
        rows.append(
            {
                "snapshot_pct": 25,
                "snapshot_step": 934,
                "direction": "A-bin/B-base",
                "bin": "(0,.25]",
                "dataset_index": i,
                "exposure_status": "exposed" if exposed else "unexposed",
                "baseline_p0": p0,
                "baseline_p0_completion_length": p0_len,
                "prompt_token_count": prompt,
                "delta_R": base + effect,
                "delta_T": 2.0 * base + 0.5 * effect,
                "delta_C": base + effect,
            }
        )
    return rows


def test_adjusted_cell_removes_linear_preoutcome_imbalance_when_treatment_effect_is_zero() -> None:
    row = esa.fit_adjusted_cell(_synthetic_rows(treatment_effect=0.0))

    assert row["n_questions"] == 12
    assert row["n_exposed"] == 6
    assert row["n_unexposed"] == 6
    assert row["n_covariates_used"] == 3
    assert math.isclose(row["adjusted_delta_C_exposed"], row["adjusted_delta_C_unexposed"], abs_tol=1e-10)
    assert math.isclose(row["adjusted_gap_C_unexposed_minus_exposed"], 0.0, abs_tol=1e-10)


def test_adjusted_cell_recovers_known_exposed_effect_and_applies_gap_classifier() -> None:
    row = esa.fit_adjusted_cell(_synthetic_rows(treatment_effect=0.06))

    assert math.isclose(
        row["adjusted_delta_C_exposed"] - row["adjusted_delta_C_unexposed"],
        0.06,
        abs_tol=1e-10,
    )
    assert math.isclose(row["adjusted_gap_C_unexposed_minus_exposed"], -0.06, abs_tol=1e-10)
    assert row["adjusted_transfer_classification"] == "mixed_or_uncertain"
    assert math.isclose(
        row["adjusted_ratio_C_unexposed_over_exposed"],
        row["adjusted_delta_C_unexposed"] / row["adjusted_delta_C_exposed"],
        abs_tol=1e-10,
    )


def test_zero_variance_covariate_is_dropped_without_changing_treatment_contrast() -> None:
    rows = _synthetic_rows(treatment_effect=0.06)
    for row in rows:
        row["prompt_token_count"] = 42.0
        # Remove prompt from the outcome-generating equation as well so the
        # frozen linear adjustment remains correctly specified.
        p0 = row["baseline_p0"]
        p0_len = row["baseline_p0_completion_length"]
        exposed = row["exposure_status"] == "exposed"
        base = 0.04 + 0.30 * p0 - 0.0002 * p0_len
        row["delta_R"] = base + (0.06 if exposed else 0.0)
        row["delta_T"] = 2.0 * base + (0.03 if exposed else 0.0)
        row["delta_C"] = base + (0.06 if exposed else 0.0)

    fitted = esa.fit_adjusted_cell(rows)
    assert fitted["n_covariates_used"] == 2
    assert "prompt_token_count" in fitted["dropped_covariates"]
    assert math.isclose(
        fitted["adjusted_delta_C_exposed"] - fitted["adjusted_delta_C_unexposed"],
        0.06,
        abs_tol=1e-10,
    )


def test_symmetrization_equal_weights_crossfit_directions() -> None:
    directional = [
        {
            "snapshot_pct": 25,
            "snapshot_step": 934,
            "direction": "A-bin/B-base",
            "bin": "(0,.25]",
            "n_questions": 80,
            "n_exposed": 20,
            "n_unexposed": 60,
            "adjusted_delta_R_exposed": 0.20,
            "adjusted_delta_R_unexposed": 0.18,
            "adjusted_delta_T_exposed": 0.30,
            "adjusted_delta_T_unexposed": 0.28,
            "adjusted_delta_C_exposed": 0.12,
            "adjusted_delta_C_unexposed": 0.10,
        },
        {
            "snapshot_pct": 25,
            "snapshot_step": 934,
            "direction": "B-bin/A-base",
            "bin": "(0,.25]",
            "n_questions": 20,
            "n_exposed": 5,
            "n_unexposed": 15,
            "adjusted_delta_R_exposed": 0.10,
            "adjusted_delta_R_unexposed": 0.08,
            "adjusted_delta_T_exposed": 0.18,
            "adjusted_delta_T_unexposed": 0.16,
            "adjusted_delta_C_exposed": 0.08,
            "adjusted_delta_C_unexposed": 0.06,
        },
    ]

    rows = esa.symmetrize_adjusted_directional(directional)
    assert len(rows) == 1
    row = rows[0]
    # Direction means are equal-weighted despite the 80-vs-20 cell sizes.
    assert math.isclose(row["adjusted_delta_C_exposed"], 0.10)
    assert math.isclose(row["adjusted_delta_C_unexposed"], 0.08)
    assert math.isclose(row["adjusted_gap_C_unexposed_minus_exposed"], -0.02)
    assert row["adjusted_transfer_classification"] == "transfer_compatible"
