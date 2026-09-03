from __future__ import annotations

import csv
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


def test_build_adjustment_rows_joins_direction_specific_covariates_and_uses_strict_boundary() -> None:
    balance = [
        {
            "direction": "A-bin/B-base",
            "bin": "(0,.25]",
            "dataset_index": 0,
            "exposure_step": 1,
            "baseline_p0": 0.10,
            "baseline_p0_completion_length": 100.0,
            "prompt_token_count": 30,
        },
        {
            "direction": "A-bin/B-base",
            "bin": "(0,.25]",
            "dataset_index": 1,
            "exposure_step": 2,
            "baseline_p0": 0.20,
            "baseline_p0_completion_length": 120.0,
            "prompt_token_count": 35,
        },
    ]
    movement = [
        {
            "snapshot_pct": 25,
            "direction": "A-bin/B-base",
            "bin": "(0,.25]",
            "dataset_index": i,
            "delta_R": 0.1,
            "delta_T": 0.2,
            "delta_C": 0.05,
        }
        for i in range(2)
    ]

    joined = esa.build_adjustment_rows(
        movement,
        balance,
        snapshot_schedule={25: 2},
        target_pcts=(25,),
    )
    assert [row["exposure_status"] for row in joined] == ["exposed", "unexposed"]
    assert joined[0]["baseline_p0"] == 0.10
    assert joined[1]["prompt_token_count"] == 35


def test_end_to_end_adjusted_analysis_writes_symmetric_output_and_records_small_cells(tmp_path) -> None:
    balance_path = tmp_path / "balance.csv"
    movement_path = tmp_path / "movement.csv"
    output_dir = tmp_path / "out"

    balance_rows: list[dict] = []
    movement_rows: list[dict] = []
    for direction in ("A-bin/B-base", "B-bin/A-base"):
        for i in range(8):
            p0 = 0.02 * i
            p0_len = 100.0 + 7.0 * i + (i % 2)
            prompt = 30.0 + (i % 3) + i / 10.0
            exposed = i < 4
            base = 0.05 + 0.25 * p0 - 0.0003 * p0_len + 0.001 * prompt
            balance_rows.append(
                {
                    "direction": direction,
                    "bin": "(0,.25]",
                    "dataset_index": i,
                    "exposure_step": i,
                    "baseline_p0": p0,
                    "baseline_p0_completion_length": p0_len,
                    "prompt_token_count": prompt,
                }
            )
            movement_rows.append(
                {
                    "snapshot_pct": 25,
                    "direction": direction,
                    "dataset_index": i,
                    "bin": "(0,.25]",
                    "delta_R": base + (0.06 if exposed else 0.0),
                    "delta_T": 2 * base + (0.03 if exposed else 0.0),
                    "delta_C": base + (0.06 if exposed else 0.0),
                }
            )

        # A deliberately unestimable boundary bin: one exposed and one unexposed.
        for i, step in ((100, 0), (101, 7)):
            balance_rows.append(
                {
                    "direction": direction,
                    "bin": "(.75,1)",
                    "dataset_index": i,
                    "exposure_step": step,
                    "baseline_p0": 0.9,
                    "baseline_p0_completion_length": 100.0,
                    "prompt_token_count": 20.0,
                }
            )
            movement_rows.append(
                {
                    "snapshot_pct": 25,
                    "direction": direction,
                    "dataset_index": i,
                    "bin": "(.75,1)",
                    "delta_R": 0.0,
                    "delta_T": 0.0,
                    "delta_C": 0.0,
                }
            )

    with balance_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(balance_rows[0]))
        writer.writeheader()
        writer.writerows(balance_rows)
    with movement_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(movement_rows[0]))
        writer.writeheader()
        writer.writerows(movement_rows)

    result = esa.run_analysis(
        balance_question_csv=balance_path,
        per_question_csv=movement_path,
        output_dir=output_dir,
        snapshot_schedule={25: 4},
        target_pcts=(25,),
    )

    assert len(result["adjusted_symmetric"]) == 1
    symmetric = result["adjusted_symmetric"][0]
    assert symmetric["bin"] == "(0,.25]"
    assert math.isclose(symmetric["adjusted_gap_C_unexposed_minus_exposed"], -0.06, abs_tol=1e-10)
    assert symmetric["adjusted_transfer_classification"] == "mixed_or_uncertain"
    assert len(result["skipped_cells"]) == 2
    assert all(row["reason"] == "group_size_lt_2" for row in result["skipped_cells"])
    assert (output_dir / "adjusted_directional.csv").exists()
    assert (output_dir / "adjusted_symmetric.csv").exists()
    assert (output_dir / "adjusted_skipped_cells.csv").exists()
