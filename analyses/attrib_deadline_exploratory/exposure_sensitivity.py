"""Post-outcome question-resampling diagnostic for the frozen exposure analysis.

This is conditional on one observed training trajectory and the observed
response banks. Percentile ranges describe sensitivity to question composition.
They are not between-training-seed intervals, randomized-treatment intervals,
or a complete account of fixed-panel response-sampling uncertainty.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "analyses" / "canonical_exposure_split_adjusted"
OUTPUT_DIR = Path(__file__).resolve().parent
REPETITIONS = 3000
RANDOM_SEED = 20260906
COVARIATES = (
    "baseline_p0",
    "baseline_p0_completion_length",
    "prompt_token_count",
)
DIRECTIONS = ("A-bin/B-base", "B-bin/A-base")
BINS = ("0", "(0,.25]", "(.25,.5]", "(.5,.75]", "(.75,1)")
CUTOFFS = (25, 45, 65)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _prepare_cells(rows: list[dict[str, str]]):
    question_ids = sorted({int(row["dataset_index"]) for row in rows})
    if question_ids != list(range(256)):
        raise ValueError("expected exactly GSM8K train indices 0..255")
    question_positions = {
        question_id: index for index, question_id in enumerate(question_ids)
    }

    grouped: dict[tuple[int, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["bin"] in BINS:
            key = (int(row["snapshot_pct"]), row["bin"], row["direction"])
            grouped[key].append(row)

    cells = {}
    for key, cell_rows in grouped.items():
        question_indices = np.asarray(
            [question_positions[int(row["dataset_index"])] for row in cell_rows],
            dtype=int,
        )
        exposure = np.asarray(
            [float(row["exposure_status"] == "exposed") for row in cell_rows],
            dtype=float,
        )
        covariates = np.asarray(
            [[float(row[name]) for name in COVARIATES] for row in cell_rows],
            dtype=float,
        )
        standard_deviation = covariates.std(axis=0, ddof=1)
        keep = standard_deviation > 0
        standardized = (
            covariates[:, keep] - covariates[:, keep].mean(axis=0)
        ) / standard_deviation[keep]
        design = np.column_stack([np.ones(len(cell_rows)), exposure, standardized])
        outcome = np.asarray([float(row["delta_C"]) for row in cell_rows], dtype=float)
        cells[key] = (question_indices, design, outcome)

    expected_keys = {
        (cutoff, bin_name, direction)
        for cutoff in CUTOFFS
        for bin_name in BINS
        for direction in DIRECTIONS
    }
    missing = sorted(expected_keys - set(cells))
    if missing:
        raise ValueError(f"missing frozen exposure cells: {missing}")
    return question_ids, cells


def _fit_cell(cell, multiplicities: np.ndarray) -> np.ndarray:
    question_indices, design, outcome = cell
    weights = multiplicities[question_indices]
    retained = weights > 0
    if not retained.any():
        return np.full(3, np.nan)

    root_weight = np.sqrt(weights[retained])
    weighted_design = design[retained] * root_weight[:, None]
    weighted_outcome = outcome[retained] * root_weight
    coefficients, _, rank, _ = np.linalg.lstsq(
        weighted_design,
        weighted_outcome,
        rcond=None,
    )
    if rank != design.shape[1]:
        return np.full(3, np.nan)

    # Predict at the resampled cell's pooled covariate mean. Keeping the
    # original affine standardization is algebraically equivalent to
    # re-standardizing the same retained covariates within each resample.
    mean_covariates = np.average(design[:, 2:], axis=0, weights=weights)
    unexposed = coefficients[0] + coefficients[2:] @ mean_covariates
    exposed = unexposed + coefficients[1]
    return np.asarray([exposed, unexposed, -coefficients[1]])


def run_analysis(
    *,
    repetitions: int = REPETITIONS,
    random_seed: int = RANDOM_SEED,
    output_dir: Path = OUTPUT_DIR,
) -> dict:
    rows = _read_csv(SOURCE_DIR / "adjustment_input_rows.csv")
    frozen = _read_csv(SOURCE_DIR / "adjusted_symmetric.csv")
    question_ids, cells = _prepare_cells(rows)

    keys = [(cutoff, bin_name) for cutoff in CUTOFFS for bin_name in BINS]
    ones = np.ones(len(question_ids), dtype=int)
    point_estimates: dict[tuple[int, str], np.ndarray] = {}
    for key in keys:
        point = np.mean(
            [_fit_cell(cells[(*key, direction)], ones) for direction in DIRECTIONS],
            axis=0,
        )
        source = next(
            row
            for row in frozen
            if int(row["snapshot_pct"]) == key[0] and row["bin"] == key[1]
        )
        expected = np.asarray(
            [
                float(source["adjusted_delta_C_exposed"]),
                float(source["adjusted_delta_C_unexposed"]),
                float(source["adjusted_gap_C_unexposed_minus_exposed"]),
            ]
        )
        maximum_error = float(np.max(np.abs(point - expected)))
        if maximum_error >= 1e-12:
            raise AssertionError(
                f"failed to reproduce frozen point estimate {key}: {maximum_error}"
            )
        point_estimates[key] = point

    generator = np.random.default_rng(random_seed)
    bootstrap_weights = generator.multinomial(
        len(question_ids),
        np.ones(len(question_ids)) / len(question_ids),
        size=repetitions,
    )
    bootstrap = {key: np.full((repetitions, 3), np.nan) for key in keys}
    for repetition, multiplicities in enumerate(bootstrap_weights):
        # One question multiplicity is shared across all cutoffs and both
        # cross-fit directions, preserving the paired question structure.
        for key in keys:
            fitted = np.asarray(
                [
                    _fit_cell(cells[(*key, direction)], multiplicities)
                    for direction in DIRECTIONS
                ]
            )
            if np.isfinite(fitted).all():
                bootstrap[key][repetition] = fitted.mean(axis=0)

    output_rows: list[dict] = []
    for key in keys:
        cutoff, bin_name = key
        valid = np.isfinite(bootstrap[key]).all(axis=1)
        interval = np.percentile(bootstrap[key][valid], [2.5, 97.5], axis=0)
        result = {
            "cutoff": cutoff,
            "bin": bin_name,
            "valid_replicates": int(valid.sum()),
        }
        for index, metric in enumerate(("exposed", "unexposed", "gap_U_minus_E")):
            result[f"{metric}_pp"] = 100 * point_estimates[key][index]
            result[f"{metric}_lower_pp"] = 100 * interval[0, index]
            result[f"{metric}_upper_pp"] = 100 * interval[1, index]
        output_rows.append(result)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "exposure_question_bootstrap.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    # Leave-one-question-out sensitivity for the primary low nonzero bin.
    leave_one_out = {}
    for cutoff in CUTOFFS:
        key = (cutoff, "(0,.25]")
        results = []
        for omitted_index in range(len(question_ids)):
            multiplicities = ones.copy()
            multiplicities[omitted_index] = 0
            fitted = np.asarray(
                [
                    _fit_cell(cells[(*key, direction)], multiplicities)
                    for direction in DIRECTIONS
                ]
            )
            if not np.isfinite(fitted).all():
                raise AssertionError(
                    f"unexpected singular leave-one-out fit at cutoff {cutoff}, "
                    f"question {omitted_index}"
                )
            results.append(fitted.mean(axis=0))
        results = np.asarray(results)
        leave_one_out[cutoff] = {
            "unexposed_min_max_pp": (
                100 * np.asarray([results[:, 1].min(), results[:, 1].max()])
            ).tolist(),
            "gap_min_max_pp": (
                100 * np.asarray([results[:, 2].min(), results[:, 2].max()])
            ).tolist(),
        }

    summary = {
        "status": "post-outcome exploratory diagnostic",
        "repetitions": repetitions,
        "random_seed": random_seed,
        "question_count": len(question_ids),
        "interval": "pointwise 2.5/97.5 percentile; no multiplicity correction",
        "conditioning": (
            "one observed training trajectory, frozen bin memberships, "
            "observed response banks"
        ),
        "low_bin": [row for row in output_rows if row["bin"] == "(0,.25]"],
        "leave_one_out_low_bin": leave_one_out,
    }
    (output_dir / "exposure_question_bootstrap_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    print(json.dumps(run_analysis(), indent=2))


if __name__ == "__main__":
    main()
