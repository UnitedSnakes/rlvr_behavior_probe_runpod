#!/usr/bin/env python3
"""Covariate-adjusted exposed-vs-unexposed movement contrasts.

The model is frozen in the pre-outcome checkpoint before real split outcomes are
inspected. Within each snapshot x cross-fit direction x frozen p0 bin, fit
separate OLS models for DeltaR, DeltaT, and DeltaC:

    DeltaX = alpha + tau * exposed + beta' z(pre-outcome covariates) + error

The three pre-outcome covariates are opposite-half p0 reward probability,
opposite-half pi0 completion length, and canonical prompt token count. Covariates
are centered/scaled within the cell; zero-variance covariates are dropped.
Adjusted group means are predictions at the pooled covariate mean, so the
adjusted unexposed-minus-exposed gap is -tau.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from analyses.exposure_cutoff_balance import classify_delta_c_gap
from analyses.exposure_split_transfer import (
    DEFAULT_SPLIT_PCTS,
    DIRECTIONS,
    load_per_question_movement,
)
from analyses.ledger_crossfit_signal_allocation import DEFAULT_SNAPSHOT_SCHEDULE
from analyses.snapshot_crossfit_trajectory import BIN_ORDER, write_csv


COVARIATES = (
    "baseline_p0",
    "baseline_p0_completion_length",
    "prompt_token_count",
)
METRICS = ("R", "T", "C")


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average empty values")
    return sum(values) / len(values)


def _sample_sd(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    return float(np.std(values, ddof=1))


def _cell_metadata(rows: list[dict]) -> tuple[int, int, str, str]:
    if not rows:
        raise ValueError("adjusted cell must be non-empty")
    pcts = {int(row["snapshot_pct"]) for row in rows}
    steps = {int(row["snapshot_step"]) for row in rows}
    directions = {row["direction"] for row in rows}
    bins = {row["bin"] for row in rows}
    if len(pcts) != 1 or len(steps) != 1 or len(directions) != 1 or len(bins) != 1:
        raise ValueError("adjusted cell rows must share snapshot, direction, and bin")
    pct = next(iter(pcts))
    step = next(iter(steps))
    direction = next(iter(directions))
    bin_label = next(iter(bins))
    if direction not in DIRECTIONS:
        raise ValueError(f"unexpected direction {direction}")
    if bin_label not in BIN_ORDER:
        raise ValueError(f"unexpected bin {bin_label}")
    return pct, step, direction, bin_label


def _design_matrix(rows: list[dict]) -> tuple[np.ndarray, list[str], list[str]]:
    exposed = np.asarray(
        [1.0 if row["exposure_status"] == "exposed" else 0.0 for row in rows],
        dtype=float,
    )
    statuses = {row["exposure_status"] for row in rows}
    if not statuses.issubset({"exposed", "unexposed"}):
        raise ValueError(f"unexpected exposure statuses {sorted(statuses)}")
    if statuses != {"exposed", "unexposed"}:
        raise ValueError("adjusted cell requires both exposed and unexposed questions")

    columns = [np.ones(len(rows), dtype=float), exposed]
    used: list[str] = []
    dropped: list[str] = []
    for name in COVARIATES:
        values = np.asarray([float(row[name]) for row in rows], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"non-finite covariate {name}")
        mean = float(np.mean(values))
        sd = _sample_sd(values)
        if sd <= 0.0:
            dropped.append(name)
            continue
        columns.append((values - mean) / sd)
        used.append(name)

    design = np.column_stack(columns)
    rank = int(np.linalg.matrix_rank(design))
    if rank != design.shape[1]:
        raise ValueError(
            f"rank-deficient adjusted design: rank={rank}, columns={design.shape[1]}"
        )
    return design, used, dropped


def _fit_metric(design: np.ndarray, rows: list[dict], metric: str) -> dict[str, float]:
    key = f"delta_{metric}"
    y = np.asarray([float(row[key]) for row in rows], dtype=float)
    if not np.all(np.isfinite(y)):
        raise ValueError(f"non-finite outcome {key}")
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    intercept = float(coef[0])
    tau = float(coef[1])
    return {
        f"adjusted_delta_{metric}_unexposed": intercept,
        f"adjusted_delta_{metric}_exposed": intercept + tau,
        f"adjusted_gap_{metric}_unexposed_minus_exposed": -tau,
    }


def fit_adjusted_cell(rows: Iterable[dict]) -> dict:
    """Fit one frozen OLS cell and return raw + adjusted descriptive means."""
    cell = list(rows)
    pct, step, direction, bin_label = _cell_metadata(cell)
    design, used, dropped = _design_matrix(cell)
    exposed_rows = [row for row in cell if row["exposure_status"] == "exposed"]
    unexposed_rows = [row for row in cell if row["exposure_status"] == "unexposed"]

    out: dict = {
        "snapshot_pct": pct,
        "snapshot_step": step,
        "direction": direction,
        "bin": bin_label,
        "n_questions": len(cell),
        "n_exposed": len(exposed_rows),
        "n_unexposed": len(unexposed_rows),
        "n_covariates_used": len(used),
        "used_covariates": ";".join(used),
        "dropped_covariates": ";".join(dropped),
    }
    for metric in METRICS:
        key = f"delta_{metric}"
        raw_e = _mean([float(row[key]) for row in exposed_rows])
        raw_u = _mean([float(row[key]) for row in unexposed_rows])
        out[f"raw_delta_{metric}_exposed"] = raw_e
        out[f"raw_delta_{metric}_unexposed"] = raw_u
        out[f"raw_gap_{metric}_unexposed_minus_exposed"] = raw_u - raw_e
        out.update(_fit_metric(design, cell, metric))

    criterion = classify_delta_c_gap(
        delta_c_exposed=float(out["adjusted_delta_C_exposed"]),
        delta_c_unexposed=float(out["adjusted_delta_C_unexposed"]),
    )
    out["adjusted_gap_C_unexposed_minus_exposed"] = criterion[
        "gap_unexposed_minus_exposed"
    ]
    out["adjusted_ratio_C_unexposed_over_exposed"] = criterion[
        "ratio_unexposed_over_exposed"
    ]
    out["adjusted_transfer_classification"] = criterion["classification"]
    return out


def fit_adjusted_directional(rows: Iterable[dict]) -> list[dict]:
    """Fit all snapshot x direction x bin cells independently."""
    grouped: dict[tuple[int, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        direction = row["direction"]
        if direction not in DIRECTIONS:
            continue
        key = (int(row["snapshot_pct"]), direction, row["bin"])
        grouped[key].append(row)

    result: list[dict] = []
    for key, cell in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            DIRECTIONS.index(item[0][1]),
            BIN_ORDER.index(item[0][2]),
        ),
    ):
        del key
        result.append(fit_adjusted_cell(cell))
    return result


def symmetrize_adjusted_directional(directional_rows: Iterable[dict]) -> list[dict]:
    """Equal-weight A/B adjusted direction means within snapshot and bin."""
    indexed: dict[tuple[int, str, str], dict] = {}
    for row in directional_rows:
        direction = row["direction"]
        if direction not in DIRECTIONS:
            raise ValueError(f"unexpected direction {direction}")
        key = (int(row["snapshot_pct"]), direction, row["bin"])
        if key in indexed:
            raise ValueError(f"duplicate adjusted directional cell {key}")
        indexed[key] = row

    out: list[dict] = []
    for pct in sorted({key[0] for key in indexed}):
        for bin_label in BIN_ORDER:
            a = indexed.get((pct, "A-bin/B-base", bin_label))
            b = indexed.get((pct, "B-bin/A-base", bin_label))
            if a is None or b is None:
                continue
            if int(a["snapshot_step"]) != int(b["snapshot_step"]):
                raise ValueError("snapshot step mismatch between adjusted directions")
            row: dict = {
                "snapshot_pct": pct,
                "snapshot_step": int(a["snapshot_step"]),
                "direction": "symmetric",
                "bin": bin_label,
                "n_questions_A": int(a["n_questions"]),
                "n_questions_B": int(b["n_questions"]),
                "n_exposed_A": int(a["n_exposed"]),
                "n_exposed_B": int(b["n_exposed"]),
                "n_unexposed_A": int(a["n_unexposed"]),
                "n_unexposed_B": int(b["n_unexposed"]),
            }
            for metric in METRICS:
                for status in ("exposed", "unexposed"):
                    adjusted = f"adjusted_delta_{metric}_{status}"
                    row[adjusted] = (float(a[adjusted]) + float(b[adjusted])) / 2.0
                    raw = f"raw_delta_{metric}_{status}"
                    if raw in a and raw in b:
                        row[raw] = (float(a[raw]) + float(b[raw])) / 2.0
                row[f"adjusted_gap_{metric}_unexposed_minus_exposed"] = (
                    float(row[f"adjusted_delta_{metric}_unexposed"])
                    - float(row[f"adjusted_delta_{metric}_exposed"])
                )
                raw_e = f"raw_delta_{metric}_exposed"
                raw_u = f"raw_delta_{metric}_unexposed"
                if raw_e in row and raw_u in row:
                    row[f"raw_gap_{metric}_unexposed_minus_exposed"] = (
                        float(row[raw_u]) - float(row[raw_e])
                    )

            criterion = classify_delta_c_gap(
                delta_c_exposed=float(row["adjusted_delta_C_exposed"]),
                delta_c_unexposed=float(row["adjusted_delta_C_unexposed"]),
            )
            row["adjusted_gap_C_unexposed_minus_exposed"] = criterion[
                "gap_unexposed_minus_exposed"
            ]
            row["adjusted_ratio_C_unexposed_over_exposed"] = criterion[
                "ratio_unexposed_over_exposed"
            ]
            row["adjusted_transfer_classification"] = criterion["classification"]
            out.append(row)
    return out


def load_balance_question_rows(path: Path) -> list[dict]:
    """Read direction-specific pre-outcome balance covariates."""
    required = {
        "direction",
        "bin",
        "dataset_index",
        "exposure_step",
        "baseline_p0",
        "baseline_p0_completion_length",
        "prompt_token_count",
    }
    rows: list[dict] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"balance CSV missing required fields {sorted(required)}: {reader.fieldnames}"
            )
        for raw in reader:
            direction = raw["direction"]
            if direction not in DIRECTIONS:
                continue
            rows.append(
                {
                    "direction": direction,
                    "bin": raw["bin"],
                    "dataset_index": int(raw["dataset_index"]),
                    "exposure_step": int(raw["exposure_step"]),
                    "baseline_p0": float(raw["baseline_p0"]),
                    "baseline_p0_completion_length": float(
                        raw["baseline_p0_completion_length"]
                    ),
                    "prompt_token_count": float(raw["prompt_token_count"]),
                }
            )
    if not rows:
        raise ValueError(f"no direction-level balance rows found in {path}")
    return rows


def build_adjustment_rows(
    movement_rows: Iterable[dict],
    balance_rows: Iterable[dict],
    *,
    snapshot_schedule: dict[int, int],
    target_pcts: Iterable[int] = DEFAULT_SPLIT_PCTS,
) -> list[dict]:
    """Join frozen covariates to movement rows and assign own-exposure status."""
    target = tuple(int(pct) for pct in target_pcts)
    if len(set(target)) != len(target):
        raise ValueError("target_pcts must be unique")
    schedule = {int(pct): int(step) for pct, step in snapshot_schedule.items()}
    missing = sorted(set(target) - set(schedule))
    if missing:
        raise ValueError(f"target snapshots missing from schedule: {missing}")

    balance_index: dict[tuple[str, int], dict] = {}
    for row in balance_rows:
        direction = row["direction"]
        if direction not in DIRECTIONS:
            continue
        key = (direction, int(row["dataset_index"]))
        if key in balance_index:
            raise ValueError(f"duplicate balance row {key}")
        balance_index[key] = row

    out: list[dict] = []
    seen: set[tuple[int, str, int]] = set()
    for raw in movement_rows:
        pct = int(raw["snapshot_pct"])
        direction = raw["direction"]
        if pct not in target or direction not in DIRECTIONS:
            continue
        index = int(raw["dataset_index"])
        unique = (pct, direction, index)
        if unique in seen:
            raise ValueError(f"duplicate movement row {unique}")
        seen.add(unique)
        balance = balance_index.get((direction, index))
        if balance is None:
            raise ValueError(f"missing balance row for direction={direction}, dataset_index={index}")
        if raw["bin"] != balance["bin"]:
            raise ValueError(
                f"bin mismatch for direction={direction}, dataset_index={index}: "
                f"movement={raw['bin']} balance={balance['bin']}"
            )
        snapshot_step = schedule[pct]
        exposure_step = int(balance["exposure_step"])
        out.append(
            {
                "snapshot_pct": pct,
                "snapshot_step": snapshot_step,
                "direction": direction,
                "bin": raw["bin"],
                "dataset_index": index,
                "exposure_step": exposure_step,
                "exposure_status": "exposed" if exposure_step < snapshot_step else "unexposed",
                "baseline_p0": float(balance["baseline_p0"]),
                "baseline_p0_completion_length": float(
                    balance["baseline_p0_completion_length"]
                ),
                "prompt_token_count": float(balance["prompt_token_count"]),
                "delta_R": float(raw["delta_R"]),
                "delta_T": float(raw["delta_T"]),
                "delta_C": float(raw["delta_C"]),
            }
        )
    if not out:
        raise ValueError("no movement rows matched target snapshots/directions")
    return out


def _fit_eligible_cells(rows: Iterable[dict]) -> tuple[list[dict], list[dict]]:
    grouped: dict[tuple[int, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["snapshot_pct"]), row["direction"], row["bin"])].append(row)

    directional: list[dict] = []
    skipped: list[dict] = []
    for (pct, direction, bin_label), cell in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            DIRECTIONS.index(item[0][1]),
            BIN_ORDER.index(item[0][2]),
        ),
    ):
        n_exposed = sum(row["exposure_status"] == "exposed" for row in cell)
        n_unexposed = sum(row["exposure_status"] == "unexposed" for row in cell)
        if min(n_exposed, n_unexposed) < 2:
            skipped.append(
                {
                    "snapshot_pct": pct,
                    "snapshot_step": int(cell[0]["snapshot_step"]),
                    "direction": direction,
                    "bin": bin_label,
                    "n_questions": len(cell),
                    "n_exposed": n_exposed,
                    "n_unexposed": n_unexposed,
                    "reason": "group_size_lt_2",
                }
            )
            continue
        directional.append(fit_adjusted_cell(cell))
    return directional, skipped


def run_analysis(
    *,
    balance_question_csv: Path,
    per_question_csv: Path,
    output_dir: Path,
    snapshot_schedule: dict[int, int] = DEFAULT_SNAPSHOT_SCHEDULE,
    target_pcts: Iterable[int] = DEFAULT_SPLIT_PCTS,
) -> dict:
    """Run the frozen covariate-adjusted 25/45/65 exposure analysis."""
    balance = load_balance_question_rows(Path(balance_question_csv))
    movement = load_per_question_movement(Path(per_question_csv))
    joined = build_adjustment_rows(
        movement,
        balance,
        snapshot_schedule=snapshot_schedule,
        target_pcts=target_pcts,
    )
    directional, skipped = _fit_eligible_cells(joined)
    symmetric = symmetrize_adjusted_directional(directional)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    write_csv(destination / "adjustment_input_rows.csv", joined)
    write_csv(destination / "adjusted_directional.csv", directional)
    write_csv(destination / "adjusted_symmetric.csv", symmetric)
    write_csv(destination / "adjusted_skipped_cells.csv", skipped)

    return {
        "target_pcts": [int(pct) for pct in target_pcts],
        "adjustment_rows": joined,
        "adjusted_directional": directional,
        "adjusted_symmetric": symmetric,
        "skipped_cells": skipped,
        "output_dir": str(destination),
    }


def _fmt(value, *, scale: float = 1.0, digits: int = 2) -> str:
    if value is None:
        return "NA"
    return f"{scale * float(value):.{digits}f}"


def _print_symmetric(rows: list[dict], target_pcts: Iterable[int]) -> None:
    for pct in target_pcts:
        selected = [row for row in rows if int(row["snapshot_pct"]) == int(pct)]
        print(f"\n{int(pct)}% covariate-adjusted exposed vs unexposed:")
        print(
            f"{'bin':<10} {'nE A/B':>11} {'nU A/B':>11} "
            f"{'dR_E':>7} {'dR_U':>7} {'dT_E':>7} {'dT_U':>7} "
            f"{'dC_E':>7} {'dC_U':>7} {'gapC':>7} {'class':>22}"
        )
        for row in selected:
            print(
                f"{row['bin']:<10} "
                f"{int(row['n_exposed_A'])}/{int(row['n_exposed_B']):<5} "
                f"{int(row['n_unexposed_A'])}/{int(row['n_unexposed_B']):<5} "
                f"{_fmt(row['adjusted_delta_R_exposed'], scale=100):>7} "
                f"{_fmt(row['adjusted_delta_R_unexposed'], scale=100):>7} "
                f"{_fmt(row['adjusted_delta_T_exposed'], scale=100):>7} "
                f"{_fmt(row['adjusted_delta_T_unexposed'], scale=100):>7} "
                f"{_fmt(row['adjusted_delta_C_exposed'], scale=100):>7} "
                f"{_fmt(row['adjusted_delta_C_unexposed'], scale=100):>7} "
                f"{_fmt(row['adjusted_gap_C_unexposed_minus_exposed'], scale=100):>7} "
                f"{row['adjusted_transfer_classification']:>22}"
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Covariate-adjusted own-exposure versus transfer analysis."
    )
    parser.add_argument(
        "--balance-question-csv",
        type=Path,
        default=Path("analyses/canonical_exposure_split_transfer/balance_question_rows.csv"),
    )
    parser.add_argument(
        "--per-question-csv",
        type=Path,
        default=Path("analyses/canonical_snapshot_crossfit/per_question_crossfit.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analyses/canonical_exposure_split_adjusted"),
    )
    args = parser.parse_args(argv)

    result = run_analysis(
        balance_question_csv=args.balance_question_csv,
        per_question_csv=args.per_question_csv,
        output_dir=args.output_dir,
        snapshot_schedule=DEFAULT_SNAPSHOT_SCHEDULE,
        target_pcts=DEFAULT_SPLIT_PCTS,
    )
    _print_symmetric(result["adjusted_symmetric"], result["target_pcts"])
    if result["skipped_cells"]:
        print("\nSkipped direction-specific cells:")
        for row in result["skipped_cells"]:
            print(
                f"  {row['snapshot_pct']}% {row['direction']} {row['bin']}: "
                f"nE={row['n_exposed']} nU={row['n_unexposed']} ({row['reason']})"
            )
    print(f"\noutputs: {result['output_dir']}")
    print("CANONICAL COVARIATE-ADJUSTED EXPOSURE SPLIT: COMPLETE")


if __name__ == "__main__":
    main()
