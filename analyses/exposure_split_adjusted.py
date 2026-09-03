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

import math
from collections import defaultdict
from typing import Iterable

import numpy as np

from analyses.exposure_cutoff_balance import classify_delta_c_gap
from analyses.exposure_split_transfer import DIRECTIONS
from analyses.snapshot_crossfit_trajectory import BIN_ORDER


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
        "used_covariates": list(used),
        "dropped_covariates": list(dropped),
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
                    field = f"adjusted_delta_{metric}_{status}"
                    row[field] = (float(a[field]) + float(b[field])) / 2.0
                row[f"adjusted_gap_{metric}_unexposed_minus_exposed"] = (
                    float(row[f"adjusted_delta_{metric}_unexposed"])
                    - float(row[f"adjusted_delta_{metric}_exposed"])
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
