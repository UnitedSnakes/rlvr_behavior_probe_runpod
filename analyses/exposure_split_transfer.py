#!/usr/bin/env python3
"""Exposure-timing balance diagnostics and exposed-vs-unexposed transfer analysis.

This module deliberately separates two stages:

1. pre-outcome balance diagnostics for own-exposure timing within frozen p0 bins;
2. exposed-vs-unexposed movement contrasts at matched snapshots.

The primary transfer-ratio criterion is frozen before looking at split outcomes:

    ratio = DeltaC_unexposed / DeltaC_exposed

- ratio <= 0.25: own-exposure-dominant
- 0.25 < ratio < 0.75: mixed
- ratio >= 0.75: transfer-dominant

The ratio is not classified when DeltaC_exposed <= 0 because division would be
unstable or sign interpretation would be misleading.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

from analyses.snapshot_crossfit_trajectory import (
    BIN_ORDER,
    assign_reward_bin,
    validate_p0_record,
)


DIRECTIONS = ("A-bin/B-base", "B-bin/A-base")
OWN_EXPOSURE_RATIO_MAX = 0.25
TRANSFER_RATIO_MIN = 0.75


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty list")
    return sum(values) / len(values)


def _pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y):
        raise ValueError("correlation inputs must have equal length")
    if len(x) < 2:
        return None
    mx = _mean(x)
    my = _mean(y)
    dx = [value - mx for value in x]
    dy = [value - my for value in y]
    sx2 = sum(value * value for value in dx)
    sy2 = sum(value * value for value in dy)
    if sx2 <= 0.0 or sy2 <= 0.0:
        return None
    return sum(a * b for a, b in zip(dx, dy, strict=True)) / math.sqrt(sx2 * sy2)


def _half_completion_length(record: dict, half: str) -> float:
    rows = list(record[f"rollouts_{half}"])
    if not rows:
        raise ValueError("p0 rollout half must be non-empty")
    lengths: list[float] = []
    for row in rows:
        if "n_tokens" not in row:
            raise ValueError(
                f"dataset_index={record['dataset_index']}: p0 rollout missing n_tokens"
            )
        value = float(row["n_tokens"])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"dataset_index={record['dataset_index']}: invalid p0 n_tokens={value}"
            )
        lengths.append(value)
    return _mean(lengths)


def build_balance_rows(
    p0_records: Iterable[dict],
    *,
    exposure_steps: dict[int, int],
    prompt_token_counts: dict[int, int],
) -> list[dict]:
    """Build direction-specific pre-exposure balance rows.

    The same p0 half used for binning is never reused as the baseline covariate:
    A-bin uses B-half p0 reward/completion length and vice versa.
    """
    records = list(p0_records)
    indexed: dict[int, dict] = {}
    for record in records:
        index = int(record["dataset_index"])
        if index in indexed:
            raise ValueError(f"duplicate p0 dataset_index={index}")
        indexed[index] = record
        validate_p0_record(record)

    expected = set(indexed)
    if set(exposure_steps) != expected:
        raise ValueError("exposure-step indices must match p0 indices exactly")
    if set(prompt_token_counts) != expected:
        raise ValueError("prompt-token indices must match p0 indices exactly")

    rows: list[dict] = []
    for index in sorted(indexed):
        record = indexed[index]
        halves = validate_p0_record(record)
        step = int(exposure_steps[index])
        prompt_tokens = int(prompt_token_counts[index])
        if step < 0:
            raise ValueError(f"negative exposure step for dataset_index={index}")
        if prompt_tokens <= 0:
            raise ValueError(f"non-positive prompt token count for dataset_index={index}")

        for direction, bin_half, baseline_half in (
            ("A-bin/B-base", "A", "B"),
            ("B-bin/A-base", "B", "A"),
        ):
            rows.append(
                {
                    "direction": direction,
                    "bin": assign_reward_bin(halves[bin_half]["R"]),
                    "dataset_index": index,
                    "exposure_step": step,
                    "baseline_p0": float(halves[baseline_half]["R"]),
                    "baseline_p0_completion_length": _half_completion_length(
                        record, baseline_half
                    ),
                    "prompt_token_count": prompt_tokens,
                }
            )
    return rows


def _uniform_ks_midpoint(exposure_steps: list[int], total_steps: int) -> float:
    """One-sample KS distance to Uniform(0,1) using step midpoints."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not exposure_steps:
        raise ValueError("cannot compute uniformity on an empty bin")
    positions = sorted((int(step) + 0.5) / total_steps for step in exposure_steps)
    if positions[0] < 0.0 or positions[-1] > 1.0:
        raise ValueError("exposure step outside declared training range")
    n = len(positions)
    d_plus = max((i + 1) / n - value for i, value in enumerate(positions))
    d_minus = max(value - i / n for i, value in enumerate(positions))
    return max(d_plus, d_minus)


def summarize_balance(rows: Iterable[dict], *, total_steps: int) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        direction = row["direction"]
        if direction not in DIRECTIONS:
            raise ValueError(f"unexpected direction {direction}")
        grouped[(direction, row["bin"])].append(row)

    result: list[dict] = []
    for (direction, bin_label), group in sorted(
        grouped.items(),
        key=lambda item: (DIRECTIONS.index(item[0][0]), BIN_ORDER.index(item[0][1])),
    ):
        exposure = [float(row["exposure_step"]) for row in group]
        baseline_p0 = [float(row["baseline_p0"]) for row in group]
        completion = [float(row["baseline_p0_completion_length"]) for row in group]
        prompt_tokens = [float(row["prompt_token_count"]) for row in group]
        q_counts = [0, 0, 0, 0]
        for row in group:
            u = (int(row["exposure_step"]) + 0.5) / total_steps
            if not 0.0 <= u <= 1.0:
                raise ValueError("exposure step outside declared training range")
            quartile = min(int(u * 4), 3)
            q_counts[quartile] += 1
        n = len(group)
        result.append(
            {
                "direction": direction,
                "bin": bin_label,
                "n_questions": n,
                "corr_exposure_baseline_p0": _pearson(exposure, baseline_p0),
                "corr_exposure_p0_completion_length": _pearson(exposure, completion),
                "corr_exposure_prompt_tokens": _pearson(exposure, prompt_tokens),
                "uniform_ks_D": _uniform_ks_midpoint(
                    [int(row["exposure_step"]) for row in group], total_steps
                ),
                "exposure_q1_fraction": q_counts[0] / n,
                "exposure_q2_fraction": q_counts[1] / n,
                "exposure_q3_fraction": q_counts[2] / n,
                "exposure_q4_fraction": q_counts[3] / n,
            }
        )
    return result


def classify_transfer_ratio(
    *,
    delta_c_exposed: float,
    delta_c_unexposed: float,
) -> dict:
    exposed = float(delta_c_exposed)
    unexposed = float(delta_c_unexposed)
    if not math.isfinite(exposed) or not math.isfinite(unexposed):
        raise ValueError("DeltaC values must be finite")
    if exposed <= 0.0:
        return {"ratio": None, "classification": "not_classifiable"}

    ratio = unexposed / exposed
    if ratio <= OWN_EXPOSURE_RATIO_MAX:
        classification = "own_exposure_dominant"
    elif ratio >= TRANSFER_RATIO_MIN:
        classification = "transfer_dominant"
    else:
        classification = "mixed"
    return {"ratio": ratio, "classification": classification}
