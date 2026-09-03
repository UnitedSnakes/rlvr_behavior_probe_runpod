#!/usr/bin/env python3
"""Cross-fit fixed-panel snapshot movement against the canonical K32 p0 baseline.

This module keeps the preregistered reward-success difficulty bins fixed and
prevents baseline Monte Carlo noise from appearing on both the binning axis and
the subtraction term:

- A-bin/B-base: p0_A chooses the bin; the opposite B half supplies X_0.
- B-bin/A-base: p0_B chooses the bin; the opposite A half supplies X_0.

The snapshot side X_t is the independent C-bank evaluation shared across
snapshots. X is decomposed into canonical reward (R), termination (T), and
unconditional correctness (C).
"""

from __future__ import annotations

import math
from collections import defaultdict


BIN_ORDER = ["0", "(0,.25]", "(.25,.5]", "(.5,.75]", "(.75,1)", "1"]
DIRECTIONS = ("A-bin/B-base", "B-bin/A-base")
METRICS = ("R", "T", "C")


def assign_reward_bin(p: float) -> str:
    """Assign p to the frozen preregistered reward-success bin."""
    value = float(p)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"reward success probability must lie in [0, 1], got {value}")
    if value == 0.0:
        return "0"
    if value <= 0.25:
        return "(0,.25]"
    if value <= 0.5:
        return "(.25,.5]"
    if value <= 0.75:
        return "(.5,.75]"
    if value < 1.0:
        return "(.75,1)"
    return "1"


def _mean_flag(rows: list[dict], key: str) -> float:
    if not rows:
        raise ValueError("rollout half must be non-empty")
    return sum(float(bool(row[key])) for row in rows) / len(rows)


def rollout_metrics(rows: list[dict]) -> dict[str, float]:
    """Return R/T/C empirical rates for one rollout half."""
    if not rows:
        raise ValueError("rollout half must be non-empty")
    return {
        "R": sum(float(row["canonical_reward"]) for row in rows) / len(rows),
        "T": _mean_flag(rows, "terminated"),
        "C": _mean_flag(rows, "correct"),
    }


def _assert_close(actual: float, expected: float, message: str) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{message}: actual={actual}, expected={expected}")


def validate_p0_record(record: dict) -> dict[str, dict[str, float]]:
    """Validate one K32 record and return per-half R/T/C rates."""
    a = list(record["rollouts_A"])
    b = list(record["rollouts_B"])
    half_size = int(record["half_size"])
    n_rollouts = int(record["n_rollouts"])

    if len(a) != half_size or len(b) != half_size:
        raise ValueError(
            f"dataset_index={record['dataset_index']}: half-size mismatch; "
            f"declared={half_size}, A={len(a)}, B={len(b)}"
        )
    if n_rollouts != len(a) + len(b):
        raise ValueError(
            f"dataset_index={record['dataset_index']}: n_rollouts mismatch; "
            f"declared={n_rollouts}, observed={len(a) + len(b)}"
        )

    metrics = {"A": rollout_metrics(a), "B": rollout_metrics(b)}
    combined = {
        metric: (metrics["A"][metric] + metrics["B"][metric]) / 2.0
        for metric in METRICS
    }

    _assert_close(record["p0_A"], metrics["A"]["R"], "p0_A mismatch")
    _assert_close(record["p0_B"], metrics["B"]["R"], "p0_B mismatch")
    _assert_close(record["p0"], combined["R"], "p0 aggregate mismatch")
    _assert_close(
        record["termination_rate"], combined["T"], "termination aggregate mismatch"
    )
    _assert_close(
        record["correctness_p0"], combined["C"], "correctness aggregate mismatch"
    )
    return metrics


def snapshot_metrics(record: dict) -> dict[str, float]:
    n = int(record["n_rollouts"])
    if n <= 0:
        raise ValueError("snapshot n_rollouts must be positive")
    counts = {
        "R": int(record["n_reward"]),
        "T": int(record["n_terminated"]),
        "C": int(record["n_correct"]),
    }
    for metric, count in counts.items():
        if not 0 <= count <= n:
            raise ValueError(f"snapshot {metric} count out of range: {count}/{n}")
    return {metric: count / n for metric, count in counts.items()}


def _index_unique(records: list[dict], name: str) -> dict[int, dict]:
    indexed: dict[int, dict] = {}
    for record in records:
        index = int(record["dataset_index"])
        if index in indexed:
            raise ValueError(f"duplicate dataset_index={index} in {name}")
        indexed[index] = record
    return indexed


def build_crossfit_question_rows(
    p0_records: list[dict],
    snapshot_records: list[dict],
    *,
    snapshot_pct: int,
) -> list[dict]:
    """Build the two cross-fit directions at question level for one snapshot."""
    p0_by_index = _index_unique(p0_records, "p0 records")
    snapshot_by_index = _index_unique(snapshot_records, "snapshot records")
    if set(p0_by_index) != set(snapshot_by_index):
        missing = sorted(set(p0_by_index) - set(snapshot_by_index))
        extra = sorted(set(snapshot_by_index) - set(p0_by_index))
        raise ValueError(
            f"snapshot/p0 index mismatch; missing_snapshot={missing}, extra_snapshot={extra}"
        )

    rows: list[dict] = []
    for index in sorted(p0_by_index):
        p0 = p0_by_index[index]
        halves = validate_p0_record(p0)
        snap = snapshot_metrics(snapshot_by_index[index])

        for direction, bin_half, baseline_half in (
            ("A-bin/B-base", "A", "B"),
            ("B-bin/A-base", "B", "A"),
        ):
            baseline = halves[baseline_half]
            bin_probability = halves[bin_half]["R"]
            row = {
                "snapshot_pct": int(snapshot_pct),
                "direction": direction,
                "dataset_index": index,
                "bin": assign_reward_bin(bin_probability),
                "bin_p0": bin_probability,
            }
            for metric in METRICS:
                row[f"baseline_{metric}"] = baseline[metric]
                row[f"snapshot_{metric}"] = snap[metric]
                row[f"delta_{metric}"] = snap[metric] - baseline[metric]
            rows.append(row)

    return rows


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty list")
    return sum(values) / len(values)


def aggregate_crossfit_rows(question_rows: list[dict]) -> list[dict]:
    """Aggregate direction-specific means, then average directions equally."""
    grouped: dict[tuple[int, str, str], list[dict]] = defaultdict(list)
    for row in question_rows:
        grouped[(int(row["snapshot_pct"]), row["direction"], row["bin"])].append(row)

    directional: list[dict] = []
    for (pct, direction, bin_label), rows in sorted(
        grouped.items(), key=lambda item: (item[0][0], DIRECTIONS.index(item[0][1]), BIN_ORDER.index(item[0][2]))
    ):
        out = {
            "snapshot_pct": pct,
            "direction": direction,
            "bin": bin_label,
            "n_questions": len(rows),
        }
        for metric in METRICS:
            out[f"delta_{metric}"] = _mean([float(row[f"delta_{metric}"]) for row in rows])
        directional.append(out)

    by_key = {
        (row["snapshot_pct"], row["direction"], row["bin"]): row
        for row in directional
    }
    pcts = sorted({int(row["snapshot_pct"]) for row in question_rows})
    symmetric: list[dict] = []
    for pct in pcts:
        for bin_label in BIN_ORDER:
            a = by_key.get((pct, "A-bin/B-base", bin_label))
            b = by_key.get((pct, "B-bin/A-base", bin_label))
            if a is None or b is None:
                continue
            row = {
                "snapshot_pct": pct,
                "direction": "symmetric",
                "bin": bin_label,
                "n_questions": None,
                "n_questions_A": int(a["n_questions"]),
                "n_questions_B": int(b["n_questions"]),
            }
            for metric in METRICS:
                row[f"delta_{metric}"] = (
                    float(a[f"delta_{metric}"]) + float(b[f"delta_{metric"])
                ) / 2.0
            symmetric.append(row)

    return directional + symmetric
