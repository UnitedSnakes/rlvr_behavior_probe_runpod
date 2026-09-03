#!/usr/bin/env python3
"""Cross-fit fixed-panel snapshot movement against the canonical K32 p0 baseline.

This analysis keeps the frozen reward-success difficulty bins and prevents the
same p0 Monte Carlo noise from appearing on both the binning axis and baseline
subtraction:

- A-bin/B-base: p0_A chooses the bin; the independent B half supplies X_0.
- B-bin/A-base: p0_B chooses the bin; the independent A half supplies X_0.

The snapshot side X_t is the independent C-bank evaluation shared across all
20 policy snapshots. X is decomposed into canonical reward (R), termination
(T), and unconditional correctness (C).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable


BIN_ORDER = ["0", "(0,.25]", "(.25,.5]", "(.5,.75]", "(.75,1)", "1"]
DIRECTIONS = ("A-bin/B-base", "B-bin/A-base")
METRICS = ("R", "T", "C")
DEFAULT_SNAPSHOT_PCTS = list(range(5, 101, 5))
DEFAULT_EXPECTED_INDICES = list(range(256))
DEFAULT_C_BANK_SEED_START = 4_275_000

# Transport/integrity references only. These are not analysis targets or
# statistical thresholds. Snapshot values were printed from the RunPod audit
# to four decimals, so they are checked within half a final printed unit.
KNOWN_P0_AGGREGATE = {
    "R": 2860 / 8192,
    "T": 3737 / 8192,
    "C": 4133 / 8192,
}
KNOWN_SNAPSHOT_AGGREGATES = {
    5: {"R": 0.3655, "T": 0.4788, "C": 0.5059},
    10: {"R": 0.3813, "T": 0.5190, "C": 0.5183},
    15: {"R": 0.4207, "T": 0.5786, "C": 0.5239},
    20: {"R": 0.4595, "T": 0.6418, "C": 0.5376},
    25: {"R": 0.4900, "T": 0.6833, "C": 0.5562},
    30: {"R": 0.4905, "T": 0.7126, "C": 0.5479},
    35: {"R": 0.4978, "T": 0.7200, "C": 0.5540},
    40: {"R": 0.5142, "T": 0.7439, "C": 0.5630},
    45: {"R": 0.5195, "T": 0.7542, "C": 0.5713},
    50: {"R": 0.5244, "T": 0.7502, "C": 0.5713},
    55: {"R": 0.5164, "T": 0.7588, "C": 0.5652},
    60: {"R": 0.5142, "T": 0.7534, "C": 0.5601},
    65: {"R": 0.5413, "T": 0.7688, "C": 0.5850},
    70: {"R": 0.5276, "T": 0.7588, "C": 0.5696},
    75: {"R": 0.5259, "T": 0.7686, "C": 0.5806},
    80: {"R": 0.5154, "T": 0.7629, "C": 0.5603},
    85: {"R": 0.5166, "T": 0.7505, "C": 0.5664},
    90: {"R": 0.5288, "T": 0.7600, "C": 0.5750},
    95: {"R": 0.5376, "T": 0.7686, "C": 0.5764},
    100: {"R": 0.5303, "T": 0.7725, "C": 0.5774},
}
KNOWN_SNAPSHOT_TOLERANCE = 5.1e-5


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


def _assert_close(actual: float, expected: float, message: str, *, atol: float = 1e-12) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=atol):
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
        grouped.items(),
        key=lambda item: (
            item[0][0],
            DIRECTIONS.index(item[0][1]),
            BIN_ORDER.index(item[0][2]),
        ),
    ):
        out = {
            "snapshot_pct": pct,
            "direction": direction,
            "bin": bin_label,
            "n_questions": len(rows),
            "n_questions_A": None,
            "n_questions_B": None,
        }
        for metric in METRICS:
            out[f"delta_{metric}"] = _mean(
                [float(row[f"delta_{metric}"]) for row in rows]
            )
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
                    float(a[f"delta_{metric}"]) + float(b[f"delta_{metric}"])
                ) / 2.0
            symmetric.append(row)

    return directional + symmetric


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        # Iterate the file directly rather than str.splitlines(): completions may
        # contain Unicode line-separator characters that are legal inside JSON.
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL record {path}:{lineno}: {exc}") from exc
    return rows


def load_p0_records(p0_dir: Path, expected_indices: Iterable[int]) -> list[dict]:
    root = Path(p0_dir)
    shard0 = read_jsonl(root / "rollouts_shard0of2.jsonl")
    shard1 = read_jsonl(root / "rollouts_shard1of2.jsonl")

    expected = list(int(index) for index in expected_indices)
    expected_set = set(expected)
    even_expected = {index for index in expected if index % 2 == 0}
    odd_expected = {index for index in expected if index % 2 == 1}
    shard0_indices = {int(row["dataset_index"]) for row in shard0}
    shard1_indices = {int(row["dataset_index"]) for row in shard1}

    if shard0_indices != even_expected:
        raise ValueError(
            f"p0 shard0 parity/index mismatch; expected={sorted(even_expected)}, "
            f"observed={sorted(shard0_indices)}"
        )
    if shard1_indices != odd_expected:
        raise ValueError(
            f"p0 shard1 parity/index mismatch; expected={sorted(odd_expected)}, "
            f"observed={sorted(shard1_indices)}"
        )

    merged = sorted(shard0 + shard1, key=lambda row: int(row["dataset_index"]))
    indexed = _index_unique(merged, "merged p0 records")
    if set(indexed) != expected_set:
        raise ValueError("merged p0 records do not match expected indices")
    for record in merged:
        validate_p0_record(record)
    return merged


def load_snapshot_records(
    snapshot_dir: Path,
    pct: int,
    expected_indices: Iterable[int],
    *,
    expected_k: int | None = None,
) -> list[dict]:
    path = Path(snapshot_dir) / f"pi_{int(pct):03d}" / "snapshot_raw.jsonl"
    rows = read_jsonl(path)
    indexed = _index_unique(rows, f"snapshot pi_{int(pct):03d}")
    expected = list(int(index) for index in expected_indices)
    if set(indexed) != set(expected):
        missing = sorted(set(expected) - set(indexed))
        extra = sorted(set(indexed) - set(expected))
        raise ValueError(
            f"pi_{int(pct):03d} index mismatch; missing={missing}, extra={extra}"
        )
    ordered = [indexed[index] for index in expected]
    for index, record in zip(expected, ordered):
        snapshot_metrics(record)
        if expected_k is not None and int(record["n_rollouts"]) != int(expected_k):
            raise ValueError(
                f"pi_{int(pct):03d} dataset_index={index}: expected K={expected_k}, "
                f"got {record['n_rollouts']}"
            )
        expected_seed = DEFAULT_C_BANK_SEED_START + index
        if int(record["question_seed"]) != expected_seed:
            raise ValueError(
                f"pi_{int(pct):03d} dataset_index={index}: C-bank seed mismatch; "
                f"expected={expected_seed}, got={record['question_seed']}"
            )
    return ordered


def aggregate_p0(records: list[dict]) -> dict[str, float]:
    for record in records:
        validate_p0_record(record)
    return {
        "R": _mean([float(record["p0"]) for record in records]),
        "T": _mean([float(record["termination_rate"]) for record in records]),
        "C": _mean([float(record["correctness_p0"]) for record in records]),
    }


def aggregate_snapshot(records: list[dict]) -> dict[str, float]:
    totals = {metric: 0 for metric in METRICS}
    n_total = 0
    count_keys = {"R": "n_reward", "T": "n_terminated", "C": "n_correct"}
    for record in records:
        snapshot_metrics(record)
        n = int(record["n_rollouts"])
        n_total += n
        for metric, key in count_keys.items():
            totals[metric] += int(record[key])
    if n_total <= 0:
        raise ValueError("snapshot aggregate has no rollouts")
    return {metric: totals[metric] / n_total for metric in METRICS}


def verify_transport_aggregates(
    p0_aggregate: dict[str, float],
    snapshot_aggregates: dict[int, dict[str, float]],
) -> None:
    for metric in METRICS:
        _assert_close(
            p0_aggregate[metric],
            KNOWN_P0_AGGREGATE[metric],
            f"known p0 {metric} aggregate mismatch",
        )
    for pct, aggregate in snapshot_aggregates.items():
        if pct not in KNOWN_SNAPSHOT_AGGREGATES:
            raise ValueError(f"no known transport aggregate registered for snapshot {pct}%")
        for metric in METRICS:
            _assert_close(
                aggregate[metric],
                KNOWN_SNAPSHOT_AGGREGATES[pct][metric],
                f"known snapshot {pct}% {metric} aggregate mismatch",
                atol=KNOWN_SNAPSHOT_TOLERANCE,
            )


def write_csv(path: Path, rows: list[dict]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {destination}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_metric_plot(trajectory_rows: list[dict], metric: str, out_path: Path) -> None:
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    symmetric = [row for row in trajectory_rows if row["direction"] == "symmetric"]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for bin_label in BIN_ORDER:
        rows = sorted(
            [row for row in symmetric if row["bin"] == bin_label],
            key=lambda row: int(row["snapshot_pct"]),
        )
        if not rows:
            continue
        ax.plot(
            [int(row["snapshot_pct"]) for row in rows],
            [100.0 * float(row[f"delta_{metric}"]) for row in rows],
            marker="o",
            label=bin_label,
        )
    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("Training progress (%)")
    ax.set_ylabel(f"Cross-fit ΔP({metric}) (percentage points)")
    ax.set_title(f"Canonical fixed-panel ΔP({metric}) by frozen p0 reward bin")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="p0 reward bin")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_analysis(
    *,
    p0_dir: Path,
    snapshot_dir: Path,
    output_dir: Path,
    snapshot_pcts: Iterable[int] = DEFAULT_SNAPSHOT_PCTS,
    expected_indices: Iterable[int] = DEFAULT_EXPECTED_INDICES,
    expected_snapshot_k: int | None = None,
    verify_known_aggregates: bool = True,
) -> dict:
    pcts = [int(pct) for pct in snapshot_pcts]
    indices = [int(index) for index in expected_indices]
    if not pcts:
        raise ValueError("snapshot_pcts must be non-empty")
    if not indices:
        raise ValueError("expected_indices must be non-empty")

    p0_records = load_p0_records(Path(p0_dir), indices)
    p0_aggregate = aggregate_p0(p0_records)

    all_question_rows: list[dict] = []
    snapshot_aggregates: dict[int, dict[str, float]] = {}
    for pct in pcts:
        snapshot_records = load_snapshot_records(
            Path(snapshot_dir),
            pct,
            indices,
            expected_k=expected_snapshot_k,
        )
        snapshot_aggregates[pct] = aggregate_snapshot(snapshot_records)
        all_question_rows.extend(
            build_crossfit_question_rows(
                p0_records,
                snapshot_records,
                snapshot_pct=pct,
            )
        )

    if verify_known_aggregates:
        verify_transport_aggregates(p0_aggregate, snapshot_aggregates)

    trajectory_rows = aggregate_crossfit_rows(all_question_rows)
    sanity_rows = [
        {
            "snapshot_pct": 0,
            "source": "p0_K32_AplusB",
            "R": p0_aggregate["R"],
            "T": p0_aggregate["T"],
            "C": p0_aggregate["C"],
        }
    ]
    for pct in pcts:
        sanity_rows.append(
            {
                "snapshot_pct": pct,
                "source": "snapshot_C_bank",
                "R": snapshot_aggregates[pct]["R"],
                "T": snapshot_aggregates[pct]["T"],
                "C": snapshot_aggregates[pct]["C"],
            }
        )

    destination = Path(output_dir)
    write_csv(destination / "per_question_crossfit.csv", all_question_rows)
    write_csv(destination / "crossfit_trajectory.csv", trajectory_rows)
    write_csv(destination / "aggregate_sanity.csv", sanity_rows)
    for metric in METRICS:
        make_metric_plot(
            trajectory_rows,
            metric,
            destination / f"delta_{metric}_by_p0_bin.png",
        )

    return {
        "p0_records": len(p0_records),
        "snapshots": len(pcts),
        "question_crossfit_rows": len(all_question_rows),
        "trajectory_rows": len(trajectory_rows),
        "output_dir": str(destination),
        "p0_aggregate": p0_aggregate,
        "snapshot_aggregates": snapshot_aggregates,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Cross-fit canonical train256 K16 C-bank snapshot trajectory."
    )
    parser.add_argument(
        "--p0-dir",
        type=Path,
        default=Path("p0_train_k32_top_p1_canonical"),
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=Path("snapshot_eval_train256_k16_cbank"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analyses/canonical_snapshot_crossfit"),
    )
    parser.add_argument(
        "--skip-known-aggregate-check",
        action="store_true",
        help="Skip transport-value verification; not recommended for canonical data.",
    )
    args = parser.parse_args(argv)

    result = run_analysis(
        p0_dir=args.p0_dir,
        snapshot_dir=args.snapshot_dir,
        output_dir=args.output_dir,
        snapshot_pcts=DEFAULT_SNAPSHOT_PCTS,
        expected_indices=DEFAULT_EXPECTED_INDICES,
        expected_snapshot_k=16,
        verify_known_aggregates=not args.skip_known_aggregate_check,
    )

    print("P0 aggregate:")
    p0 = result["p0_aggregate"]
    print(f"  R={p0['R']:.6f}  T={p0['T']:.6f}  C={p0['C']:.6f}")
    print("Snapshot aggregates:")
    for pct in DEFAULT_SNAPSHOT_PCTS:
        row = result["snapshot_aggregates"][pct]
        print(
            f"  {pct:3d}%  R={row['R']:.4f}  T={row['T']:.4f}  C={row['C']:.4f}"
        )
    print(f"\noutputs: {result['output_dir']}")
    print("CANONICAL SNAPSHOT CROSS-FIT ANALYSIS: PASS")


if __name__ == "__main__":
    main()
