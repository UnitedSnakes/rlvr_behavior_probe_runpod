#!/usr/bin/env python3
"""Re-score canonical response banks without the generic last-number fallback.

This is a measurement robustness check, not a new training experiment. It
keeps training rewards, termination labels, frozen p0 bins, exposure timing,
and adjustment covariates fixed, and replaces only correctness C with a strict
extractor that accepts the canonical boxed/final-answer patterns but refuses
the generic "last numeric token anywhere" fallback.

Primary outputs:
  - extractor_disagreement.csv
  - whole_panel_correctness.csv
  - primary_low_bin_current_vs_strict.csv
  - strict_adjusted_symmetric.csv
  - disagreement_examples.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from analyses.exposure_split_adjusted import (
    fit_adjusted_cell,
    symmetrize_adjusted_directional,
)
from analyses.snapshot_crossfit_trajectory import (
    BIN_ORDER,
    DIRECTIONS,
    DEFAULT_EXPECTED_INDICES,
    load_p0_records,
    load_snapshot_records,
    write_csv,
)
from probe.scoring import (
    _to_number,
    extract_numeric_answer,
    extract_numeric_answer_strict,
    numeric_equal,
)

DEFAULT_PCTS = (25, 45, 65)
LOW_NONZERO_BIN = "(0,.25]"


def _gold_value(record: dict) -> float:
    raw = record.get("gold", record.get("answer"))
    if raw is None:
        raise ValueError(
            f"record dataset_index={record.get('dataset_index')} has no gold/answer"
        )
    value = _to_number(str(raw))
    if value is None:
        raise ValueError(
            f"could not parse gold for dataset_index={record.get('dataset_index')}: {raw!r}"
        )
    return float(value)


def score_rollouts(
    rollouts: Iterable[dict],
    *,
    gold: float,
    source: str,
    dataset_index: int,
) -> tuple[dict, list[dict]]:
    """Compare canonical and strict correctness for one rollout collection."""
    rows = list(rollouts)
    if not rows:
        raise ValueError(f"{source} dataset_index={dataset_index} has no rollouts")

    current_correct = 0
    strict_correct = 0
    disagreements = 0
    current_only = 0
    strict_only = 0
    last_number = 0
    last_number_correct = 0
    scorer_drift = 0
    examples: list[dict] = []

    for rollout_position, rollout in enumerate(rows):
        text = str(rollout.get("text", ""))
        current_pred, current_token, current_method = extract_numeric_answer(text)
        strict_pred, strict_token, strict_method = extract_numeric_answer_strict(text)
        current_label = bool(numeric_equal(current_pred, gold))
        strict_label = bool(numeric_equal(strict_pred, gold))

        stored = rollout.get("correct")
        if stored is not None and bool(stored) != current_label:
            scorer_drift += 1

        current_correct += int(current_label)
        strict_correct += int(strict_label)
        last_number += int(current_method == "last_number")
        last_number_correct += int(current_method == "last_number" and current_label)

        if current_label != strict_label:
            disagreements += 1
            current_only += int(current_label and not strict_label)
            strict_only += int(strict_label and not current_label)
            examples.append(
                {
                    "source": source,
                    "dataset_index": int(dataset_index),
                    "rollout_position": int(
                        rollout.get("rollout", rollout_position)
                    ),
                    "gold": gold,
                    "current_pred": current_pred,
                    "strict_pred": strict_pred,
                    "current_method": current_method,
                    "strict_method": strict_method,
                    "current_token": current_token,
                    "strict_token": strict_token,
                    "current_correct": current_label,
                    "strict_correct": strict_label,
                    "terminated": rollout.get("terminated"),
                    "completion_length": rollout.get("completion_length"),
                    "text": text,
                }
            )

    n = len(rows)
    return (
        {
            "n_rollouts": n,
            "current_n_correct": current_correct,
            "strict_n_correct": strict_correct,
            "n_disagree": disagreements,
            "n_current_correct_strict_incorrect": current_only,
            "n_current_incorrect_strict_correct": strict_only,
            "n_last_number": last_number,
            "n_last_number_correct": last_number_correct,
            "scorer_drift_count": scorer_drift,
        },
        examples,
    )


def _combine_summaries(source: str, parts: Iterable[dict]) -> dict:
    pieces = list(parts)
    if not pieces:
        raise ValueError(f"cannot combine empty summary for {source}")

    fields = (
        "n_rollouts",
        "current_n_correct",
        "strict_n_correct",
        "n_disagree",
        "n_current_correct_strict_incorrect",
        "n_current_incorrect_strict_correct",
        "n_last_number",
        "n_last_number_correct",
        "scorer_drift_count",
    )
    total = {name: sum(int(part[name]) for part in pieces) for name in fields}
    n = total["n_rollouts"]
    if n <= 0:
        raise ValueError(f"{source} has no rollouts")

    return {
        "source": source,
        **total,
        "current_C": total["current_n_correct"] / n,
        "strict_C": total["strict_n_correct"] / n,
        "strict_minus_current_C": (
            total["strict_n_correct"] - total["current_n_correct"]
        ) / n,
        "disagreement_rate": total["n_disagree"] / n,
        "last_number_rate": total["n_last_number"] / n,
        "last_number_correct_rate_all": total["n_last_number_correct"] / n,
    }


def score_p0_records(
    records: list[dict],
) -> tuple[dict, dict[int, dict[str, float]], list[dict]]:
    """Return aggregate p0 summary and strict A/B C rates per question."""
    summaries: list[dict] = []
    strict_halves: dict[int, dict[str, float]] = {}
    examples: list[dict] = []

    for record in records:
        index = int(record["dataset_index"])
        gold = _gold_value(record)
        strict_halves[index] = {}
        for half in ("A", "B"):
            summary, ex = score_rollouts(
                record[f"rollouts_{half}"],
                gold=gold,
                source=f"p0_{half}",
                dataset_index=index,
            )
            summaries.append(summary)
            examples.extend(ex)
            strict_halves[index][half] = (
                int(summary["strict_n_correct"]) / int(summary["n_rollouts"])
            )

    return _combine_summaries("p0_AplusB", summaries), strict_halves, examples


def score_snapshot_records(
    records: list[dict],
    *,
    source: str,
) -> tuple[dict, dict[int, float], list[dict]]:
    summaries: list[dict] = []
    strict_by_index: dict[int, float] = {}
    examples: list[dict] = []

    for record in records:
        index = int(record["dataset_index"])
        summary, ex = score_rollouts(
            record["rollouts"],
            gold=_gold_value(record),
            source=source,
            dataset_index=index,
        )
        summaries.append(summary)
        examples.extend(ex)
        strict_by_index[index] = (
            int(summary["strict_n_correct"]) / int(summary["n_rollouts"])
        )

    return _combine_summaries(source, summaries), strict_by_index, examples


def load_csv_rows(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def replace_delta_c_with_strict(
    adjustment_rows: Iterable[dict],
    *,
    p0_strict_halves: dict[int, dict[str, float]],
    snapshot_strict: dict[int, dict[int, float]],
) -> list[dict]:
    """Keep the frozen analysis design and replace only correctness C."""
    out: list[dict] = []

    for raw in adjustment_rows:
        row = dict(raw)
        pct = int(row["snapshot_pct"])
        index = int(row["dataset_index"])
        direction = row["direction"]

        if direction == "A-bin/B-base":
            baseline_half = "B"
        elif direction == "B-bin/A-base":
            baseline_half = "A"
        else:
            raise ValueError(f"unexpected direction {direction!r}")

        baseline = float(p0_strict_halves[index][baseline_half])
        snapshot = float(snapshot_strict[pct][index])
        row["delta_C_current"] = float(row["delta_C"])
        row["baseline_C_strict"] = baseline
        row["snapshot_C_strict"] = snapshot
        row["delta_C"] = snapshot - baseline
        out.append(row)

    return out


def fit_adjusted_rows(rows: Iterable[dict]) -> tuple[list[dict], list[dict]]:
    grouped: dict[tuple[int, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[
            (int(row["snapshot_pct"]), row["direction"], row["bin"])
        ].append(row)

    directional: list[dict] = []
    for key, cell in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            DIRECTIONS.index(item[0][1]),
            BIN_ORDER.index(item[0][2]),
        ),
    ):
        del key
        statuses = {row["exposure_status"] for row in cell}
        if statuses != {"exposed", "unexposed"}:
            continue
        n_e = sum(row["exposure_status"] == "exposed" for row in cell)
        n_u = sum(row["exposure_status"] == "unexposed" for row in cell)
        if min(n_e, n_u) < 2:
            continue
        directional.append(fit_adjusted_cell(cell))

    return directional, symmetrize_adjusted_directional(directional)


def _filter_adjusted_c(rows: Iterable[dict]) -> list[dict]:
    fields = (
        "snapshot_pct",
        "snapshot_step",
        "direction",
        "bin",
        "n_questions",
        "n_exposed",
        "n_unexposed",
        "n_questions_A",
        "n_questions_B",
        "n_exposed_A",
        "n_exposed_B",
        "n_unexposed_A",
        "n_unexposed_B",
        "adjusted_delta_C_exposed",
        "adjusted_delta_C_unexposed",
        "adjusted_gap_C_unexposed_minus_exposed",
    )
    return [{key: row[key] for key in fields if key in row} for row in rows]


def primary_low_bin_comparison(
    current_symmetric: Iterable[dict],
    strict_symmetric: Iterable[dict],
) -> list[dict]:
    current = {
        (int(row["snapshot_pct"]), row["bin"]): row
        for row in current_symmetric
    }
    strict = {
        (int(row["snapshot_pct"]), row["bin"]): row
        for row in strict_symmetric
    }

    out: list[dict] = []
    for pct in DEFAULT_PCTS:
        key = (pct, LOW_NONZERO_BIN)
        if key not in current or key not in strict:
            raise ValueError(f"missing primary low-bin adjusted cell at {pct}%")
        c = current[key]
        s = strict[key]
        out.append(
            {
                "snapshot_pct": pct,
                "current_delta_C_unexposed": float(
                    c["adjusted_delta_C_unexposed"]
                ),
                "strict_delta_C_unexposed": float(
                    s["adjusted_delta_C_unexposed"]
                ),
                "current_delta_C_exposed": float(
                    c["adjusted_delta_C_exposed"]
                ),
                "strict_delta_C_exposed": float(
                    s["adjusted_delta_C_exposed"]
                ),
                "current_gap_U_minus_E": float(
                    c["adjusted_gap_C_unexposed_minus_exposed"]
                ),
                "strict_gap_U_minus_E": float(
                    s["adjusted_gap_C_unexposed_minus_exposed"]
                ),
            }
        )
    return out


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_analysis(
    *,
    p0_dir: Path,
    grpo_snapshot_dir: Path,
    adjustment_input_csv: Path,
    output_dir: Path,
    maxrl_snapshot_dir: Path | None = None,
    pcts: Iterable[int] = DEFAULT_PCTS,
    expected_indices: Iterable[int] = DEFAULT_EXPECTED_INDICES,
) -> dict:
    pcts = tuple(int(pct) for pct in pcts)
    indices = tuple(int(index) for index in expected_indices)
    required_snapshot_pcts = sorted(set(pcts) | {100})

    p0_records = load_p0_records(Path(p0_dir), indices)
    p0_summary, p0_strict_halves, disagreement_examples = score_p0_records(
        p0_records
    )

    disagreement_rows = [p0_summary]
    grpo_strict: dict[int, dict[int, float]] = {}
    for pct in required_snapshot_pcts:
        records = load_snapshot_records(
            Path(grpo_snapshot_dir),
            pct,
            indices,
            expected_k=16,
        )
        summary, strict_map, examples = score_snapshot_records(
            records,
            source=f"grpo_{pct:03d}",
        )
        disagreement_rows.append(summary)
        grpo_strict[pct] = strict_map
        disagreement_examples.extend(examples)

    maxrl_endpoint_summary = None
    if maxrl_snapshot_dir is not None:
        records = load_snapshot_records(
            Path(maxrl_snapshot_dir),
            100,
            indices,
            expected_k=16,
        )
        maxrl_endpoint_summary, _, examples = score_snapshot_records(
            records,
            source="maxrl_100",
        )
        disagreement_rows.append(maxrl_endpoint_summary)
        disagreement_examples.extend(examples)

    original_adjustment = [
        row
        for row in load_csv_rows(Path(adjustment_input_csv))
        if int(row["snapshot_pct"]) in set(pcts)
    ]
    _, current_symmetric = fit_adjusted_rows(original_adjustment)

    strict_adjustment = replace_delta_c_with_strict(
        original_adjustment,
        p0_strict_halves=p0_strict_halves,
        snapshot_strict=grpo_strict,
    )
    strict_directional, strict_symmetric = fit_adjusted_rows(strict_adjustment)
    primary = primary_low_bin_comparison(current_symmetric, strict_symmetric)

    grpo_endpoint = next(
        row for row in disagreement_rows if row["source"] == "grpo_100"
    )
    whole_panel = [
        {
            "policy": "initial",
            "current_C": p0_summary["current_C"],
            "strict_C": p0_summary["strict_C"],
            "current_delta_C": 0.0,
            "strict_delta_C": 0.0,
        },
        {
            "policy": "GRPO_endpoint",
            "current_C": grpo_endpoint["current_C"],
            "strict_C": grpo_endpoint["strict_C"],
            "current_delta_C": (
                grpo_endpoint["current_C"] - p0_summary["current_C"]
            ),
            "strict_delta_C": (
                grpo_endpoint["strict_C"] - p0_summary["strict_C"]
            ),
        },
    ]
    if maxrl_endpoint_summary is not None:
        whole_panel.append(
            {
                "policy": "MaxRL_endpoint",
                "current_C": maxrl_endpoint_summary["current_C"],
                "strict_C": maxrl_endpoint_summary["strict_C"],
                "current_delta_C": (
                    maxrl_endpoint_summary["current_C"] - p0_summary["current_C"]
                ),
                "strict_delta_C": (
                    maxrl_endpoint_summary["strict_C"] - p0_summary["strict_C"]
                ),
            }
        )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    write_csv(destination / "extractor_disagreement.csv", disagreement_rows)
    write_csv(destination / "whole_panel_correctness.csv", whole_panel)
    write_csv(
        destination / "strict_adjustment_input_rows.csv",
        strict_adjustment,
    )
    write_csv(
        destination / "strict_adjusted_directional.csv",
        _filter_adjusted_c(strict_directional),
    )
    write_csv(
        destination / "strict_adjusted_symmetric.csv",
        _filter_adjusted_c(strict_symmetric),
    )
    write_csv(
        destination / "primary_low_bin_current_vs_strict.csv",
        primary,
    )
    write_jsonl(
        destination / "disagreement_examples.jsonl",
        disagreement_examples,
    )

    summary = {
        "strict_definition": (
            "canonical boxed/final-phrase extraction with generic "
            "last-number fallback disabled"
        ),
        "p0": p0_summary,
        "grpo_endpoint": grpo_endpoint,
        "maxrl_endpoint": maxrl_endpoint_summary,
        "primary_low_bin": primary,
        "n_disagreement_examples": len(disagreement_examples),
        "output_dir": str(destination),
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _pp(value: float) -> str:
    return f"{100.0 * float(value):+.2f} pp"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Re-score canonical response banks without the generic "
            "last-number fallback."
        )
    )
    parser.add_argument(
        "--p0-dir",
        type=Path,
        default=Path("p0_train_k32_top_p1_canonical"),
    )
    parser.add_argument(
        "--grpo-snapshot-dir",
        type=Path,
        default=Path("snapshot_eval_train256_k16_cbank"),
    )
    parser.add_argument(
        "--maxrl-snapshot-dir",
        type=Path,
        default=None,
        help=(
            "Optional MaxRL train256 K16 C-bank root; only endpoint 100%% "
            "is required."
        ),
    )
    parser.add_argument(
        "--adjustment-input-csv",
        type=Path,
        default=Path(
            "analyses/canonical_exposure_split_adjusted/"
            "adjustment_input_rows.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analyses/strict_extractor_robustness"),
    )
    args = parser.parse_args(argv)

    result = run_analysis(
        p0_dir=args.p0_dir,
        grpo_snapshot_dir=args.grpo_snapshot_dir,
        maxrl_snapshot_dir=args.maxrl_snapshot_dir,
        adjustment_input_csv=args.adjustment_input_csv,
        output_dir=args.output_dir,
        pcts=DEFAULT_PCTS,
        expected_indices=DEFAULT_EXPECTED_INDICES,
    )

    print(
        "Strict extractor = canonical explicit patterns, "
        "no generic last-number fallback"
    )
    print(
        f"P0: current={result['p0']['current_C']:.4f} "
        f"strict={result['p0']['strict_C']:.4f} "
        f"disagree={result['p0']['disagreement_rate']:.4f}"
    )
    print(
        f"GRPO endpoint: current={result['grpo_endpoint']['current_C']:.4f} "
        f"strict={result['grpo_endpoint']['strict_C']:.4f}"
    )
    if result["maxrl_endpoint"] is not None:
        print(
            f"MaxRL endpoint: "
            f"current={result['maxrl_endpoint']['current_C']:.4f} "
            f"strict={result['maxrl_endpoint']['strict_C']:.4f}"
        )

    print("Low-nonzero p0 adjusted correctness:")
    for row in result["primary_low_bin"]:
        print(
            f"  {row['snapshot_pct']:2d}%  "
            f"U current={_pp(row['current_delta_C_unexposed'])} "
            f"strict={_pp(row['strict_delta_C_unexposed'])}; "
            f"E current={_pp(row['current_delta_C_exposed'])} "
            f"strict={_pp(row['strict_delta_C_exposed'])}; "
            f"U-E strict={_pp(row['strict_gap_U_minus_E'])}"
        )
    print(f"outputs: {result['output_dir']}")


if __name__ == "__main__":
    main()
