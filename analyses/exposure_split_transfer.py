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

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from analyses.ledger_crossfit_signal_allocation import (
    DEFAULT_SNAPSHOT_SCHEDULE,
    load_ledger_rows,
)
from analyses.snapshot_crossfit_trajectory import (
    BIN_ORDER,
    assign_reward_bin,
    load_p0_records,
    validate_p0_record,
    write_csv,
)
from controlled_run.constants import BASE_MODEL, CONTROLLED_SYSTEM_PROMPT


DIRECTIONS = ("A-bin/B-base", "B-bin/A-base")
DEFAULT_SPLIT_PCTS = (25, 45, 65)
DEFAULT_EXPECTED_INDICES = tuple(range(256))
DEFAULT_TOTAL_STEPS = 3736
DEFAULT_GROUP_SIZE = 16
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


def build_exposure_split_directional(
    per_question_rows: Iterable[dict],
    *,
    exposure_steps: dict[int, int],
    snapshot_schedule: dict[int, int],
    target_pcts: Iterable[int] = DEFAULT_SPLIT_PCTS,
) -> list[dict]:
    """Split direction-level fixed-panel movement by own-exposure status.

    A question is exposed to snapshot S iff its unique ledger exposure step is
    strictly less than S. This matches the saved-policy boundary used by the
    canonical ledger/snapshot join.
    """
    target = tuple(int(pct) for pct in target_pcts)
    if len(set(target)) != len(target):
        raise ValueError("target_pcts must be unique")
    missing_schedule = sorted(set(target) - {int(p) for p in snapshot_schedule})
    if missing_schedule:
        raise ValueError(f"target snapshots missing from schedule: {missing_schedule}")

    selected: list[dict] = []
    for raw in per_question_rows:
        pct = int(raw["snapshot_pct"])
        if pct not in target:
            continue
        direction = raw["direction"]
        if direction not in DIRECTIONS:
            continue
        index = int(raw["dataset_index"])
        if index not in exposure_steps:
            raise ValueError(f"missing exposure step for dataset_index={index}")
        step = int(snapshot_schedule[pct])
        status = "exposed" if int(exposure_steps[index]) < step else "unexposed"
        selected.append(
            {
                "snapshot_pct": pct,
                "snapshot_step": step,
                "direction": direction,
                "bin": raw["bin"],
                "dataset_index": index,
                "exposure_status": status,
                "delta_R": float(raw["delta_R"]),
                "delta_T": float(raw["delta_T"]),
                "delta_C": float(raw["delta_C"]),
            }
        )

    if not selected:
        raise ValueError("no per-question rows matched target snapshots/directions")

    grouped: dict[tuple[int, str, str, str], list[dict]] = defaultdict(list)
    for row in selected:
        key = (
            int(row["snapshot_pct"]),
            row["direction"],
            row["bin"],
            row["exposure_status"],
        )
        grouped[key].append(row)

    out: list[dict] = []
    for (pct, direction, bin_label, status), group in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            DIRECTIONS.index(item[0][1]),
            BIN_ORDER.index(item[0][2]),
            0 if item[0][3] == "exposed" else 1,
        ),
    ):
        snapshot_steps = {int(row["snapshot_step"]) for row in group}
        if len(snapshot_steps) != 1:
            raise ValueError("inconsistent snapshot step within split cell")
        out.append(
            {
                "snapshot_pct": pct,
                "snapshot_step": next(iter(snapshot_steps)),
                "direction": direction,
                "bin": bin_label,
                "exposure_status": status,
                "n_questions": len(group),
                "delta_R": _mean([float(row["delta_R"]) for row in group]),
                "delta_T": _mean([float(row["delta_T"]) for row in group]),
                "delta_C": _mean([float(row["delta_C"]) for row in group]),
            }
        )
    return out


def symmetrize_exposure_split(directional_rows: Iterable[dict]) -> list[dict]:
    """Equal-weight A/B direction means within each exposure-status cell."""
    indexed: dict[tuple[int, str, str, str], dict] = {}
    for row in directional_rows:
        direction = row["direction"]
        if direction not in DIRECTIONS:
            raise ValueError(f"unexpected direction {direction}")
        status = row["exposure_status"]
        if status not in {"exposed", "unexposed"}:
            raise ValueError(f"unexpected exposure status {status}")
        key = (int(row["snapshot_pct"]), direction, row["bin"], status)
        if key in indexed:
            raise ValueError(f"duplicate directional split row {key}")
        indexed[key] = row

    pcts = sorted({key[0] for key in indexed})
    out: list[dict] = []
    for pct in pcts:
        for bin_label in BIN_ORDER:
            for status in ("exposed", "unexposed"):
                a = indexed.get((pct, "A-bin/B-base", bin_label, status))
                b = indexed.get((pct, "B-bin/A-base", bin_label, status))
                if a is None or b is None:
                    continue
                if int(a["snapshot_step"]) != int(b["snapshot_step"]):
                    raise ValueError("snapshot step mismatch between cross-fit directions")
                out.append(
                    {
                        "snapshot_pct": pct,
                        "snapshot_step": int(a["snapshot_step"]),
                        "direction": "symmetric",
                        "bin": bin_label,
                        "exposure_status": status,
                        "n_questions_A": int(a["n_questions"]),
                        "n_questions_B": int(b["n_questions"]),
                        "delta_R": (float(a["delta_R"]) + float(b["delta_R"])) / 2.0,
                        "delta_T": (float(a["delta_T"]) + float(b["delta_T"])) / 2.0,
                        "delta_C": (float(a["delta_C"]) + float(b["delta_C"])) / 2.0,
                    }
                )
    return out


def build_transfer_contrasts(symmetric_rows: Iterable[dict]) -> list[dict]:
    """Pair exposed/unexposed cells and apply the frozen DeltaC ratio rule."""
    indexed: dict[tuple[int, str, str], dict] = {}
    for row in symmetric_rows:
        if row.get("direction") != "symmetric":
            raise ValueError("transfer contrasts require symmetric rows")
        key = (int(row["snapshot_pct"]), row["bin"], row["exposure_status"])
        if key in indexed:
            raise ValueError(f"duplicate symmetric split row {key}")
        indexed[key] = row

    out: list[dict] = []
    for pct in sorted({key[0] for key in indexed}):
        for bin_label in BIN_ORDER:
            exposed = indexed.get((pct, bin_label, "exposed"))
            unexposed = indexed.get((pct, bin_label, "unexposed"))
            if exposed is None or unexposed is None:
                continue
            criterion = classify_transfer_ratio(
                delta_c_exposed=float(exposed["delta_C"]),
                delta_c_unexposed=float(unexposed["delta_C"]),
            )
            out.append(
                {
                    "snapshot_pct": pct,
                    "snapshot_step": int(exposed["snapshot_step"]),
                    "bin": bin_label,
                    "n_exposed_A": int(exposed["n_questions_A"]),
                    "n_exposed_B": int(exposed["n_questions_B"]),
                    "n_unexposed_A": int(unexposed["n_questions_A"]),
                    "n_unexposed_B": int(unexposed["n_questions_B"]),
                    "delta_R_exposed": float(exposed["delta_R"]),
                    "delta_R_unexposed": float(unexposed["delta_R"]),
                    "delta_T_exposed": float(exposed["delta_T"]),
                    "delta_T_unexposed": float(unexposed["delta_T"]),
                    "delta_C_exposed": float(exposed["delta_C"]),
                    "delta_C_unexposed": float(unexposed["delta_C"]),
                    "transfer_ratio_C": criterion["ratio"],
                    "transfer_classification": criterion["classification"],
                }
            )
    return out


def build_exposure_steps(
    ledger_rows: Iterable[dict],
    *,
    expected_indices: Iterable[int],
    expected_group_size: int,
) -> dict[int, int]:
    """Recover each panel question's unique own-exposure ledger step."""
    indices = [int(index) for index in expected_indices]
    expected = set(indices)
    if len(expected) != len(indices) or not indices:
        raise ValueError("expected_indices must be non-empty and unique")
    if expected_group_size <= 0:
        raise ValueError("expected_group_size must be positive")

    steps: dict[int, set[int]] = {index: set() for index in indices}
    counts: dict[int, int] = {index: 0 for index in indices}
    for row in ledger_rows:
        index = int(row["dataset_index"])
        if index not in expected:
            continue
        group_size = int(row["group_size"])
        if group_size != expected_group_size:
            raise ValueError(
                f"dataset_index={index}: expected group_size={expected_group_size}, got {group_size}"
            )
        steps[index].add(int(row["generation_global_step"]))
        counts[index] += 1

    result: dict[int, int] = {}
    for index in indices:
        if len(steps[index]) != 1:
            raise ValueError(
                f"dataset_index={index}: expected exactly one own-exposure step, got {sorted(steps[index])}"
            )
        if counts[index] != expected_group_size:
            raise ValueError(
                f"dataset_index={index}: expected {expected_group_size} rollout rows at own exposure, got {counts[index]}"
            )
        result[index] = next(iter(steps[index]))
    return result


def compute_prompt_token_counts(
    p0_records: Iterable[dict],
    *,
    tokenizer_name: str = BASE_MODEL,
) -> dict[int, int]:
    """Count frozen pre-exposure prompt tokens using the canonical chat template."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    counts: dict[int, int] = {}
    for record in p0_records:
        index = int(record["dataset_index"])
        question = record.get("question")
        if not isinstance(question, str) or not question:
            raise ValueError(f"dataset_index={index}: missing p0 question text")
        messages = [
            {"role": "system", "content": CONTROLLED_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        token_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        if not token_ids:
            raise ValueError(f"dataset_index={index}: empty rendered prompt tokenization")
        if index in counts:
            raise ValueError(f"duplicate p0 dataset_index={index}")
        counts[index] = len(token_ids)
    return counts


def load_per_question_movement(path: Path) -> list[dict]:
    """Read the fixed-panel question-level cross-fit outcome file."""
    rows: list[dict] = []
    required = {
        "snapshot_pct",
        "direction",
        "dataset_index",
        "bin",
        "delta_R",
        "delta_T",
        "delta_C",
    }
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"per-question movement CSV missing required fields {sorted(required)}: {reader.fieldnames}"
            )
        for raw in reader:
            if raw["direction"] not in DIRECTIONS:
                continue
            rows.append(
                {
                    "snapshot_pct": int(raw["snapshot_pct"]),
                    "direction": raw["direction"],
                    "dataset_index": int(raw["dataset_index"]),
                    "bin": raw["bin"],
                    "delta_R": float(raw["delta_R"]),
                    "delta_T": float(raw["delta_T"]),
                    "delta_C": float(raw["delta_C"]),
                }
            )
    if not rows:
        raise ValueError(f"no direction-level per-question movement rows found in {path}")
    return rows


def run_balance_analysis(
    *,
    p0_dir: Path,
    ledger_dir: Path,
    output_dir: Path,
    expected_indices: Iterable[int] = DEFAULT_EXPECTED_INDICES,
    total_steps: int = DEFAULT_TOTAL_STEPS,
    expected_group_size: int = DEFAULT_GROUP_SIZE,
    prompt_token_counts: dict[int, int] | None = None,
    tokenizer_name: str = BASE_MODEL,
) -> dict:
    """Run only pre-outcome balance diagnostics; no outcome file is opened."""
    indices = [int(index) for index in expected_indices]
    p0_records = load_p0_records(Path(p0_dir), indices)
    ledger_files, ledger_rows = load_ledger_rows(Path(ledger_dir))
    exposure_steps = build_exposure_steps(
        ledger_rows,
        expected_indices=indices,
        expected_group_size=expected_group_size,
    )
    token_counts = (
        {int(k): int(v) for k, v in prompt_token_counts.items()}
        if prompt_token_counts is not None
        else compute_prompt_token_counts(p0_records, tokenizer_name=tokenizer_name)
    )
    if set(token_counts) != set(indices):
        raise ValueError("prompt-token indices must match expected panel indices exactly")

    balance_rows = build_balance_rows(
        p0_records,
        exposure_steps=exposure_steps,
        prompt_token_counts=token_counts,
    )
    balance_summary = summarize_balance(balance_rows, total_steps=total_steps)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    write_csv(destination / "balance_question_rows.csv", balance_rows)
    write_csv(destination / "balance_summary.csv", balance_summary)
    write_csv(
        destination / "exposure_steps.csv",
        [
            {
                "dataset_index": index,
                "exposure_step": exposure_steps[index],
                "prompt_token_count": token_counts[index],
            }
            for index in indices
        ],
    )

    return {
        "n_ledger_files": len(ledger_files),
        "n_panel_questions": len(indices),
        "n_balance_rows": len(balance_rows),
        "exposure_steps": exposure_steps,
        "prompt_token_counts": token_counts,
        "balance_rows": balance_rows,
        "balance_summary": balance_summary,
        "output_dir": str(destination),
    }


def run_analysis(
    *,
    p0_dir: Path,
    ledger_dir: Path,
    per_question_csv: Path,
    output_dir: Path,
    expected_indices: Iterable[int] = DEFAULT_EXPECTED_INDICES,
    total_steps: int = DEFAULT_TOTAL_STEPS,
    expected_group_size: int = DEFAULT_GROUP_SIZE,
    snapshot_schedule: dict[int, int] = DEFAULT_SNAPSHOT_SCHEDULE,
    target_pcts: Iterable[int] = DEFAULT_SPLIT_PCTS,
    prompt_token_counts: dict[int, int] | None = None,
    tokenizer_name: str = BASE_MODEL,
) -> dict:
    """Run balance diagnostics, then the predeclared 25/45/65 split."""
    indices = [int(index) for index in expected_indices]
    balance = run_balance_analysis(
        p0_dir=p0_dir,
        ledger_dir=ledger_dir,
        output_dir=output_dir,
        expected_indices=indices,
        total_steps=total_steps,
        expected_group_size=expected_group_size,
        prompt_token_counts=prompt_token_counts,
        tokenizer_name=tokenizer_name,
    )

    per_question = load_per_question_movement(Path(per_question_csv))
    directional = build_exposure_split_directional(
        per_question,
        exposure_steps=balance["exposure_steps"],
        snapshot_schedule=snapshot_schedule,
        target_pcts=target_pcts,
    )
    symmetric = symmetrize_exposure_split(directional)
    contrasts = build_transfer_contrasts(symmetric)

    destination = Path(output_dir)
    write_csv(destination / "split_directional.csv", directional)
    write_csv(destination / "split_symmetric.csv", symmetric)
    write_csv(destination / "split_contrasts.csv", contrasts)

    return {
        **balance,
        "target_pcts": [int(pct) for pct in target_pcts],
        "split_directional": directional,
        "split_symmetric": symmetric,
        "split_contrasts": contrasts,
    }


def _fmt(value, *, scale: float = 1.0, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{scale * float(value):.{digits}f}"


def _print_balance(summary: list[dict]) -> None:
    print("Exposure-timing balance diagnostics (pre-outcome):")
    print(
        f"{'direction':<15} {'bin':<10} {'n':>4} {'r(p0)':>8} {'r(p0len)':>9} "
        f"{'r(prompt)':>9} {'KS_D':>7} {'Q1':>6} {'Q2':>6} {'Q3':>6} {'Q4':>6}"
    )
    for row in summary:
        print(
            f"{row['direction']:<15} {row['bin']:<10} {int(row['n_questions']):>4} "
            f"{_fmt(row['corr_exposure_baseline_p0']):>8} "
            f"{_fmt(row['corr_exposure_p0_completion_length']):>9} "
            f"{_fmt(row['corr_exposure_prompt_tokens']):>9} "
            f"{_fmt(row['uniform_ks_D']):>7} "
            f"{_fmt(row['exposure_q1_fraction'], scale=100, digits=1):>6} "
            f"{_fmt(row['exposure_q2_fraction'], scale=100, digits=1):>6} "
            f"{_fmt(row['exposure_q3_fraction'], scale=100, digits=1):>6} "
            f"{_fmt(row['exposure_q4_fraction'], scale=100, digits=1):>6}"
        )


def _print_split(contrasts: list[dict]) -> None:
    for pct in DEFAULT_SPLIT_PCTS:
        rows = [row for row in contrasts if int(row["snapshot_pct"]) == pct]
        print(f"\n{pct}% exposed vs unexposed symmetric cross-fit:")
        print(
            f"{'bin':<10} {'nE A/B':>11} {'nU A/B':>11} "
            f"{'dR_E':>7} {'dR_U':>7} {'dT_E':>7} {'dT_U':>7} "
            f"{'dC_E':>7} {'dC_U':>7} {'ratioC':>7} {'class':>22}"
        )
        for row in rows:
            print(
                f"{row['bin']:<10} "
                f"{int(row['n_exposed_A'])}/{int(row['n_exposed_B']):<5} "
                f"{int(row['n_unexposed_A'])}/{int(row['n_unexposed_B']):<5} "
                f"{_fmt(row['delta_R_exposed'], scale=100, digits=2):>7} "
                f"{_fmt(row['delta_R_unexposed'], scale=100, digits=2):>7} "
                f"{_fmt(row['delta_T_exposed'], scale=100, digits=2):>7} "
                f"{_fmt(row['delta_T_unexposed'], scale=100, digits=2):>7} "
                f"{_fmt(row['delta_C_exposed'], scale=100, digits=2):>7} "
                f"{_fmt(row['delta_C_unexposed'], scale=100, digits=2):>7} "
                f"{_fmt(row['transfer_ratio_C'], digits=2):>7} "
                f"{row['transfer_classification']:>22}"
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Balance-first own-exposure versus transfer analysis on the canonical train panel."
    )
    parser.add_argument(
        "--p0-dir", type=Path, default=Path("p0_train_k32_top_p1_canonical")
    )
    parser.add_argument("--ledger-dir", type=Path, default=Path("signal_ledger"))
    parser.add_argument(
        "--per-question-csv",
        type=Path,
        default=Path("analyses/canonical_snapshot_crossfit/per_question_crossfit.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analyses/canonical_exposure_split_transfer"),
    )
    parser.add_argument("--tokenizer", default=BASE_MODEL)
    parser.add_argument(
        "--balance-only",
        action="store_true",
        help="Run only pre-outcome balance diagnostics; do not read movement outcomes.",
    )
    args = parser.parse_args(argv)

    if args.balance_only:
        result = run_balance_analysis(
            p0_dir=args.p0_dir,
            ledger_dir=args.ledger_dir,
            output_dir=args.output_dir,
            expected_indices=DEFAULT_EXPECTED_INDICES,
            total_steps=DEFAULT_TOTAL_STEPS,
            expected_group_size=DEFAULT_GROUP_SIZE,
            tokenizer_name=args.tokenizer,
        )
        _print_balance(result["balance_summary"])
        print(f"\noutputs: {result['output_dir']}")
        print("EXPOSURE TIMING BALANCE DIAGNOSTICS: COMPLETE")
        return

    result = run_analysis(
        p0_dir=args.p0_dir,
        ledger_dir=args.ledger_dir,
        per_question_csv=args.per_question_csv,
        output_dir=args.output_dir,
        expected_indices=DEFAULT_EXPECTED_INDICES,
        total_steps=DEFAULT_TOTAL_STEPS,
        expected_group_size=DEFAULT_GROUP_SIZE,
        snapshot_schedule=DEFAULT_SNAPSHOT_SCHEDULE,
        target_pcts=DEFAULT_SPLIT_PCTS,
        tokenizer_name=args.tokenizer,
    )
    _print_balance(result["balance_summary"])
    _print_split(result["split_contrasts"])
    print(f"\noutputs: {result['output_dir']}")
    print("CANONICAL EXPOSED-VS-UNEXPOSED ANALYSIS: COMPLETE")


if __name__ == "__main__":
    main()
