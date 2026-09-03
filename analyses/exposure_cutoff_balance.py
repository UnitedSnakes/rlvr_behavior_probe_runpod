#!/usr/bin/env python3
"""Cutoff-specific pre-outcome balance diagnostics for exposure timing.

This module intentionally has no movement/outcome input. It compares already-
exposed and not-yet-exposed questions at the frozen 25/45/65% snapshot cutoffs
using only covariates measured before RL outcomes are inspected.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from analyses.exposure_split_transfer import (
    DEFAULT_EXPECTED_INDICES,
    DEFAULT_GROUP_SIZE,
    DEFAULT_SPLIT_PCTS,
    DEFAULT_TOTAL_STEPS,
    DIRECTIONS,
    run_balance_analysis,
)
from analyses.ledger_crossfit_signal_allocation import DEFAULT_SNAPSHOT_SCHEDULE
from analyses.snapshot_crossfit_trajectory import BIN_ORDER, write_csv
from controlled_run.constants import BASE_MODEL


# Frozen before exposed-vs-unexposed outcome deblinding.
DELTA_C_EXPOSED_MIN = 0.02
GAP_SMALL = 0.04
GAP_LARGE = 0.08


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average empty values")
    return sum(values) / len(values)


def _sample_variance(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = _mean(values)
    return sum((value - mean) ** 2 for value in values) / (len(values) - 1)


def _pooled_smd(exposed: list[float], unexposed: list[float]) -> float | None:
    var_e = _sample_variance(exposed)
    var_u = _sample_variance(unexposed)
    if var_e is None or var_u is None:
        return None
    df = len(exposed) + len(unexposed) - 2
    if df <= 0:
        return None
    pooled_var = ((len(exposed) - 1) * var_e + (len(unexposed) - 1) * var_u) / df
    if pooled_var <= 0.0:
        return None
    return (_mean(exposed) - _mean(unexposed)) / math.sqrt(pooled_var)


def _metric_summary(exposed: list[float], unexposed: list[float]) -> dict[str, float | None]:
    if not exposed or not unexposed:
        raise ValueError("cutoff balance requires non-empty exposed and unexposed groups")
    mean_e = _mean(exposed)
    mean_u = _mean(unexposed)
    return {
        "mean_exposed": mean_e,
        "mean_unexposed": mean_u,
        "diff": mean_e - mean_u,
        "smd": _pooled_smd(exposed, unexposed),
    }


def summarize_cutoff_balance(
    balance_rows: Iterable[dict],
    *,
    snapshot_schedule: dict[int, int] = DEFAULT_SNAPSHOT_SCHEDULE,
    target_pcts: Iterable[int] = DEFAULT_SPLIT_PCTS,
) -> list[dict]:
    """Compare pre-RL covariates across the actual binary exposure cutoffs."""
    rows = list(balance_rows)
    target = tuple(int(pct) for pct in target_pcts)
    missing = sorted(set(target) - {int(pct) for pct in snapshot_schedule})
    if missing:
        raise ValueError(f"missing target snapshots from schedule: {missing}")

    grouped: dict[tuple[int, str, str], dict[str, list[dict]]] = defaultdict(
        lambda: {"exposed": [], "unexposed": []}
    )
    for row in rows:
        direction = row["direction"]
        if direction not in DIRECTIONS:
            raise ValueError(f"unexpected direction {direction}")
        bin_label = row["bin"]
        if bin_label not in BIN_ORDER:
            raise ValueError(f"unexpected p0 bin {bin_label}")
        exposure_step = int(row["exposure_step"])
        for pct in target:
            snapshot_step = int(snapshot_schedule[pct])
            status = "exposed" if exposure_step < snapshot_step else "unexposed"
            grouped[(pct, direction, bin_label)][status].append(row)

    result: list[dict] = []
    for (pct, direction, bin_label), cell in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            DIRECTIONS.index(item[0][1]),
            BIN_ORDER.index(item[0][2]),
        ),
    ):
        exposed = cell["exposed"]
        unexposed = cell["unexposed"]
        if not exposed or not unexposed:
            # Boundary bins with a single observation can legitimately fail to
            # populate both sides. Do not fabricate a balance statistic.
            continue

        p0 = _metric_summary(
            [float(row["baseline_p0"]) for row in exposed],
            [float(row["baseline_p0"]) for row in unexposed],
        )
        completion = _metric_summary(
            [float(row["baseline_p0_completion_length"]) for row in exposed],
            [float(row["baseline_p0_completion_length"]) for row in unexposed],
        )
        prompt = _metric_summary(
            [float(row["prompt_token_count"]) for row in exposed],
            [float(row["prompt_token_count"]) for row in unexposed],
        )
        result.append(
            {
                "snapshot_pct": pct,
                "snapshot_step": int(snapshot_schedule[pct]),
                "direction": direction,
                "bin": bin_label,
                "n_exposed": len(exposed),
                "n_unexposed": len(unexposed),
                "baseline_p0_mean_exposed": p0["mean_exposed"],
                "baseline_p0_mean_unexposed": p0["mean_unexposed"],
                "baseline_p0_diff_exposed_minus_unexposed": p0["diff"],
                "baseline_p0_smd_exposed_minus_unexposed": p0["smd"],
                "p0_completion_length_mean_exposed": completion["mean_exposed"],
                "p0_completion_length_mean_unexposed": completion["mean_unexposed"],
                "p0_completion_length_diff_exposed_minus_unexposed": completion["diff"],
                "p0_completion_length_smd_exposed_minus_unexposed": completion["smd"],
                "prompt_tokens_mean_exposed": prompt["mean_exposed"],
                "prompt_tokens_mean_unexposed": prompt["mean_unexposed"],
                "prompt_tokens_diff_exposed_minus_unexposed": prompt["diff"],
                "prompt_tokens_smd_exposed_minus_unexposed": prompt["smd"],
            }
        )
    return result


def classify_delta_c_gap(*, delta_c_exposed: float, delta_c_unexposed: float) -> dict:
    """Frozen descriptive DeltaC gap rule; ratio is secondary reference only."""
    exposed = float(delta_c_exposed)
    unexposed = float(delta_c_unexposed)
    if not math.isfinite(exposed) or not math.isfinite(unexposed):
        raise ValueError("DeltaC inputs must be finite")
    gap = unexposed - exposed
    if exposed < DELTA_C_EXPOSED_MIN:
        return {
            "gap_unexposed_minus_exposed": gap,
            "ratio_unexposed_over_exposed": None,
            "classification": "not_classifiable",
        }

    ratio = unexposed / exposed
    if gap >= GAP_SMALL:
        classification = "unexposed_higher"
    elif gap > -GAP_SMALL:
        classification = "transfer_compatible"
    elif gap > -GAP_LARGE:
        classification = "mixed_or_uncertain"
    else:
        classification = "own_exposure_candidate"
    return {
        "gap_unexposed_minus_exposed": gap,
        "ratio_unexposed_over_exposed": ratio,
        "classification": classification,
    }


def run_cutoff_balance(
    *,
    p0_dir: Path,
    ledger_dir: Path,
    output_dir: Path,
    expected_indices: Iterable[int] = DEFAULT_EXPECTED_INDICES,
    total_steps: int = DEFAULT_TOTAL_STEPS,
    expected_group_size: int = DEFAULT_GROUP_SIZE,
    snapshot_schedule: dict[int, int] = DEFAULT_SNAPSHOT_SCHEDULE,
    target_pcts: Iterable[int] = DEFAULT_SPLIT_PCTS,
    tokenizer_name: str = BASE_MODEL,
) -> dict:
    """Run continuous + cutoff-specific pre-outcome diagnostics only."""
    base = run_balance_analysis(
        p0_dir=Path(p0_dir),
        ledger_dir=Path(ledger_dir),
        output_dir=Path(output_dir),
        expected_indices=expected_indices,
        total_steps=total_steps,
        expected_group_size=expected_group_size,
        tokenizer_name=tokenizer_name,
    )
    cutoff = summarize_cutoff_balance(
        base["balance_rows"],
        snapshot_schedule=snapshot_schedule,
        target_pcts=target_pcts,
    )
    destination = Path(output_dir)
    write_csv(destination / "cutoff_balance.csv", cutoff)
    return {**base, "cutoff_balance": cutoff}


def _fmt(value, *, scale: float = 1.0, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{scale * float(value):.{digits}f}"


def _print_cutoff(rows: list[dict]) -> None:
    print("Cutoff-specific pre-outcome balance (E - U):")
    print(
        f"{'pct':>4} {'direction':<15} {'bin':<10} {'nE/nU':>9} "
        f"{'dp0(pp)':>8} {'SMDp0':>7} {'dLen':>8} {'SMDlen':>7} "
        f"{'dTok':>7} {'SMDtok':>7}"
    )
    for row in rows:
        print(
            f"{int(row['snapshot_pct']):>4} {row['direction']:<15} {row['bin']:<10} "
            f"{int(row['n_exposed']):>3}/{int(row['n_unexposed']):<3} "
            f"{_fmt(row['baseline_p0_diff_exposed_minus_unexposed'], scale=100, digits=2):>8} "
            f"{_fmt(row['baseline_p0_smd_exposed_minus_unexposed']):>7} "
            f"{_fmt(row['p0_completion_length_diff_exposed_minus_unexposed'], digits=1):>8} "
            f"{_fmt(row['p0_completion_length_smd_exposed_minus_unexposed']):>7} "
            f"{_fmt(row['prompt_tokens_diff_exposed_minus_unexposed'], digits=1):>7} "
            f"{_fmt(row['prompt_tokens_smd_exposed_minus_unexposed']):>7}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Cutoff-specific pre-outcome balance; never reads movement outcomes."
    )
    parser.add_argument("--p0-dir", type=Path, default=Path("p0_train_k32_top_p1_canonical"))
    parser.add_argument("--ledger-dir", type=Path, default=Path("signal_ledger"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analyses/canonical_exposure_split_transfer"),
    )
    parser.add_argument("--tokenizer", default=BASE_MODEL)
    args = parser.parse_args(argv)

    result = run_cutoff_balance(
        p0_dir=args.p0_dir,
        ledger_dir=args.ledger_dir,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer,
    )
    _print_cutoff(result["cutoff_balance"])
    print(f"\noutputs: {result['output_dir']}")
    print("CUTOFF-SPECIFIC PRE-OUTCOME BALANCE: COMPLETE")


if __name__ == "__main__":
    main()
