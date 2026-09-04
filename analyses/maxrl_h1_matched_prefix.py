from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from analyses.ledger_crossfit_signal_allocation import reconstruct_prompt_groups
from analyses.snapshot_crossfit_trajectory import (
    BIN_ORDER,
    assign_reward_bin,
    load_p0_records,
)


EXPECTED_WORLD_SIZE = 2
EXPECTED_ROWS_PER_STEP = 32
EXPECTED_GROUPS_PER_STEP = 2
EXPECTED_GROUP_SIZE = 16
DEFAULT_PREFIX_STEPS = 150
DEFAULT_PANEL_INDICES = list(range(256))

LEDGER_FILE_RE = re.compile(
    r"^signal_ledger_(?P<launch>.+)_rank(?P<rank>\d+)\.jsonl$"
)

DIRECTIONS = (
    ("A-bin", "p0_A"),
    ("B-bin", "p0_B"),
)

CONDITIONAL_FIELDS = (
    "active_group_fraction",
    "zero_group_fraction",
    "all_success_group_fraction",
    "mean_k_over_G",
    "actual_is_ess_fraction",
    "actual_is_mean_ratio",
    "actual_is_cap_fraction",
)

CUMULATIVE_FIELDS = (
    "cumulative_abs_advantage_per_panel_question",
    "exploratory_dapo_is_abs_mass_per_panel_question",
)


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty list")
    return sum(values) / len(values)


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return numerator / denominator


def _token_is_summary(groups: list[dict]) -> tuple[float | None, float | None, float | None]:
    if not groups:
        return None, None, None
    total_sum = sum(float(group["actual_is_ratio_sum"]) for group in groups)
    total_sq_sum = sum(float(group["actual_is_ratio_sq_sum"]) for group in groups)
    total_count = sum(int(group["actual_is_ratio_count"]) for group in groups)
    if total_count <= 0 or total_sq_sum <= 0.0:
        raise ValueError("invalid aggregate token-IS totals")
    ess_fraction = (total_sum * total_sum) / (total_count * total_sq_sum)
    mean_ratio = total_sum / total_count
    cap_fraction = (
        sum(float(group["actual_is_capped_token_mass"]) for group in groups)
        / total_count
    )
    if not 0.0 < ess_fraction <= 1.0 + 1e-9:
        raise ValueError(f"invalid aggregate token-IS ESS/N {ess_fraction}")
    return ess_fraction, mean_ratio, cap_fraction


def load_ledger_prefix(
    ledger_dir: Path,
    *,
    prefix_steps: int,
) -> tuple[list[Path], list[dict], list[dict]]:
    if prefix_steps <= 0:
        raise ValueError("prefix_steps must be positive")

    root = Path(ledger_dir)
    files = sorted(root.glob("signal_ledger_*_rank*.jsonl"))
    if len(files) != EXPECTED_WORLD_SIZE:
        raise ValueError(
            f"expected exactly {EXPECTED_WORLD_SIZE} ledger rank files under {root}, "
            f"found {len(files)}"
        )

    launches: set[str] = set()
    filename_ranks: set[int] = set()
    rows: list[dict] = []

    for path in files:
        match = LEDGER_FILE_RE.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unexpected ledger filename {path.name!r}")
        launches.add(match.group("launch"))
        rank = int(match.group("rank"))
        if rank in filename_ranks:
            raise ValueError(f"duplicate ledger file for rank={rank}")
        filename_ranks.add(rank)

        with path.open("r", encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL {path}:{lineno}: {exc}") from exc
                if int(row["rank"]) != rank:
                    raise ValueError(
                        f"ledger row/file rank mismatch in {path.name}: "
                        f"{row['rank']} != {rank}"
                    )
                step = int(row["generation_global_step"])
                if step < prefix_steps:
                    rows.append(row)

    if len(launches) != 1:
        raise ValueError(f"ledger prefix spans multiple launches: {sorted(launches)}")
    if filename_ranks != {0, 1}:
        raise ValueError(f"expected ledger ranks [0, 1], found {sorted(filename_ranks)}")

    expected_rows = prefix_steps * EXPECTED_ROWS_PER_STEP
    if len(rows) != expected_rows:
        raise ValueError(
            f"expected {expected_rows} prefix rows for {prefix_steps} steps, "
            f"found {len(rows)}"
        )

    step_counts = Counter(int(row["generation_global_step"]) for row in rows)
    if set(step_counts) != set(range(prefix_steps)):
        raise ValueError(
            f"ledger prefix steps must be exactly 0..{prefix_steps - 1}"
        )
    bad_steps = {
        step: count
        for step, count in step_counts.items()
        if count != EXPECTED_ROWS_PER_STEP
    }
    if bad_steps:
        raise ValueError(
            f"expected {EXPECTED_ROWS_PER_STEP} rows per prefix step; "
            f"mismatches={sorted(bad_steps.items())[:10]}"
        )

    groups = reconstruct_prompt_groups(rows)
    expected_groups = prefix_steps * EXPECTED_GROUPS_PER_STEP
    if len(groups) != expected_groups:
        raise ValueError(
            f"expected {expected_groups} prompt groups in prefix, found {len(groups)}"
        )
    sizes = {int(group["group_size"]) for group in groups}
    if sizes != {EXPECTED_GROUP_SIZE}:
        raise ValueError(
            f"expected group_size={EXPECTED_GROUP_SIZE}, found {sorted(sizes)}"
        )

    return files, rows, groups


def assert_matched_prompt_schedule(
    grpo_groups: list[dict],
    maxrl_groups: list[dict],
) -> list[tuple[int, int]]:
    def keys(groups: list[dict]) -> list[tuple[int, int]]:
        return [
            (
                int(group["generation_global_step"]),
                int(group["dataset_index"]),
            )
            for group in groups
        ]

    grpo_keys = keys(grpo_groups)
    maxrl_keys = keys(maxrl_groups)
    if grpo_keys != maxrl_keys:
        grpo_set = set(grpo_keys)
        maxrl_set = set(maxrl_keys)
        raise ValueError(
            "matched-prefix prompt schedule differs between GRPO and MaxRL; "
            f"GRPO-only={sorted(grpo_set - maxrl_set)[:10]}, "
            f"MaxRL-only={sorted(maxrl_set - grpo_set)[:10]}"
        )
    return grpo_keys


def aggregate_objective_by_p0(
    *,
    objective: str,
    p0_records: Iterable[dict],
    groups: list[dict],
    panel_indices: Iterable[int],
) -> tuple[list[dict], list[dict], list[dict]]:
    panel = [int(index) for index in panel_indices]
    if not panel or len(panel) != len(set(panel)):
        raise ValueError("panel_indices must be non-empty and unique")
    panel_set = set(panel)

    p0_by_index: dict[int, dict] = {}
    for record in p0_records:
        index = int(record["dataset_index"])
        if index in p0_by_index:
            raise ValueError(f"duplicate p0 dataset_index={index}")
        p0_by_index[index] = record
    if set(p0_by_index) != panel_set:
        raise ValueError("p0 records do not match the frozen panel")

    panel_groups = [
        group for group in groups if int(group["dataset_index"]) in panel_set
    ]

    directional: list[dict] = []
    weighted_p0_rows: list[dict] = []

    for direction, p0_field in DIRECTIONS:
        bins = {label: [] for label in BIN_ORDER}
        for index in panel:
            p0_value = float(p0_by_index[index][p0_field])
            bins[assign_reward_bin(p0_value)].append(index)

        total_mass = 0.0
        total_weighted_p0 = 0.0
        for group in panel_groups:
            index = int(group["dataset_index"])
            mass = float(group["group_total_abs_advantage"])
            p0_value = float(p0_by_index[index][p0_field])
            total_mass += mass
            total_weighted_p0 += p0_value * mass

        weighted_p0_rows.append(
            {
                "objective": objective,
                "direction": direction,
                "signal_weighted_mean_p0": (
                    total_weighted_p0 / total_mass if total_mass > 0.0 else None
                ),
                "total_abs_advantage_mass": total_mass,
            }
        )

        for bin_label in BIN_ORDER:
            members = bins[bin_label]
            if not members:
                continue
            member_set = set(members)
            exposed = [
                group
                for group in panel_groups
                if int(group["dataset_index"]) in member_set
            ]
            ess, mean_ratio, cap_fraction = _token_is_summary(exposed)

            row = {
                "objective": objective,
                "direction": direction,
                "bin": bin_label,
                "n_panel_questions": len(members),
                "n_exposed_groups": len(exposed),
                "n_unique_exposed_questions": len(
                    {int(group["dataset_index"]) for group in exposed}
                ),
                "exposure_fraction": (
                    len({int(group["dataset_index"]) for group in exposed})
                    / len(members)
                ),
                "active_group_fraction": (
                    _mean([float(group["active_group"]) for group in exposed])
                    if exposed
                    else None
                ),
                "zero_group_fraction": (
                    _mean(
                        [
                            float(int(group["group_successes"]) == 0)
                            for group in exposed
                        ]
                    )
                    if exposed
                    else None
                ),
                "all_success_group_fraction": (
                    _mean(
                        [
                            float(
                                int(group["group_successes"])
                                == int(group["group_size"])
                            )
                            for group in exposed
                        ]
                    )
                    if exposed
                    else None
                ),
                "mean_k_over_G": (
                    _mean([float(group["k_over_G"]) for group in exposed])
                    if exposed
                    else None
                ),
                "cumulative_abs_advantage_per_panel_question": (
                    sum(
                        float(group["group_total_abs_advantage"])
                        for group in exposed
                    )
                    / len(members)
                ),
                "actual_is_ess_fraction": ess,
                "actual_is_mean_ratio": mean_ratio,
                "actual_is_cap_fraction": cap_fraction,
                "exploratory_dapo_is_abs_mass_per_panel_question": (
                    sum(
                        float(group["exploratory_dapo_is_abs_mass"])
                        for group in exposed
                    )
                    / len(members)
                ),
            }
            directional.append(row)

    indexed = {
        (row["direction"], row["bin"]): row
        for row in directional
    }
    symmetric: list[dict] = []
    for bin_label in BIN_ORDER:
        a = indexed.get(("A-bin", bin_label))
        b = indexed.get(("B-bin", bin_label))
        if a is None or b is None:
            continue
        row = {
            "objective": objective,
            "direction": "symmetric",
            "bin": bin_label,
            "n_panel_questions_A": int(a["n_panel_questions"]),
            "n_panel_questions_B": int(b["n_panel_questions"]),
            "n_exposed_groups_A": int(a["n_exposed_groups"]),
            "n_exposed_groups_B": int(b["n_exposed_groups"]),
        }
        for field in ("exposure_fraction",) + CONDITIONAL_FIELDS + CUMULATIVE_FIELDS:
            av = a[field]
            bv = b[field]
            row[field] = (
                None
                if av is None or bv is None
                else (float(av) + float(bv)) / 2.0
            )
        symmetric.append(row)

    return directional, symmetric, weighted_p0_rows


def build_objective_contrast(
    grpo_symmetric: list[dict],
    maxrl_symmetric: list[dict],
) -> list[dict]:
    grpo = {row["bin"]: row for row in grpo_symmetric}
    maxrl = {row["bin"]: row for row in maxrl_symmetric}
    if set(grpo) != set(maxrl):
        raise ValueError(
            f"GRPO/MaxRL symmetric p0-bin mismatch: "
            f"{sorted(grpo)} != {sorted(maxrl)}"
        )

    rows: list[dict] = []
    for bin_label in BIN_ORDER:
        if bin_label not in grpo:
            continue
        g = grpo[bin_label]
        m = maxrl[bin_label]
        g_mass = float(g["cumulative_abs_advantage_per_panel_question"])
        m_mass = float(m["cumulative_abs_advantage_per_panel_question"])
        g_proxy = float(g["exploratory_dapo_is_abs_mass_per_panel_question"])
        m_proxy = float(m["exploratory_dapo_is_abs_mass_per_panel_question"])
        rows.append(
            {
                "bin": bin_label,
                "grpo_cumulative_abs_advantage_per_panel_question": g_mass,
                "maxrl_cumulative_abs_advantage_per_panel_question": m_mass,
                "maxrl_minus_grpo_cumulative_abs_advantage_per_panel_question": (
                    m_mass - g_mass
                ),
                "maxrl_over_grpo_cumulative_abs_advantage_per_panel_question": (
                    _safe_ratio(m_mass, g_mass)
                ),
                "grpo_exploratory_dapo_is_abs_mass_per_panel_question": g_proxy,
                "maxrl_exploratory_dapo_is_abs_mass_per_panel_question": m_proxy,
                "maxrl_over_grpo_exploratory_dapo_is_abs_mass_per_panel_question": (
                    _safe_ratio(m_proxy, g_proxy)
                ),
                "grpo_active_group_fraction": g["active_group_fraction"],
                "maxrl_active_group_fraction": m["active_group_fraction"],
                "grpo_zero_group_fraction": g["zero_group_fraction"],
                "maxrl_zero_group_fraction": m["zero_group_fraction"],
                "grpo_mean_k_over_G": g["mean_k_over_G"],
                "maxrl_mean_k_over_G": m["mean_k_over_G"],
                "grpo_actual_is_ess_fraction": g["actual_is_ess_fraction"],
                "maxrl_actual_is_ess_fraction": m["actual_is_ess_fraction"],
            }
        )
    return rows


def finite_g_prediction_rows(
    *,
    group_size: int = EXPECTED_GROUP_SIZE,
    grpo_epsilon: float = 1e-4,
) -> list[dict]:
    if group_size <= 1:
        raise ValueError("group_size must be greater than 1")
    rows: list[dict] = []
    n = int(group_size)
    for k in range(n + 1):
        if k in (0, n):
            grpo_mass = 0.0
        else:
            centered_abs_mass = 2.0 * k * (n - k) / n
            sample_std = math.sqrt(k * (n - k) / (n * (n - 1)))
            grpo_mass = centered_abs_mass / (sample_std + grpo_epsilon)

        maxrl_mass = 0.0 if k == 0 else 2.0 * (n - k)
        rows.append(
            {
                "K": k,
                "G": n,
                "grpo_group_abs_advantage_mass": grpo_mass,
                "maxrl_group_abs_advantage_mass": maxrl_mass,
                "maxrl_over_grpo_mass": _safe_ratio(maxrl_mass, grpo_mass),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV {destination}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_analysis(
    *,
    reference_root: Path,
    maxrl_root: Path,
    output_dir: Path,
    prefix_steps: int = DEFAULT_PREFIX_STEPS,
    panel_indices: Iterable[int] = DEFAULT_PANEL_INDICES,
) -> dict:
    reference = Path(reference_root)
    maxrl = Path(maxrl_root)
    panel = [int(index) for index in panel_indices]

    p0_records = load_p0_records(
        reference / "p0_train_k32_top_p1_canonical",
        panel,
    )

    _, grpo_rows, grpo_groups = load_ledger_prefix(
        reference / "signal_ledger",
        prefix_steps=prefix_steps,
    )
    _, maxrl_rows, maxrl_groups = load_ledger_prefix(
        maxrl / "signal_ledger",
        prefix_steps=prefix_steps,
    )
    matched_keys = assert_matched_prompt_schedule(grpo_groups, maxrl_groups)

    grpo_directional, grpo_symmetric, grpo_weighted = aggregate_objective_by_p0(
        objective="GRPO",
        p0_records=p0_records,
        groups=grpo_groups,
        panel_indices=panel,
    )
    maxrl_directional, maxrl_symmetric, maxrl_weighted = aggregate_objective_by_p0(
        objective="MaxRL",
        p0_records=p0_records,
        groups=maxrl_groups,
        panel_indices=panel,
    )
    contrasts = build_objective_contrast(grpo_symmetric, maxrl_symmetric)

    weighted_by_obj = {
        objective: {
            row["direction"]: row
            for row in grpo_weighted + maxrl_weighted
            if row["objective"] == objective
        }
        for objective in ("GRPO", "MaxRL")
    }
    weighted_summary: dict[str, float] = {}
    for objective in ("GRPO", "MaxRL"):
        a = weighted_by_obj[objective]["A-bin"]["signal_weighted_mean_p0"]
        b = weighted_by_obj[objective]["B-bin"]["signal_weighted_mean_p0"]
        if a is None or b is None:
            raise ValueError(f"{objective} signal-weighted p0 is undefined")
        weighted_summary[objective] = (float(a) + float(b)) / 2.0

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _write_csv(
        destination / "signal_by_p0_directional.csv",
        grpo_directional + maxrl_directional,
    )
    _write_csv(
        destination / "signal_by_p0_symmetric.csv",
        grpo_symmetric + maxrl_symmetric,
    )
    _write_csv(destination / "objective_contrast.csv", contrasts)
    _write_csv(
        destination / "signal_weighted_p0_directional.csv",
        grpo_weighted + maxrl_weighted,
    )
    _write_csv(
        destination / "finite_g_prediction.csv",
        finite_g_prediction_rows(),
    )

    summary = {
        "status": "COMPLETE_NO_AUTOMATIC_H1_VERDICT",
        "prefix_steps": prefix_steps,
        "matched_prompt_schedule": True,
        "matched_prompt_groups": len(matched_keys),
        "grpo_rows": len(grpo_rows),
        "maxrl_rows": len(maxrl_rows),
        "grpo_groups": len(grpo_groups),
        "maxrl_groups": len(maxrl_groups),
        "grpo_signal_weighted_mean_p0": weighted_summary["GRPO"],
        "maxrl_signal_weighted_mean_p0": weighted_summary["MaxRL"],
        "maxrl_minus_grpo_signal_weighted_mean_p0": (
            weighted_summary["MaxRL"] - weighted_summary["GRPO"]
        ),
        "interpretation": (
            "Negative maxrl_minus_grpo_signal_weighted_mean_p0 is a "
            "predeclared scalar description compatible with a left shift; "
            "the frozen binwise table remains primary and no numerical H1 "
            "threshold was preregistered."
        ),
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        **summary,
        "contrasts": contrasts,
        "output_dir": str(destination),
    }


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the matched first-N-step canonical GRPO and practical "
            "MaxRL realized signal allocation using the frozen K=32 A/B p0 bank."
        )
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path(
            "controlled_run_outputs/reference_inputs/"
            "canonical_grpo_seed42_h1_reference"
        ),
    )
    parser.add_argument(
        "--maxrl-root",
        type=Path,
        default=Path("controlled_run_outputs/maxrl_shakedown_150"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("controlled_run_outputs/maxrl_h1_shakedown_150_analysis"),
    )
    parser.add_argument("--prefix-steps", type=int, default=DEFAULT_PREFIX_STEPS)
    args = parser.parse_args(argv)

    result = run_analysis(
        reference_root=args.reference_root,
        maxrl_root=args.maxrl_root,
        output_dir=args.output_dir,
        prefix_steps=args.prefix_steps,
    )

    print("MATCHED FIRST-N SIGNAL ALLOCATION")
    print(
        f"steps={result['prefix_steps']} groups={result['matched_prompt_groups']} "
        f"schedule_match={result['matched_prompt_schedule']}"
    )
    print(
        "signal-weighted mean p0: "
        f"GRPO={result['grpo_signal_weighted_mean_p0']:.6f} "
        f"MaxRL={result['maxrl_signal_weighted_mean_p0']:.6f} "
        f"delta={result['maxrl_minus_grpo_signal_weighted_mean_p0']:+.6f}"
    )
    print()
    print(
        f"{'bin':<10} {'GRPO|A|':>10} {'MaxRL|A|':>10} "
        f"{'M/G':>8} {'G K0':>8} {'M K0':>8} "
        f"{'G K/G':>8} {'M K/G':>8}"
    )
    for row in result["contrasts"]:
        print(
            f"{row['bin']:<10} "
            f"{_fmt(row['grpo_cumulative_abs_advantage_per_panel_question']):>10} "
            f"{_fmt(row['maxrl_cumulative_abs_advantage_per_panel_question']):>10} "
            f"{_fmt(row['maxrl_over_grpo_cumulative_abs_advantage_per_panel_question']):>8} "
            f"{_fmt(row['grpo_zero_group_fraction']):>8} "
            f"{_fmt(row['maxrl_zero_group_fraction']):>8} "
            f"{_fmt(row['grpo_mean_k_over_G']):>8} "
            f"{_fmt(row['maxrl_mean_k_over_G']):>8}"
        )

    print()
    print(f"outputs: {result['output_dir']}")
    print(result["status"])


if __name__ == "__main__":
    main()
