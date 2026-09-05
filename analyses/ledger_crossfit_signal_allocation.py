#!/usr/bin/env python3
"""Cross-fit canonical GRPO signal allocation over frozen p0 reward bins.

The confirmatory quantities in this module come directly from the preregistered
signal ledger: realized group state, realized advantage, completion length, and
actual token-level vLLM importance-sampling diagnostics. The A/B p0 halves are
used only to define two independent difficulty stratifications, which are then
symmetrized by equally weighting the two direction-level estimates.

One additional quantity is intentionally labeled exploratory:

    exploratory_dapo_is_abs_mass
        = sum_rollouts |advantage_i| * sum_active_tokens rho_{i,t}

It is a pre-PPO DAPO numerator-mass proxy, not an exact gradient norm. The
ledger does not contain the full optimizer-time tokenwise PPO ratios/clipping or
gradient-vector geometry required for an exact gradient attribution.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from analyses.snapshot_crossfit_trajectory import (
    BIN_ORDER,
    assign_reward_bin,
    load_p0_records,
    read_jsonl,
    write_csv,
)


DIRECTIONS = ("A-bin", "B-bin")
DEFAULT_EXPECTED_INDICES = list(range(256))
DEFAULT_SNAPSHOT_SCHEDULE = {
    5: 187,
    10: 374,
    15: 560,
    20: 747,
    25: 934,
    30: 1121,
    35: 1308,
    40: 1494,
    45: 1681,
    50: 1868,
    55: 2055,
    60: 2242,
    65: 2428,
    70: 2615,
    75: 2802,
    80: 2989,
    85: 3176,
    90: 3362,
    95: 3549,
    100: 3736,
}

LEDGER_FILE_RE = re.compile(r"^signal_ledger_(?P<launch>.+)_rank(?P<rank>\d+)\.jsonl$")
KNOWN_LEDGER_FILES = 2
KNOWN_LEDGER_ROWS = 119_552
KNOWN_LEDGER_STEP_MIN = 0
KNOWN_LEDGER_STEP_MAX = 3_735
KNOWN_LEDGER_STEPS = 3_736
KNOWN_LEDGER_ROWS_PER_STEP = 32
KNOWN_LEDGER_GROUPS = 7_472
KNOWN_LEDGER_GROUP_SIZE = 16
KNOWN_LEDGER_RANKS = {0, 1}

# These quantities are defined even when a direction has seen zero panel
# prompts: zero exposure implies zero cumulative signal mass for that direction.
CUMULATIVE_SYMMETRIC_SCALARS = (
    "exposure_fraction",
    "cumulative_abs_advantage_per_panel_question",
    "exploratory_dapo_is_abs_mass_per_panel_question",
)

# These are conditional on at least one exposed group/token. If either
# cross-fit direction is undefined at an early sparse snapshot, the symmetric
# value remains undefined rather than borrowing the other direction.
CONDITIONAL_SYMMETRIC_SCALARS = (
    "active_group_fraction",
    "mean_k_over_G",
    "mean_group_total_abs_advantage",
    "mean_completion_length",
    "actual_is_ess_fraction",
    "actual_is_mean_ratio",
    "actual_is_cap_fraction",
)


def _require_finite(value, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value}")
    return result


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty list")
    return sum(values) / len(values)


def reconstruct_prompt_groups(rows: Iterable[dict]) -> list[dict]:
    """Reconstruct global G-sized prompt groups from per-rank ledger rows.

    Canonical signal-ledger rows are keyed by generation step and dataset index;
    one prompt group may be split across ranks, so rank is deliberately not part
    of the grouping key.
    """
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in rows:
        key = (int(row["generation_global_step"]), int(row["dataset_index"]))
        grouped[key].append(row)

    out: list[dict] = []
    for (step, dataset_index), group_rows in sorted(grouped.items()):
        sizes = {int(row["group_size"]) for row in group_rows}
        successes = {int(row["group_successes"]) for row in group_rows}
        if len(sizes) != 1:
            raise ValueError(
                f"step={step} dataset_index={dataset_index}: inconsistent group_size values {sorted(sizes)}"
            )
        if len(successes) != 1:
            raise ValueError(
                f"step={step} dataset_index={dataset_index}: inconsistent group_successes values {sorted(successes)}"
            )

        group_size = next(iter(sizes))
        group_successes = next(iter(successes))
        if group_size <= 0:
            raise ValueError(f"step={step} dataset_index={dataset_index}: group_size must be positive")
        if len(group_rows) != group_size:
            raise ValueError(
                f"step={step} dataset_index={dataset_index}: expected {group_size} rollout rows, "
                f"observed {len(group_rows)}"
            )
        if not 0 <= group_successes <= group_size:
            raise ValueError(
                f"step={step} dataset_index={dataset_index}: invalid successes "
                f"{group_successes}/{group_size}"
            )

        reward_successes = sum(float(row["canonical_reward"]) for row in group_rows)
        if not math.isclose(
            reward_successes,
            float(group_successes),
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise ValueError(
                f"step={step} dataset_index={dataset_index}: canonical reward sum "
                f"{reward_successes} != group_successes {group_successes}"
            )

        abs_advantage = 0.0
        lengths: list[float] = []
        actual_sum = 0.0
        actual_sq_sum = 0.0
        actual_count = 0
        capped_token_mass = 0.0
        exploratory_mass = 0.0

        for row in group_rows:
            advantage = abs(_require_finite(row["advantage"], "advantage"))
            length = int(row["completion_length"])
            if length <= 0:
                raise ValueError(
                    f"step={step} dataset_index={dataset_index}: completion_length must be positive"
                )

            count = int(row["actual_is_ratio_count"])
            ratio_sum = _require_finite(row["actual_is_ratio_sum"], "actual_is_ratio_sum")
            ratio_sq_sum = _require_finite(
                row["actual_is_ratio_sq_sum"], "actual_is_ratio_sq_sum"
            )
            cap_fraction = _require_finite(
                row["actual_is_ratio_at_upper_cap_fraction"],
                "actual_is_ratio_at_upper_cap_fraction",
            )
            if count <= 0:
                raise ValueError(
                    f"step={step} dataset_index={dataset_index}: actual IS token count must be positive"
                )
            if ratio_sum <= 0.0 or ratio_sq_sum <= 0.0:
                raise ValueError(
                    f"step={step} dataset_index={dataset_index}: actual IS sums must be positive"
                )
            if not 0.0 <= cap_fraction <= 1.0:
                raise ValueError(
                    f"step={step} dataset_index={dataset_index}: invalid cap fraction {cap_fraction}"
                )

            abs_advantage += advantage
            lengths.append(float(length))
            actual_sum += ratio_sum
            actual_sq_sum += ratio_sq_sum
            actual_count += count
            capped_token_mass += cap_fraction * count
            exploratory_mass += advantage * ratio_sum

        out.append(
            {
                "generation_global_step": step,
                "dataset_index": dataset_index,
                "group_successes": group_successes,
                "group_size": group_size,
                "k_over_G": group_successes / group_size,
                "active_group": 0 < group_successes < group_size,
                "rollout_count": len(group_rows),
                "group_total_abs_advantage": abs_advantage,
                "mean_completion_length": _mean(lengths),
                "actual_is_ratio_sum": actual_sum,
                "actual_is_ratio_sq_sum": actual_sq_sum,
                "actual_is_ratio_count": actual_count,
                "actual_is_capped_token_mass": capped_token_mass,
                "exploratory_dapo_is_abs_mass": exploratory_mass,
            }
        )

    return out


def _index_p0_records(p0_records: Iterable[dict], panel_indices: list[int]) -> dict[int, dict]:
    indexed: dict[int, dict] = {}
    for record in p0_records:
        index = int(record["dataset_index"])
        if index in indexed:
            raise ValueError(f"duplicate p0 dataset_index={index}")
        indexed[index] = record

    expected = set(panel_indices)
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        extra = sorted(set(indexed) - expected)
        raise ValueError(f"p0/panel index mismatch; missing={missing}, extra={extra}")
    return indexed


def _token_is_summary(groups: list[dict]) -> tuple[float | None, float | None, float | None]:
    if not groups:
        return None, None, None
    total_sum = sum(float(group["actual_is_ratio_sum"]) for group in groups)
    total_sq_sum = sum(float(group["actual_is_ratio_sq_sum"]) for group in groups)
    total_count = sum(int(group["actual_is_ratio_count"]) for group in groups)
    if total_count <= 0 or total_sq_sum <= 0.0:
        raise ValueError("invalid aggregate actual token IS totals")
    ess_fraction = (total_sum * total_sum) / (total_count * total_sq_sum)
    mean_ratio = total_sum / total_count
    cap_fraction = (
        sum(float(group["actual_is_capped_token_mass"]) for group in groups)
        / total_count
    )
    return ess_fraction, mean_ratio, cap_fraction


def build_crossfit_signal_trajectory(
    p0_records: Iterable[dict],
    ledger_rows: Iterable[dict],
    *,
    snapshot_schedule: dict[int, int],
    panel_indices: Iterable[int],
) -> list[dict]:
    """Aggregate cumulative ledger signal within A-bin and B-bin p0 strata.

    A snapshot saved after optimizer step ``S`` is paired with generation-ledger
    rows satisfying ``generation_global_step < S``. This matches the canonical
    callback/index convention where the final snapshot is step 3736 and the
    final generation ledger index is 3735.
    """
    panel = [int(index) for index in panel_indices]
    if not panel or len(set(panel)) != len(panel):
        raise ValueError("panel_indices must be non-empty and unique")
    if not snapshot_schedule:
        raise ValueError("snapshot_schedule must be non-empty")

    p0_by_index = _index_p0_records(p0_records, panel)
    panel_set = set(panel)
    groups = [
        group
        for group in reconstruct_prompt_groups(ledger_rows)
        if int(group["dataset_index"]) in panel_set
    ]

    bin_members: dict[str, dict[str, list[int]]] = {}
    for direction, p0_field in (("A-bin", "p0_A"), ("B-bin", "p0_B")):
        direction_bins = {label: [] for label in BIN_ORDER}
        for index in panel:
            record = p0_by_index[index]
            label = assign_reward_bin(float(record[p0_field]))
            direction_bins[label].append(index)
        bin_members[direction] = direction_bins

    rows: list[dict] = []
    for pct, snapshot_step in sorted(
        ((int(pct), int(step)) for pct, step in snapshot_schedule.items())
    ):
        if snapshot_step <= 0:
            raise ValueError(f"snapshot step must be positive, got {snapshot_step}")
        visible = [
            group
            for group in groups
            if int(group["generation_global_step"]) < snapshot_step
        ]

        for direction in DIRECTIONS:
            for bin_label in BIN_ORDER:
                members = bin_members[direction][bin_label]
                if not members:
                    continue
                member_set = set(members)
                exposed = [
                    group for group in visible if int(group["dataset_index"]) in member_set
                ]
                unique_exposed = {int(group["dataset_index"]) for group in exposed}
                ess, mean_ratio, cap_fraction = _token_is_summary(exposed)

                n_panel = len(members)
                n_exposed = len(exposed)
                row = {
                    "snapshot_pct": pct,
                    "snapshot_step": snapshot_step,
                    "direction": direction,
                    "bin": bin_label,
                    "n_panel_questions": n_panel,
                    "n_exposed_groups": n_exposed,
                    "n_unique_exposed_questions": len(unique_exposed),
                    "exposure_fraction": len(unique_exposed) / n_panel,
                    "active_group_fraction": (
                        _mean([float(group["active_group"]) for group in exposed])
                        if exposed
                        else None
                    ),
                    "mean_k_over_G": (
                        _mean([float(group["k_over_G"]) for group in exposed])
                        if exposed
                        else None
                    ),
                    "mean_group_total_abs_advantage": (
                        _mean(
                            [
                                float(group["group_total_abs_advantage"])
                                for group in exposed
                            ]
                        )
                        if exposed
                        else None
                    ),
                    "mean_completion_length": (
                        _mean([float(group["mean_completion_length"]) for group in exposed])
                        if exposed
                        else None
                    ),
                    "cumulative_abs_advantage_per_panel_question": (
                        sum(
                            float(group["group_total_abs_advantage"])
                            for group in exposed
                        )
                        / n_panel
                    ),
                    "actual_is_ess_fraction": ess,
                    "actual_is_mean_ratio": mean_ratio,
                    "actual_is_cap_fraction": cap_fraction,
                    "exploratory_dapo_is_abs_mass_per_panel_question": (
                        sum(
                            float(group["exploratory_dapo_is_abs_mass"])
                            for group in exposed
                        )
                        / n_panel
                    ),
                }
                rows.append(row)

    return rows


def _equal_direction_average(a, b) -> float:
    if a is None or b is None:
        raise ValueError("cumulative cross-fit quantities must be defined in both directions")
    return (float(a) + float(b)) / 2.0


def _conditional_direction_average(a, b) -> float | None:
    if a is None or b is None:
        return None
    return (float(a) + float(b)) / 2.0


def symmetrize_signal_trajectory(directional_rows: Iterable[dict]) -> list[dict]:
    """Average A-bin and B-bin direction means with equal direction weight.

    Cumulative quantities remain defined through zero exposure. Conditional
    quantities remain undefined unless both cross-fit directions have observed
    at least one relevant group/token at that snapshot.
    """
    indexed: dict[tuple[int, str, str], dict] = {}
    for row in directional_rows:
        direction = row["direction"]
        if direction not in DIRECTIONS:
            raise ValueError(f"unexpected direction {direction}")
        key = (int(row["snapshot_pct"]), direction, row["bin"])
        if key in indexed:
            raise ValueError(f"duplicate directional signal row {key}")
        indexed[key] = row

    pcts = sorted({key[0] for key in indexed})
    result: list[dict] = []
    for pct in pcts:
        for bin_label in BIN_ORDER:
            a = indexed.get((pct, "A-bin", bin_label))
            b = indexed.get((pct, "B-bin", bin_label))
            if a is None or b is None:
                continue
            if int(a["snapshot_step"]) != int(b["snapshot_step"]):
                raise ValueError(
                    f"snapshot step mismatch between directions for {pct}% {bin_label}"
                )

            row = {
                "snapshot_pct": pct,
                "snapshot_step": int(a["snapshot_step"]),
                "direction": "symmetric",
                "bin": bin_label,
                "n_panel_questions_A": int(a["n_panel_questions"]),
                "n_panel_questions_B": int(b["n_panel_questions"]),
                "n_exposed_groups_A": int(a["n_exposed_groups"]),
                "n_exposed_groups_B": int(b["n_exposed_groups"]),
            }
            for field in CUMULATIVE_SYMMETRIC_SCALARS:
                row[field] = _equal_direction_average(a[field], b[field])
            for field in CONDITIONAL_SYMMETRIC_SCALARS:
                row[field] = _conditional_direction_average(a[field], b[field])
            result.append(row)

    return result


def load_ledger_rows(
    ledger_dir: Path,
    *,
    allow_rank_local_launch_ids: bool = False,
) -> tuple[list[Path], list[dict]]:
    """Load one canonical distributed signal ledger.

    Some distributed launches stamp each rank's filename independently, so
    adjacent ranks can legitimately differ in the filename launch token.
    Filename launch-token equality is therefore optional; rank/file
    consistency and downstream canonical geometry checks remain mandatory.
    """
    root = Path(ledger_dir)
    files = sorted(root.glob("signal_ledger_*_rank*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no signal ledger files found under {root}")

    launches: set[str] = set()
    filename_ranks: set[int] = set()
    rows: list[dict] = []
    for path in files:
        match = LEDGER_FILE_RE.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unexpected ledger filename: {path.name}")
        launch = match.group("launch")
        rank = int(match.group("rank"))
        if rank in filename_ranks:
            raise ValueError(f"duplicate ledger rank file for rank={rank}")
        launches.add(launch)
        filename_ranks.add(rank)

        file_rows = read_jsonl(path)
        for row in file_rows:
            if int(row["rank"]) != rank:
                raise ValueError(
                    f"ledger row/file rank mismatch in {path.name}: "
                    f"row rank={row['rank']}, filename rank={rank}"
                )
        rows.extend(file_rows)

    if len(launches) != 1 and not allow_rank_local_launch_ids:
        raise ValueError(f"ledger files span multiple launches: {sorted(launches)}")
    return files, rows


def verify_canonical_ledger_integrity(
    files: list[Path], rows: list[dict], groups: list[dict]
) -> dict:
    """Fail closed on the already-established canonical transport geometry."""
    if len(files) != KNOWN_LEDGER_FILES:
        raise ValueError(f"expected {KNOWN_LEDGER_FILES} ledger files, found {len(files)}")
    if len(rows) != KNOWN_LEDGER_ROWS:
        raise ValueError(f"expected {KNOWN_LEDGER_ROWS} ledger rows, found {len(rows)}")

    ranks = {int(row["rank"]) for row in rows}
    if ranks != KNOWN_LEDGER_RANKS:
        raise ValueError(f"expected ranks {sorted(KNOWN_LEDGER_RANKS)}, found {sorted(ranks)}")

    step_counts = Counter(int(row["generation_global_step"]) for row in rows)
    steps = sorted(step_counts)
    if not steps or steps[0] != KNOWN_LEDGER_STEP_MIN or steps[-1] != KNOWN_LEDGER_STEP_MAX:
        raise ValueError(
            f"ledger step range mismatch: expected {KNOWN_LEDGER_STEP_MIN}..{KNOWN_LEDGER_STEP_MAX}, "
            f"found {steps[0] if steps else None}..{steps[-1] if steps else None}"
        )
    if len(steps) != KNOWN_LEDGER_STEPS:
        raise ValueError(f"expected {KNOWN_LEDGER_STEPS} distinct steps, found {len(steps)}")
    bad_step_counts = {
        step: count
        for step, count in step_counts.items()
        if count != KNOWN_LEDGER_ROWS_PER_STEP
    }
    if bad_step_counts:
        sample = sorted(bad_step_counts.items())[:10]
        raise ValueError(
            f"expected {KNOWN_LEDGER_ROWS_PER_STEP} rows/step; mismatches include {sample}"
        )

    if len(groups) != KNOWN_LEDGER_GROUPS:
        raise ValueError(f"expected {KNOWN_LEDGER_GROUPS} prompt groups, found {len(groups)}")
    group_sizes = {int(group["group_size"]) for group in groups}
    if group_sizes != {KNOWN_LEDGER_GROUP_SIZE}:
        raise ValueError(
            f"expected group_size={KNOWN_LEDGER_GROUP_SIZE}, found {sorted(group_sizes)}"
        )
    if DEFAULT_SNAPSHOT_SCHEDULE[100] != KNOWN_LEDGER_STEP_MAX + 1:
        raise ValueError("100% snapshot schedule does not close immediately after final ledger step")

    return {
        "ledger_files": len(files),
        "ledger_rows": len(rows),
        "ranks": sorted(ranks),
        "step_min": steps[0],
        "step_max": steps[-1],
        "steps": len(steps),
        "rows_per_step": KNOWN_LEDGER_ROWS_PER_STEP,
        "prompt_groups": len(groups),
        "group_size": KNOWN_LEDGER_GROUP_SIZE,
    }


def basic_ledger_integrity(files: list[Path], rows: list[dict], groups: list[dict]) -> dict:
    steps = [int(row["generation_global_step"]) for row in rows]
    return {
        "ledger_files": len(files),
        "ledger_rows": len(rows),
        "ranks": sorted({int(row["rank"]) for row in rows}),
        "step_min": min(steps) if steps else None,
        "step_max": max(steps) if steps else None,
        "steps": len(set(steps)),
        "prompt_groups": len(groups),
        "group_sizes": sorted({int(group["group_size"]) for group in groups}),
    }


def load_symmetric_movement(path: Path) -> list[dict]:
    rows: list[dict] = []
    seen: set[tuple[int, str]] = set()
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"snapshot_pct", "direction", "bin", "delta_R", "delta_T", "delta_C"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"movement CSV missing required fields {sorted(required)}: {reader.fieldnames}"
            )
        for raw in reader:
            if raw["direction"] != "symmetric":
                continue
            key = (int(raw["snapshot_pct"]), raw["bin"])
            if key in seen:
                raise ValueError(f"duplicate symmetric movement row {key}")
            seen.add(key)
            rows.append(
                {
                    "snapshot_pct": key[0],
                    "bin": key[1],
                    "delta_R": _require_finite(raw["delta_R"], "delta_R"),
                    "delta_T": _require_finite(raw["delta_T"], "delta_T"),
                    "delta_C": _require_finite(raw["delta_C"], "delta_C"),
                }
            )
    if not rows:
        raise ValueError(f"no symmetric movement rows found in {path}")
    return rows


def join_signal_and_movement(signal_rows: list[dict], movement_rows: list[dict]) -> list[dict]:
    signal_by_key = {(int(row["snapshot_pct"]), row["bin"]): row for row in signal_rows}
    movement_by_key = {
        (int(row["snapshot_pct"]), row["bin"]): row for row in movement_rows
    }
    if set(signal_by_key) != set(movement_by_key):
        missing_movement = sorted(set(signal_by_key) - set(movement_by_key))
        extra_movement = sorted(set(movement_by_key) - set(signal_by_key))
        raise ValueError(
            "signal/movement key mismatch; "
            f"missing_movement={missing_movement}, extra_movement={extra_movement}"
        )

    joined: list[dict] = []
    for key in sorted(
        signal_by_key,
        key=lambda item: (item[0], BIN_ORDER.index(item[1])),
    ):
        out = dict(signal_by_key[key])
        movement = movement_by_key[key]
        out["delta_R"] = movement["delta_R"]
        out["delta_T"] = movement["delta_T"]
        out["delta_C"] = movement["delta_C"]
        joined.append(out)
    return joined


def make_signal_plot(
    symmetric_rows: list[dict],
    *,
    field: str,
    out_path: Path,
    ylabel: str,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for bin_label in BIN_ORDER:
        rows = sorted(
            [row for row in symmetric_rows if row["bin"] == bin_label],
            key=lambda row: int(row["snapshot_pct"]),
        )
        points = [
            (int(row["snapshot_pct"]), row[field])
            for row in rows
            if row[field] is not None
        ]
        if not points:
            continue
        ax.plot(
            [point[0] for point in points],
            [float(point[1]) for point in points],
            marker="o",
            label=bin_label,
        )
    ax.set_xlabel("Training progress (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="p0 reward bin")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_analysis(
    *,
    p0_dir: Path,
    ledger_dir: Path,
    movement_csv: Path,
    output_dir: Path,
    snapshot_schedule: dict[int, int] = DEFAULT_SNAPSHOT_SCHEDULE,
    expected_indices: Iterable[int] = DEFAULT_EXPECTED_INDICES,
    verify_known_integrity: bool = True,
    allow_rank_local_launch_ids: bool = False,
    objective_label: str = "GRPO",
) -> dict:
    schedule = {int(pct): int(step) for pct, step in snapshot_schedule.items()}
    indices = [int(index) for index in expected_indices]
    if not schedule:
        raise ValueError("snapshot_schedule must be non-empty")
    if not indices:
        raise ValueError("expected_indices must be non-empty")

    p0_records = load_p0_records(Path(p0_dir), indices)
    ledger_files, ledger_rows = load_ledger_rows(
        Path(ledger_dir),
        allow_rank_local_launch_ids=allow_rank_local_launch_ids,
    )
    groups = reconstruct_prompt_groups(ledger_rows)
    integrity = (
        verify_canonical_ledger_integrity(ledger_files, ledger_rows, groups)
        if verify_known_integrity
        else basic_ledger_integrity(ledger_files, ledger_rows, groups)
    )

    directional = build_crossfit_signal_trajectory(
        p0_records,
        ledger_rows,
        snapshot_schedule=schedule,
        panel_indices=indices,
    )
    symmetric = symmetrize_signal_trajectory(directional)
    movement = load_symmetric_movement(Path(movement_csv))
    joined = join_signal_and_movement(symmetric, movement)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    write_csv(destination / "signal_trajectory_directional.csv", directional)
    write_csv(destination / "signal_trajectory_symmetric.csv", symmetric)
    write_csv(destination / "signal_movement_join.csv", joined)
    (destination / "ledger_integrity.json").write_text(
        json.dumps(integrity, indent=2) + "\n", encoding="utf-8"
    )

    make_signal_plot(
        symmetric,
        field="cumulative_abs_advantage_per_panel_question",
        out_path=destination / "cumulative_abs_advantage_per_panel_question.png",
        ylabel="Cumulative Σ|advantage| per panel question",
        title=f"Canonical realized {objective_label} advantage exposure by frozen p0 bin",
    )
    make_signal_plot(
        symmetric,
        field="active_group_fraction",
        out_path=destination / "active_group_fraction.png",
        ylabel="Active-group fraction among exposed groups",
        title=f"Canonical {objective_label} finite-G active-group rate by frozen p0 bin",
    )
    make_signal_plot(
        symmetric,
        field="exploratory_dapo_is_abs_mass_per_panel_question",
        out_path=destination / "exploratory_dapo_is_abs_mass_per_panel_question.png",
        ylabel="Exploratory |A| × token-IS mass per panel question",
        title=f"Exploratory {objective_label} pre-PPO DAPO numerator-mass proxy by frozen p0 bin",
    )

    panel_set = set(indices)
    panel_groups = [
        group for group in groups if int(group["dataset_index"]) in panel_set
    ]
    return {
        "ledger_files": len(ledger_files),
        "ledger_rows": len(ledger_rows),
        "prompt_groups": len(groups),
        "panel_prompt_groups": len(panel_groups),
        "unique_panel_questions_exposed": len(
            {int(group["dataset_index"]) for group in panel_groups}
        ),
        "directional_rows": len(directional),
        "symmetric_rows": len(symmetric),
        "joined_rows": len(joined),
        "output_dir": str(destination),
        "integrity": integrity,
        "joined": joined,
    }


def _fmt_optional(value, *, scale: float = 1.0, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{scale * float(value):.{digits}f}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Cross-fit canonical ledger signal allocation and join it to fixed-panel movement."
    )
    parser.add_argument(
        "--p0-dir", type=Path, default=Path("p0_train_k32_top_p1_canonical")
    )
    parser.add_argument("--ledger-dir", type=Path, default=Path("signal_ledger"))
    parser.add_argument(
        "--movement-csv",
        type=Path,
        default=Path("analyses/canonical_snapshot_crossfit/crossfit_trajectory.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analyses/canonical_ledger_crossfit_signal"),
    )
    parser.add_argument(
        "--skip-known-integrity-check",
        action="store_true",
        help="Skip canonical transport-geometry verification; not recommended for canonical data.",
    )
    parser.add_argument(
        "--allow-rank-local-launch-ids",
        action="store_true",
        help=(
            "Allow per-rank ledger filenames to carry different launch tokens. "
            "Use only when distributed-run provenance has been independently "
            "validated; rank/file and canonical geometry checks still apply."
        ),
    )
    parser.add_argument(
        "--objective-label",
        choices=("GRPO", "MaxRL"),
        default="GRPO",
        help="Label used only in generated plot titles.",
    )
    args = parser.parse_args(argv)

    result = run_analysis(
        p0_dir=args.p0_dir,
        ledger_dir=args.ledger_dir,
        movement_csv=args.movement_csv,
        output_dir=args.output_dir,
        snapshot_schedule=DEFAULT_SNAPSHOT_SCHEDULE,
        expected_indices=DEFAULT_EXPECTED_INDICES,
        verify_known_integrity=not args.skip_known_integrity_check,
        allow_rank_local_launch_ids=args.allow_rank_local_launch_ids,
        objective_label=args.objective_label,
    )

    integrity = result["integrity"]
    print("Ledger integrity:")
    print(
        f"  files={result['ledger_files']} rows={result['ledger_rows']} "
        f"groups={result['prompt_groups']} panel_groups={result['panel_prompt_groups']} "
        f"unique_panel_exposed={result['unique_panel_questions_exposed']}"
    )
    print(
        f"  steps={integrity.get('step_min')}..{integrity.get('step_max')} "
        f"ranks={integrity.get('ranks')}"
    )

    final_pct = max(DEFAULT_SNAPSHOT_SCHEDULE)
    final_rows = [
        row for row in result["joined"] if int(row["snapshot_pct"]) == final_pct
    ]
    print(f"\n{final_pct}% symmetric p0-bin signal + movement:")
    print(
        f"{'bin':<10} {'exposed':>8} {'active':>8} {'cum|A|':>9} "
        f"{'IS ESS':>8} {'dR(pp)':>8} {'dT(pp)':>8} {'dC(pp)':>8}"
    )
    for row in final_rows:
        print(
            f"{row['bin']:<10} "
            f"{_fmt_optional(row['exposure_fraction'], scale=100.0, digits=1):>8} "
            f"{_fmt_optional(row['active_group_fraction'], scale=100.0, digits=1):>8} "
            f"{_fmt_optional(row['cumulative_abs_advantage_per_panel_question'], digits=3):>9} "
            f"{_fmt_optional(row['actual_is_ess_fraction'], digits=4):>8} "
            f"{_fmt_optional(row['delta_R'], scale=100.0, digits=2):>8} "
            f"{_fmt_optional(row['delta_T'], scale=100.0, digits=2):>8} "
            f"{_fmt_optional(row['delta_C'], scale=100.0, digits=2):>8}"
        )

    print(f"\noutputs: {result['output_dir']}")
    print("CANONICAL LEDGER CROSS-FIT SIGNAL ANALYSIS: PASS")


if __name__ == "__main__":
    main()
