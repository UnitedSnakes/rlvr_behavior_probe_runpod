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

import math
from collections import defaultdict
from typing import Iterable

from analyses.snapshot_crossfit_trajectory import BIN_ORDER, assign_reward_bin


DIRECTIONS = ("A-bin", "B-bin")

# Scalar direction-level quantities that can be meaningfully symmetrized by an
# equal-weight average across A-bin and B-bin analyses.
SYMMETRIC_SCALARS = (
    "exposure_fraction",
    "active_group_fraction",
    "mean_k_over_G",
    "mean_group_total_abs_advantage",
    "mean_completion_length",
    "cumulative_abs_advantage_per_panel_question",
    "actual_is_ess_fraction",
    "actual_is_mean_ratio",
    "actual_is_cap_fraction",
    "exploratory_dapo_is_abs_mass_per_panel_question",
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


def _equal_direction_average(a, b):
    if a is None and b is None:
        return None
    if a is None or b is None:
        raise ValueError("cannot symmetrize when only one cross-fit direction is defined")
    return (float(a) + float(b)) / 2.0


def symmetrize_signal_trajectory(directional_rows: Iterable[dict]) -> list[dict]:
    """Average A-bin and B-bin direction means with equal direction weight."""
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
            for field in SYMMETRIC_SCALARS:
                row[field] = _equal_direction_average(a[field], b[field])
            result.append(row)

    return result
