from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from analyses import ledger_crossfit_signal_allocation as lcsa


def _p0_record(index: int, p_a: float, p_b: float) -> dict:
    return {
        "dataset_index": index,
        "p0_A": p_a,
        "p0_B": p_b,
    }


def _rollout(reward: int) -> dict:
    return {
        "canonical_reward": reward,
        "terminated": True,
        "correct": bool(reward),
    }


def _full_p0_record(index: int, a: list[dict], b: list[dict]) -> dict:
    def mean(key: str, rows: list[dict]) -> float:
        return sum(float(r[key]) for r in rows) / len(rows)

    all_rows = a + b
    return {
        "dataset_index": index,
        "half_size": len(a),
        "n_rollouts": len(all_rows),
        "p0_A": mean("canonical_reward", a),
        "p0_B": mean("canonical_reward", b),
        "p0": mean("canonical_reward", all_rows),
        "correctness_p0": mean("correct", all_rows),
        "termination_rate": mean("terminated", all_rows),
        "rollouts_A": a,
        "rollouts_B": b,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def _ledger_row(
    *,
    step: int,
    rank: int,
    index: int,
    k: int,
    g: int,
    advantage: float,
    length: int,
    actual_is_sum: float,
    actual_is_sq_sum: float,
    actual_is_count: int,
    actual_is_cap_fraction: float = 0.0,
) -> dict:
    return {
        "generation_global_step": step,
        "rank": rank,
        "dataset_index": index,
        "correct": bool(k),
        "terminated": True,
        "canonical_reward": float(advantage > 0),
        "completion_length": length,
        "group_successes": k,
        "group_size": g,
        "group_reward_std": 0.5 if 0 < k < g else 0.0,
        "advantage": advantage,
        "raw_log_rho": 0.0,
        "effective_log_rho": None,
        "importance_sampling_ratio": None,
        "upper_cap_masked": False,
        "token_delta_count": actual_is_count,
        "token_delta_mean": 0.0,
        "token_delta_std": 0.0,
        "token_delta_min": 0.0,
        "token_delta_max": 0.0,
        "token_delta_positive_fraction": 0.5,
        "token_ratio_sum": actual_is_sum,
        "token_ratio_sq_sum": actual_is_sq_sum,
        "token_ratio_gt_clip_fraction": 0.0,
        "token_abs_delta_gt_1_fraction": 0.0,
        "trimmed_token_count_1pct": actual_is_count,
        "trimmed_raw_log_rho_1pct": 0.0,
        "actual_is_ratio_count": actual_is_count,
        "actual_is_ratio_mean": actual_is_sum / actual_is_count,
        "actual_is_ratio_std": 0.0,
        "actual_is_ratio_min": 1.0,
        "actual_is_ratio_max": 1.0,
        "actual_is_ratio_sum": actual_is_sum,
        "actual_is_ratio_sq_sum": actual_is_sq_sum,
        "actual_is_log_ratio_mean": 0.0,
        "actual_is_ratio_at_upper_cap_fraction": actual_is_cap_fraction,
    }


def test_reconstruct_groups_merges_rank_rows_and_validates_group_size() -> None:
    rows = [
        _ledger_row(
            step=0, rank=0, index=7, k=1, g=2, advantage=1.0, length=2,
            actual_is_sum=2.0, actual_is_sq_sum=2.0, actual_is_count=2,
        ),
        _ledger_row(
            step=0, rank=1, index=7, k=1, g=2, advantage=-1.0, length=3,
            actual_is_sum=3.0, actual_is_sq_sum=3.0, actual_is_count=3,
        ),
    ]

    groups = lcsa.reconstruct_prompt_groups(rows)
    assert len(groups) == 1
    group = groups[0]
    assert group["generation_global_step"] == 0
    assert group["dataset_index"] == 7
    assert group["group_successes"] == 1
    assert group["group_size"] == 2
    assert group["active_group"] is True
    assert group["rollout_count"] == 2
    assert math.isclose(group["group_total_abs_advantage"], 2.0)
    assert math.isclose(group["mean_completion_length"], 2.5)
    assert math.isclose(group["actual_is_ratio_sum"], 5.0)
    assert math.isclose(group["actual_is_ratio_sq_sum"], 5.0)
    assert group["actual_is_ratio_count"] == 5
    assert math.isclose(group["exploratory_dapo_is_abs_mass"], 5.0)

    with pytest.raises(ValueError, match="expected 2 rollout rows"):
        lcsa.reconstruct_prompt_groups(rows[:1])


def test_crossfit_cumulative_signal_uses_same_p0_bins_and_snapshot_boundary() -> None:
    p0 = [
        _p0_record(0, 0.0, 0.5),
        _p0_record(1, 0.25, 0.75),
    ]
    ledger = [
        _ledger_row(
            step=0, rank=0, index=0, k=1, g=2, advantage=1.0, length=2,
            actual_is_sum=2.0, actual_is_sq_sum=2.0, actual_is_count=2,
        ),
        _ledger_row(
            step=0, rank=1, index=0, k=1, g=2, advantage=-1.0, length=2,
            actual_is_sum=2.0, actual_is_sq_sum=2.0, actual_is_count=2,
        ),
        _ledger_row(
            step=1, rank=0, index=1, k=0, g=2, advantage=0.0, length=3,
            actual_is_sum=3.0, actual_is_sq_sum=3.0, actual_is_count=3,
        ),
        _ledger_row(
            step=1, rank=1, index=1, k=0, g=2, advantage=0.0, length=3,
            actual_is_sum=3.0, actual_is_sq_sum=3.0, actual_is_count=3,
        ),
    ]

    rows = lcsa.build_crossfit_signal_trajectory(
        p0,
        ledger,
        snapshot_schedule={50: 1, 100: 2},
        panel_indices=[0, 1],
    )

    a_50_zero = next(
        r for r in rows
        if r["snapshot_pct"] == 50
        and r["direction"] == "A-bin"
        and r["bin"] == "0"
    )
    assert a_50_zero["n_panel_questions"] == 1
    assert a_50_zero["n_exposed_groups"] == 1
    assert a_50_zero["exposure_fraction"] == 1.0
    assert a_50_zero["active_group_fraction"] == 1.0
    assert math.isclose(a_50_zero["cumulative_abs_advantage_per_panel_question"], 2.0)

    a_50_low = next(
        r for r in rows
        if r["snapshot_pct"] == 50
        and r["direction"] == "A-bin"
        and r["bin"] == "(0,.25]"
    )
    assert a_50_low["n_panel_questions"] == 1
    assert a_50_low["n_exposed_groups"] == 0
    assert a_50_low["exposure_fraction"] == 0.0

    b_bins = {
        r["bin"]: r
        for r in rows
        if r["snapshot_pct"] == 100 and r["direction"] == "B-bin"
    }
    assert b_bins["(.25,.5]"]["n_exposed_groups"] == 1
    assert b_bins["(.5,.75]"]["n_exposed_groups"] == 1
    assert b_bins["(.5,.75]"]["active_group_fraction"] == 0.0


def test_token_is_ess_and_cap_fraction_are_token_weighted() -> None:
    p0 = [_p0_record(0, 0.5, 0.5)]
    ledger = [
        _ledger_row(
            step=0, rank=0, index=0, k=1, g=2, advantage=1.0, length=2,
            actual_is_sum=2.0, actual_is_sq_sum=2.0, actual_is_count=2,
            actual_is_cap_fraction=0.5,
        ),
        _ledger_row(
            step=0, rank=1, index=0, k=1, g=2, advantage=-1.0, length=6,
            actual_is_sum=6.0, actual_is_sq_sum=6.0, actual_is_count=6,
            actual_is_cap_fraction=0.0,
        ),
    ]

    rows = lcsa.build_crossfit_signal_trajectory(
        p0,
        ledger,
        snapshot_schedule={100: 1},
        panel_indices=[0],
    )
    row = next(r for r in rows if r["direction"] == "A-bin")

    assert math.isclose(row["actual_is_ess_fraction"], 1.0)
    assert math.isclose(row["actual_is_mean_ratio"], 1.0)
    assert math.isclose(row["actual_is_cap_fraction"], 1.0 / 8.0)


def test_symmetric_signal_average_weights_crossfit_directions_equally() -> None:
    directional = [
        {
            "snapshot_pct": 100,
            "snapshot_step": 2,
            "direction": "A-bin",
            "bin": "(0,.25]",
            "n_panel_questions": 10,
            "n_exposed_groups": 8,
            "exposure_fraction": 0.8,
            "active_group_fraction": 0.5,
            "mean_k_over_G": 0.25,
            "mean_group_total_abs_advantage": 2.0,
            "mean_completion_length": 100.0,
            "cumulative_abs_advantage_per_panel_question": 1.6,
            "actual_is_ess_fraction": 0.99,
            "actual_is_mean_ratio": 1.0,
            "actual_is_cap_fraction": 0.0,
            "exploratory_dapo_is_abs_mass_per_panel_question": 160.0,
        },
        {
            "snapshot_pct": 100,
            "snapshot_step": 2,
            "direction": "B-bin",
            "bin": "(0,.25]",
            "n_panel_questions": 2,
            "n_exposed_groups": 2,
            "exposure_fraction": 1.0,
            "active_group_fraction": 1.0,
            "mean_k_over_G": 0.5,
            "mean_group_total_abs_advantage": 4.0,
            "mean_completion_length": 200.0,
            "cumulative_abs_advantage_per_panel_question": 4.0,
            "actual_is_ess_fraction": 0.97,
            "actual_is_mean_ratio": 0.98,
            "actual_is_cap_fraction": 0.02,
            "exploratory_dapo_is_abs_mass_per_panel_question": 800.0,
        },
    ]

    sym = lcsa.symmetrize_signal_trajectory(directional)
    row = sym[0]
    assert row["direction"] == "symmetric"
    assert math.isclose(row["exposure_fraction"], 0.9)
    assert math.isclose(row["active_group_fraction"], 0.75)
    assert math.isclose(row["cumulative_abs_advantage_per_panel_question"], 2.8)
    assert math.isclose(row["exploratory_dapo_is_abs_mass_per_panel_question"], 480.0)


def test_sparse_symmetrization_keeps_cumulative_metrics_but_marks_conditionals_undefined() -> None:
    common_a = {
        "snapshot_pct": 5,
        "snapshot_step": 187,
        "direction": "A-bin",
        "bin": "0",
        "n_panel_questions": 3,
        "n_exposed_groups": 0,
        "exposure_fraction": 0.0,
        "active_group_fraction": None,
        "mean_k_over_G": None,
        "mean_group_total_abs_advantage": None,
        "mean_completion_length": None,
        "cumulative_abs_advantage_per_panel_question": 0.0,
        "actual_is_ess_fraction": None,
        "actual_is_mean_ratio": None,
        "actual_is_cap_fraction": None,
        "exploratory_dapo_is_abs_mass_per_panel_question": 0.0,
    }
    common_b = {
        "snapshot_pct": 5,
        "snapshot_step": 187,
        "direction": "B-bin",
        "bin": "0",
        "n_panel_questions": 4,
        "n_exposed_groups": 1,
        "exposure_fraction": 0.25,
        "active_group_fraction": 1.0,
        "mean_k_over_G": 0.25,
        "mean_group_total_abs_advantage": 3.0,
        "mean_completion_length": 120.0,
        "cumulative_abs_advantage_per_panel_question": 0.75,
        "actual_is_ess_fraction": 0.99,
        "actual_is_mean_ratio": 1.0,
        "actual_is_cap_fraction": 0.0,
        "exploratory_dapo_is_abs_mass_per_panel_question": 90.0,
    }

    row = lcsa.symmetrize_signal_trajectory([common_a, common_b])[0]
    assert math.isclose(row["exposure_fraction"], 0.125)
    assert math.isclose(row["cumulative_abs_advantage_per_panel_question"], 0.375)
    assert math.isclose(row["exploratory_dapo_is_abs_mass_per_panel_question"], 45.0)
    assert row["active_group_fraction"] is None
    assert row["mean_k_over_G"] is None
    assert row["mean_group_total_abs_advantage"] is None
    assert row["mean_completion_length"] is None
    assert row["actual_is_ess_fraction"] is None
    assert row["actual_is_mean_ratio"] is None
    assert row["actual_is_cap_fraction"] is None


def test_load_ledger_rows_allows_rank_local_launch_ids_only_when_explicit(
    tmp_path: Path,
) -> None:
    ledger_dir = tmp_path / "signal_ledger"
    ledger_dir.mkdir()

    rank0 = [
        _ledger_row(
            step=0, rank=0, index=0, k=1, g=2, advantage=1.0, length=2,
            actual_is_sum=2.0, actual_is_sq_sum=2.0, actual_is_count=2,
        )
    ]
    rank1 = [
        _ledger_row(
            step=0, rank=1, index=0, k=1, g=2, advantage=-1.0, length=2,
            actual_is_sum=2.0, actual_is_sq_sum=2.0, actual_is_count=2,
        )
    ]

    _write_jsonl(
        ledger_dir / "signal_ledger_20260904T015650Z_rank0.jsonl",
        rank0,
    )
    _write_jsonl(
        ledger_dir / "signal_ledger_20260904T015651Z_rank1.jsonl",
        rank1,
    )

    with pytest.raises(ValueError, match="span multiple launches"):
        lcsa.load_ledger_rows(ledger_dir)

    files, rows = lcsa.load_ledger_rows(
        ledger_dir,
        allow_rank_local_launch_ids=True,
    )
    assert len(files) == 2
    assert len(rows) == 2
    assert {int(row["rank"]) for row in rows} == {0, 1}


def test_run_analysis_reads_canonical_layout_and_joins_fixed_panel_movement(tmp_path: Path) -> None:
    p0_dir = tmp_path / "p0"
    ledger_dir = tmp_path / "signal_ledger"
    output_dir = tmp_path / "analysis"
    movement_csv = tmp_path / "movement.csv"

    half_mid = [_rollout(1), _rollout(1), _rollout(0), _rollout(0)]
    half_low = [_rollout(1), _rollout(0), _rollout(0), _rollout(0)]
    even = _full_p0_record(0, half_mid, half_mid)
    odd = _full_p0_record(1, half_low, half_low)
    _write_jsonl(p0_dir / "rollouts_shard0of2.jsonl", [even])
    _write_jsonl(p0_dir / "rollouts_shard1of2.jsonl", [odd])

    rank0 = [
        _ledger_row(
            step=0, rank=0, index=0, k=1, g=2, advantage=1.0, length=2,
            actual_is_sum=2.0, actual_is_sq_sum=2.0, actual_is_count=2,
        ),
        _ledger_row(
            step=1, rank=0, index=1, k=0, g=2, advantage=0.0, length=3,
            actual_is_sum=3.0, actual_is_sq_sum=3.0, actual_is_count=3,
        ),
    ]
    rank1 = [
        _ledger_row(
            step=0, rank=1, index=0, k=1, g=2, advantage=-1.0, length=2,
            actual_is_sum=2.0, actual_is_sq_sum=2.0, actual_is_count=2,
        ),
        _ledger_row(
            step=1, rank=1, index=1, k=0, g=2, advantage=0.0, length=3,
            actual_is_sum=3.0, actual_is_sq_sum=3.0, actual_is_count=3,
        ),
    ]
    _write_jsonl(ledger_dir / "signal_ledger_20260903T000000Z_rank0.jsonl", rank0)
    _write_jsonl(ledger_dir / "signal_ledger_20260903T000000Z_rank1.jsonl", rank1)

    movement_rows = []
    for pct in (50, 100):
        movement_rows.extend(
            [
                {
                    "snapshot_pct": pct,
                    "direction": "symmetric",
                    "bin": "(0,.25]",
                    "delta_R": 0.1,
                    "delta_T": 0.2,
                    "delta_C": 0.05,
                },
                {
                    "snapshot_pct": pct,
                    "direction": "symmetric",
                    "bin": "(.25,.5]",
                    "delta_R": 0.2,
                    "delta_T": 0.3,
                    "delta_C": 0.1,
                },
            ]
        )
    with movement_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(movement_rows[0]))
        writer.writeheader()
        writer.writerows(movement_rows)

    result = lcsa.run_analysis(
        p0_dir=p0_dir,
        ledger_dir=ledger_dir,
        movement_csv=movement_csv,
        output_dir=output_dir,
        snapshot_schedule={50: 1, 100: 2},
        expected_indices=[0, 1],
        verify_known_integrity=False,
    )

    assert result["ledger_files"] == 2
    assert result["ledger_rows"] == 4
    assert result["prompt_groups"] == 2
    assert result["panel_prompt_groups"] == 2

    directional = output_dir / "signal_trajectory_directional.csv"
    symmetric = output_dir / "signal_trajectory_symmetric.csv"
    joined = output_dir / "signal_movement_join.csv"
    assert directional.exists()
    assert symmetric.exists()
    assert joined.exists()
    assert (output_dir / "cumulative_abs_advantage_per_panel_question.png").exists()
    assert (output_dir / "active_group_fraction.png").exists()
    assert (output_dir / "exploratory_dapo_is_abs_mass_per_panel_question.png").exists()

    with joined.open(newline="", encoding="utf-8") as handle:
        joined_rows = list(csv.DictReader(handle))
    assert len(joined_rows) == 4
    final_mid = next(
        row for row in joined_rows
        if int(row["snapshot_pct"]) == 100 and row["bin"] == "(.25,.5]"
    )
    assert math.isclose(float(final_mid["delta_C"]), 0.1)
    assert math.isclose(float(final_mid["cumulative_abs_advantage_per_panel_question"]), 2.0)
