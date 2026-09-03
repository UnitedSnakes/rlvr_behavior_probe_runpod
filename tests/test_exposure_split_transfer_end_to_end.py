from __future__ import annotations

import csv
import json
from pathlib import Path

from analyses import exposure_split_transfer as est


def _rollout(*, reward: int, n_tokens: int) -> dict:
    return {
        "canonical_reward": reward,
        "terminated": True,
        "correct": bool(reward),
        "n_tokens": n_tokens,
    }


def _p0_record(index: int, *, a_successes: int, b_successes: int) -> dict:
    a = [_rollout(reward=int(i < a_successes), n_tokens=10 + index) for i in range(4)]
    b = [_rollout(reward=int(i < b_successes), n_tokens=20 + index) for i in range(4)]
    all_rows = a + b
    return {
        "dataset_index": index,
        "question": f"question {index}",
        "half_size": 4,
        "n_rollouts": 8,
        "p0_A": a_successes / 4,
        "p0_B": b_successes / 4,
        "p0": (a_successes + b_successes) / 8,
        "correctness_p0": sum(r["correct"] for r in all_rows) / 8,
        "termination_rate": 1.0,
        "rollouts_A": a,
        "rollouts_B": b,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _write_p0(p0_dir: Path) -> None:
    records = [
        _p0_record(0, a_successes=1, b_successes=0),
        _p0_record(1, a_successes=1, b_successes=1),
        _p0_record(2, a_successes=2, b_successes=1),
        _p0_record(3, a_successes=3, b_successes=2),
    ]
    _write_jsonl(p0_dir / "rollouts_shard0of2.jsonl", [records[0], records[2]])
    _write_jsonl(p0_dir / "rollouts_shard1of2.jsonl", [records[1], records[3]])


def _write_ledger(ledger_dir: Path) -> None:
    # Each panel question appears at exactly one global generation step and has
    # G=2 rollout rows split across rank files.
    rank0 = []
    rank1 = []
    for index, step in enumerate((0, 1, 2, 3)):
        common = {
            "generation_global_step": step,
            "dataset_index": index,
            "group_size": 2,
        }
        rank0.append({**common, "rank": 0})
        rank1.append({**common, "rank": 1})
    _write_jsonl(ledger_dir / "signal_ledger_fixture_rank0.jsonl", rank0)
    _write_jsonl(ledger_dir / "signal_ledger_fixture_rank1.jsonl", rank1)


def _write_per_question(path: Path) -> None:
    rows = []
    for pct in (25, 45, 65):
        for direction in est.DIRECTIONS:
            for index in range(4):
                rows.append(
                    {
                        "snapshot_pct": pct,
                        "direction": direction,
                        "dataset_index": index,
                        "bin": "(0,.25]",
                        "delta_R": 0.01 * (index + 1),
                        "delta_T": 0.02 * (index + 1),
                        "delta_C": 0.03 * (index + 1),
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_balance_analysis_does_not_read_outcome_file(tmp_path: Path) -> None:
    p0_dir = tmp_path / "p0"
    ledger_dir = tmp_path / "ledger"
    out_dir = tmp_path / "out"
    _write_p0(p0_dir)
    _write_ledger(ledger_dir)

    result = est.run_balance_analysis(
        p0_dir=p0_dir,
        ledger_dir=ledger_dir,
        output_dir=out_dir,
        expected_indices=[0, 1, 2, 3],
        total_steps=4,
        expected_group_size=2,
        prompt_token_counts={0: 11, 1: 12, 2: 13, 3: 14},
    )

    assert result["n_panel_questions"] == 4
    assert result["n_balance_rows"] == 8
    assert (out_dir / "balance_question_rows.csv").exists()
    assert (out_dir / "balance_summary.csv").exists()
    assert not (out_dir / "split_contrasts.csv").exists()


def test_full_analysis_writes_predeclared_split_outputs(tmp_path: Path) -> None:
    p0_dir = tmp_path / "p0"
    ledger_dir = tmp_path / "ledger"
    movement = tmp_path / "per_question.csv"
    out_dir = tmp_path / "out"
    _write_p0(p0_dir)
    _write_ledger(ledger_dir)
    _write_per_question(movement)

    result = est.run_analysis(
        p0_dir=p0_dir,
        ledger_dir=ledger_dir,
        per_question_csv=movement,
        output_dir=out_dir,
        expected_indices=[0, 1, 2, 3],
        total_steps=4,
        expected_group_size=2,
        snapshot_schedule={25: 1, 45: 2, 65: 3},
        target_pcts=(25, 45, 65),
        prompt_token_counts={0: 11, 1: 12, 2: 13, 3: 14},
    )

    assert result["target_pcts"] == [25, 45, 65]
    assert (out_dir / "balance_summary.csv").exists()
    assert (out_dir / "split_directional.csv").exists()
    assert (out_dir / "split_symmetric.csv").exists()
    assert (out_dir / "split_contrasts.csv").exists()
