from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from analyses import snapshot_crossfit_trajectory as sct


def _rollout(*, reward: int, terminated: bool, correct: bool) -> dict:
    return {
        "canonical_reward": reward,
        "terminated": terminated,
        "correct": correct,
    }


def _p0_record(index: int, a: list[dict], b: list[dict]) -> dict:
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


def _snapshot_record(index: int, *, reward: int, terminated: int, correct: int, k: int = 4) -> dict:
    return {
        "dataset_index": index,
        "question_seed": 4_275_000 + index,
        "n_rollouts": k,
        "n_reward": reward,
        "n_terminated": terminated,
        "n_correct": correct,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


@pytest.mark.parametrize(
    ("p", "expected"),
    [
        (0.0, "0"),
        (0.125, "(0,.25]"),
        (0.25, "(0,.25]"),
        (0.5, "(.25,.5]"),
        (0.75, "(.5,.75]"),
        (0.875, "(.75,1)"),
        (1.0, "1"),
    ],
)
def test_assign_reward_bin_uses_frozen_boundaries(p: float, expected: str) -> None:
    assert sct.assign_reward_bin(p) == expected


def test_crossfit_uses_opposite_half_for_baseline_subtraction() -> None:
    # A has reward=0, T=0.5, C=0.5; B has reward=1, T=1, C=1.
    p0 = _p0_record(
        0,
        [
            _rollout(reward=0, terminated=False, correct=True),
            _rollout(reward=0, terminated=True, correct=False),
        ],
        [
            _rollout(reward=1, terminated=True, correct=True),
            _rollout(reward=1, terminated=True, correct=True),
        ],
    )
    snapshot = _snapshot_record(0, reward=2, terminated=3, correct=3)

    rows = sct.build_crossfit_question_rows([p0], [snapshot], snapshot_pct=50)
    assert len(rows) == 2

    a_bin = next(row for row in rows if row["direction"] == "A-bin/B-base")
    assert a_bin["bin"] == "0"
    assert a_bin["baseline_R"] == 1.0
    assert a_bin["baseline_T"] == 1.0
    assert a_bin["baseline_C"] == 1.0
    assert a_bin["snapshot_R"] == 0.5
    assert a_bin["snapshot_T"] == 0.75
    assert a_bin["snapshot_C"] == 0.75
    assert a_bin["delta_R"] == -0.5
    assert a_bin["delta_T"] == -0.25
    assert a_bin["delta_C"] == -0.25

    b_bin = next(row for row in rows if row["direction"] == "B-bin/A-base")
    assert b_bin["bin"] == "1"
    assert b_bin["baseline_R"] == 0.0
    assert b_bin["baseline_T"] == 0.5
    assert b_bin["baseline_C"] == 0.5
    assert b_bin["delta_R"] == 0.5
    assert b_bin["delta_T"] == 0.25
    assert b_bin["delta_C"] == 0.25


def test_symmetric_bin_mean_weights_crossfit_directions_equally() -> None:
    rows = [
        {
            "snapshot_pct": 25,
            "direction": "A-bin/B-base",
            "bin": "(0,.25]",
            "dataset_index": 0,
            "delta_R": 0.2,
            "delta_T": 0.3,
            "delta_C": 0.1,
        },
        {
            "snapshot_pct": 25,
            "direction": "A-bin/B-base",
            "bin": "(0,.25]",
            "dataset_index": 1,
            "delta_R": 0.4,
            "delta_T": 0.5,
            "delta_C": 0.3,
        },
        {
            "snapshot_pct": 25,
            "direction": "B-bin/A-base",
            "bin": "(0,.25]",
            "dataset_index": 2,
            "delta_R": 0.8,
            "delta_T": 0.7,
            "delta_C": 0.6,
        },
    ]

    out = sct.aggregate_crossfit_rows(rows)
    sym = next(
        row
        for row in out
        if row["direction"] == "symmetric" and row["bin"] == "(0,.25]"
    )

    # Direction A mean R=(.2+.4)/2=.3; direction B mean=.8;
    # symmetric cross-fit mean gives each direction equal weight: (.3+.8)/2=.55.
    assert math.isclose(sym["delta_R"], 0.55)
    assert math.isclose(sym["delta_T"], 0.55)
    assert math.isclose(sym["delta_C"], 0.4)
    assert sym["n_questions_A"] == 2
    assert sym["n_questions_B"] == 1


def test_p0_record_validation_recomputes_all_three_metrics() -> None:
    record = _p0_record(
        7,
        [_rollout(reward=0, terminated=False, correct=True)],
        [_rollout(reward=1, terminated=True, correct=True)],
    )

    metrics = sct.validate_p0_record(record)
    assert metrics["A"] == {"R": 0.0, "T": 0.0, "C": 1.0}
    assert metrics["B"] == {"R": 1.0, "T": 1.0, "C": 1.0}

    corrupted = dict(record, p0=0.75)
    with pytest.raises(ValueError, match="p0 aggregate mismatch"):
        sct.validate_p0_record(corrupted)


def test_run_analysis_merges_parity_shards_and_writes_trajectory_outputs(tmp_path: Path) -> None:
    p0_dir = tmp_path / "p0"
    snapshot_dir = tmp_path / "snapshots"
    output_dir = tmp_path / "analysis"

    even = _p0_record(
        0,
        [_rollout(reward=0, terminated=False, correct=True)] * 2,
        [_rollout(reward=1, terminated=True, correct=True)] * 2,
    )
    odd = _p0_record(
        1,
        [
            _rollout(reward=0, terminated=True, correct=False),
            _rollout(reward=1, terminated=True, correct=True),
        ],
        [
            _rollout(reward=0, terminated=False, correct=True),
            _rollout(reward=1, terminated=True, correct=True),
        ],
    )
    _write_jsonl(p0_dir / "rollouts_shard0of2.jsonl", [even])
    _write_jsonl(p0_dir / "rollouts_shard1of2.jsonl", [odd])

    for pct, reward in ((5, 1), (10, 2)):
        _write_jsonl(
            snapshot_dir / f"pi_{pct:03d}" / "snapshot_raw.jsonl",
            [
                _snapshot_record(0, reward=reward, terminated=3, correct=3),
                _snapshot_record(1, reward=reward, terminated=3, correct=3),
            ],
        )

    result = sct.run_analysis(
        p0_dir=p0_dir,
        snapshot_dir=snapshot_dir,
        output_dir=output_dir,
        snapshot_pcts=[5, 10],
        expected_indices=[0, 1],
        verify_known_aggregates=False,
    )

    assert result["p0_records"] == 2
    assert result["snapshots"] == 2
    assert (output_dir / "per_question_crossfit.csv").is_file()
    assert (output_dir / "crossfit_trajectory.csv").is_file()
    assert (output_dir / "aggregate_sanity.csv").is_file()
    for metric in ("R", "T", "C"):
        assert (output_dir / f"delta_{metric}_by_p0_bin.png").is_file()

    with (output_dir / "aggregate_sanity.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["snapshot_pct"]) for row in rows] == [0, 5, 10]

    with (output_dir / "per_question_crossfit.csv").open(newline="", encoding="utf-8") as handle:
        per_question = list(csv.DictReader(handle))
    assert len(per_question) == 2 * 2 * 2  # questions × snapshots × directions
