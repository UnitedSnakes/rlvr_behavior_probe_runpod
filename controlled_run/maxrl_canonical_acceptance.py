from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from controlled_run.config import validate_grpo_config
from controlled_run.maxrl_pilot_acceptance import (
    CANONICAL_PI0_LINEAGE_ID,
    _EXPECTED_OBJECTIVE,
    _EXPECTED_RUNTIME_BATCH,
    _EXPECTED_TRAINER_COMPOSITION,
    _expected_advantage,
    _finite_number,
    _read_jsonl,
    _require,
)
from controlled_run.train_grpo import progress_step_map


CANONICAL_MAXRL_STEPS = 3736
CANONICAL_MAXRL_EXECUTION_COMMIT = "981475795538eee391c7e86aa022ee609b539770"


def _load_json_object(path: Path, *, label: str) -> dict:
    _require(path.is_file(), f"missing {path.name}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid {label}") from error
    _require(isinstance(payload, dict), f"{label} must be a JSON object")
    return payload


def _validate_snapshot_contract(
    destination: Path,
    *,
    lineage: str,
    expected_steps: int,
) -> dict[str, int]:
    schedule = _load_json_object(
        destination / "policy_snapshot_schedule.json",
        label="policy_snapshot_schedule.json",
    )
    _require(
        schedule.get("pi0_lineage_id") == lineage,
        "MaxRL snapshot schedule lineage mismatch",
    )
    _require(
        int(schedule.get("max_steps", -1)) == expected_steps,
        "MaxRL snapshot schedule max_steps mismatch",
    )

    expected_schedule = progress_step_map(expected_steps)
    raw_mapping = schedule.get("percentage_to_step")
    _require(
        isinstance(raw_mapping, dict),
        "MaxRL snapshot schedule percentage_to_step must be a mapping",
    )
    actual_schedule = {int(key): int(value) for key, value in raw_mapping.items()}
    _require(
        actual_schedule == expected_schedule,
        "MaxRL snapshot percentage-to-step schedule mismatch",
    )

    for pct, step in expected_schedule.items():
        policy_dir = destination / f"pi_{pct:03d}"
        _require(policy_dir.is_dir(), f"missing MaxRL policy snapshot pi_{pct:03d}")
        _require(
            (policy_dir / "config.json").is_file(),
            f"MaxRL snapshot pi_{pct:03d} is missing config.json",
        )
        metadata = _load_json_object(
            policy_dir / "policy_metadata.json",
            label=f"pi_{pct:03d}/policy_metadata.json",
        )
        _require(
            metadata.get("pi0_lineage_id") == lineage,
            f"MaxRL snapshot pi_{pct:03d} lineage mismatch",
        )
        _require(
            int(metadata.get("target_percentage", -1)) == pct,
            f"MaxRL snapshot pi_{pct:03d} percentage mismatch",
        )
        _require(
            int(metadata.get("actual_step", -1)) == step,
            f"MaxRL snapshot pi_{pct:03d} step mismatch",
        )

    return expected_schedule


def validate_maxrl_canonical(
    output_dir: Path,
    *,
    expected_steps: int = CANONICAL_MAXRL_STEPS,
    expected_execution_commit: str = CANONICAL_MAXRL_EXECUTION_COMMIT,
) -> dict[str, object]:
    """Fail-closed structural acceptance for a completed canonical MaxRL run."""
    destination = Path(output_dir)
    manifest = _load_json_object(
        destination / "maxrl_run_manifest.json",
        label="maxrl_run_manifest.json",
    )

    _require(manifest.get("mode") == "canonical", "MaxRL canonical manifest mode must be canonical")
    _require(
        manifest.get("scientific_use") is True,
        "MaxRL canonical run must have scientific_use=true",
    )
    _require(
        manifest.get("pilot_steps") is None,
        "MaxRL canonical run must not contain a pilot-step override",
    )
    _require(
        manifest.get("pi0_lineage_id") == CANONICAL_PI0_LINEAGE_ID,
        "MaxRL canonical pi0_lineage_id does not match corrected canonical pi0",
    )
    _require(
        manifest.get("runtime_batch") == _EXPECTED_RUNTIME_BATCH,
        "MaxRL canonical runtime_batch does not match the frozen 2xA40 geometry",
    )

    config = manifest.get("config")
    _require(isinstance(config, dict), "MaxRL canonical manifest config must be present")
    validate_grpo_config(config)

    objective = manifest.get("objective")
    _require(isinstance(objective, dict), "MaxRL canonical objective metadata must be present")
    for key, expected in _EXPECTED_OBJECTIVE.items():
        _require(
            objective.get(key) == expected,
            f"MaxRL canonical objective provenance mismatch for {key}",
        )
    _require(
        objective.get("trainer_composition") == _EXPECTED_TRAINER_COMPOSITION,
        "MaxRL canonical trainer composition mismatch",
    )

    execution_commit_path = destination / "execution_git_commit.txt"
    _require(
        execution_commit_path.is_file(),
        "missing canonical execution_git_commit.txt provenance",
    )
    execution_commit = execution_commit_path.read_text(encoding="utf-8").strip()
    _require(
        execution_commit == expected_execution_commit,
        "canonical MaxRL execution commit mismatch",
    )

    lineage = str(manifest["pi0_lineage_id"])
    snapshot_schedule = _validate_snapshot_contract(
        destination,
        lineage=lineage,
        expected_steps=expected_steps,
    )

    ledger_meta = manifest.get("signal_ledger")
    _require(
        isinstance(ledger_meta, dict) and ledger_meta.get("enabled") is True,
        "MaxRL canonical signal ledger must be enabled",
    )
    _require(
        ledger_meta.get("step_semantics") == "generation_global_step",
        "MaxRL canonical signal ledger step semantics mismatch",
    )
    _require(
        ledger_meta.get("actual_is_semantics") == "token_truncate",
        "MaxRL canonical signal ledger must record token-level IS diagnostics",
    )

    ledger_dir = destination / str(ledger_meta.get("directory", "signal_ledger"))
    _require(ledger_dir.is_dir(), "missing MaxRL canonical signal_ledger directory")
    ledger_files = sorted(ledger_dir.glob("*.jsonl"))
    _require(
        len(ledger_files) == 2,
        "MaxRL canonical run requires exactly 2 rank ledger files",
    )

    rows: list[dict] = []
    for path in ledger_files:
        rows.extend(_read_jsonl(path))

    expected_rows = expected_steps * 32
    expected_groups = expected_steps * 2
    _require(
        len(rows) == expected_rows,
        f"MaxRL canonical run requires exactly {expected_rows} rollout rows",
    )

    rank_counts: dict[int, int] = defaultdict(int)
    step_counts: dict[int, int] = defaultdict(int)
    step_rank_counts: dict[tuple[int, int], int] = defaultdict(int)
    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)

    token_ratio_count = 0
    token_ratio_sum = 0.0
    token_ratio_sq_sum = 0.0
    nonfinite_numeric_fields = 0

    for row in rows:
        for value in row.values():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not math.isfinite(float(value)):
                    nonfinite_numeric_fields += 1

        step = row.get("generation_global_step")
        rank = row.get("rank")
        dataset_index = row.get("dataset_index")
        _require(isinstance(step, int) and not isinstance(step, bool), "invalid generation_global_step")
        _require(isinstance(rank, int) and not isinstance(rank, bool), "invalid rank")
        _require(isinstance(dataset_index, int) and not isinstance(dataset_index, bool), "invalid dataset_index")
        _require(rank in {0, 1}, "MaxRL canonical ledger rank must be 0 or 1")

        rank_counts[rank] += 1
        step_counts[step] += 1
        step_rank_counts[(step, rank)] += 1
        groups[(step, dataset_index)].append(row)

        count = row.get("actual_is_ratio_count")
        _require(
            isinstance(count, int) and not isinstance(count, bool) and count > 0,
            "token-level IS diagnostics require positive actual_is_ratio_count",
        )
        ratio_sum = _finite_number(row.get("actual_is_ratio_sum"), name="actual_is_ratio_sum")
        ratio_sq_sum = _finite_number(
            row.get("actual_is_ratio_sq_sum"),
            name="actual_is_ratio_sq_sum",
        )
        _require(
            ratio_sq_sum > 0.0,
            "token-level IS diagnostics require positive actual_is_ratio_sq_sum",
        )
        token_ratio_count += count
        token_ratio_sum += ratio_sum
        token_ratio_sq_sum += ratio_sq_sum

    _require(
        nonfinite_numeric_fields == 0,
        f"MaxRL canonical ledger contains {nonfinite_numeric_fields} non-finite numeric fields",
    )
    _require(set(rank_counts) == {0, 1}, "MaxRL canonical run must contain ranks 0 and 1")
    _require(
        set(step_counts) == set(range(expected_steps)),
        f"MaxRL canonical steps must be exactly 0..{expected_steps - 1}",
    )
    _require(
        all(count == 32 for count in step_counts.values()),
        "MaxRL canonical run requires exactly 32 rollout rows per generation step",
    )
    _require(
        all(
            step_rank_counts[(step, rank)] == 16
            for step in range(expected_steps)
            for rank in (0, 1)
        ),
        "MaxRL canonical run requires exactly 16 rollout rows per rank per generation step",
    )
    _require(
        rank_counts[0] == expected_rows // 2
        and rank_counts[1] == expected_rows // 2,
        "MaxRL canonical rank row counts are unbalanced",
    )
    _require(
        len(groups) == expected_groups,
        f"MaxRL canonical run requires exactly {expected_groups} prompt groups",
    )

    max_advantage_error = 0.0
    for (step, dataset_index), group_rows in groups.items():
        _require(
            len(group_rows) == 16,
            f"MaxRL group ({step}, {dataset_index}) must contain exactly 16 rows",
        )
        rewards = [
            _finite_number(row.get("canonical_reward"), name="canonical_reward")
            for row in group_rows
        ]
        _require(
            all(reward in {0.0, 1.0} for reward in rewards),
            "MaxRL canonical rewards must be binary",
        )
        successes = int(sum(rewards))
        for row, reward in zip(group_rows, rewards, strict=True):
            _require(row.get("group_size") == 16, "MaxRL canonical group_size must equal 16")
            _require(
                row.get("group_successes") == successes,
                "MaxRL canonical group_successes does not match canonical rewards",
            )
            observed = _finite_number(row.get("advantage"), name="advantage")
            expected = _expected_advantage(
                reward=reward,
                successes=successes,
                group_size=16,
            )
            error = abs(observed - expected)
            max_advantage_error = max(max_advantage_error, error)
            _require(
                error <= 1e-6,
                "MaxRL canonical advantage identity mismatch for practical MaxRL-15",
            )

    aggregate_ess = (token_ratio_sum * token_ratio_sum) / token_ratio_sq_sum
    aggregate_ess_fraction = aggregate_ess / token_ratio_count
    _require(
        math.isfinite(aggregate_ess_fraction)
        and 0.0 < aggregate_ess_fraction <= 1.0 + 1e-9,
        "MaxRL canonical token-level IS diagnostics produced invalid aggregate ESS/N",
    )

    return {
        "status": "PASS",
        "mode": "canonical",
        "scientific_use": True,
        "execution_commit": execution_commit,
        "steps": expected_steps,
        "rows": len(rows),
        "groups": len(groups),
        "rank_files": len(ledger_files),
        "group_size": 16,
        "snapshots": len(snapshot_schedule),
        "snapshot_final_step": snapshot_schedule[100],
        "max_advantage_error": max_advantage_error,
        "aggregate_token_is_ess_fraction": aggregate_ess_fraction,
        "nonfinite_numeric_fields": nonfinite_numeric_fields,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Validate the completed canonical practical MaxRL-15 trajectory."
    )
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    report = validate_maxrl_canonical(args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
