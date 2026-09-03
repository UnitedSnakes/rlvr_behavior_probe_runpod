from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


CANONICAL_PI0_LINEAGE_ID = (
    "f89fc90226a67a6a3c7374f9c13abadfcecda88f397ab812fa4130f1f425605b"
)
_EXPECTED_RUNTIME_BATCH = {
    "world_size": 2,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "global_optimizer_batch_size": 32,
    "generation_batch_size": 32,
    "steps_per_generation": 4,
    "num_generations": 16,
    "unique_prompts_per_generation_batch": 2,
}
_EXPECTED_OBJECTIVE = {
    "objective_family": "MaxRL",
    "objective_intervention": "replace_group_advantages_only",
    "advantage_estimator": "practical_maxrl",
    "rollouts_per_prompt": 16,
    "effective_maxrl_order": 15,
    "all_failure_behavior": "zero_group_gradient",
    "maxrl_denominator_epsilon": 0.0,
    "grouping_semantics": "trl_global_reward_order_grouped_by_num_generations",
}
_EXPECTED_TRAINER_COMPOSITION = [
    "trl.GRPOTrainer",
    "PracticalMaxRLTrainer",
    "SignalLedgerGRPOTrainer",
]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite_number(value, *, name: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{name} must be numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{name} must be finite")
    return result


def _expected_advantage(*, reward: float, successes: int, group_size: int) -> float:
    if successes in {0, group_size}:
        return 0.0
    if reward == 1.0:
        return (group_size - successes) / successes
    return -1.0


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSONL in {path.name} line {line_number}"
                ) from error
            _require(isinstance(row, dict), f"ledger row in {path.name} must be an object")
            rows.append(row)
    return rows


def validate_maxrl_pilot(output_dir: Path) -> dict[str, object]:
    """Fail-closed structural acceptance check for the frozen 20-step MaxRL pilot."""
    destination = Path(output_dir)
    manifest_path = destination / "maxrl_run_manifest.json"
    _require(manifest_path.is_file(), "missing maxrl_run_manifest.json")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("invalid maxrl_run_manifest.json") from error
    _require(isinstance(manifest, dict), "MaxRL manifest must be a JSON object")

    _require(manifest.get("mode") == "pilot", "MaxRL pilot manifest mode must be pilot")
    _require(
        manifest.get("scientific_use") is False,
        "MaxRL pilot must have scientific_use=false",
    )
    _require(manifest.get("pilot_steps") == 20, "MaxRL pilot must contain exactly 20 steps")
    _require(
        manifest.get("pi0_lineage_id") == CANONICAL_PI0_LINEAGE_ID,
        "MaxRL pilot pi0_lineage_id does not match corrected canonical pi0",
    )
    _require(
        manifest.get("runtime_batch") == _EXPECTED_RUNTIME_BATCH,
        "MaxRL pilot runtime_batch does not match the frozen 2xA40 geometry",
    )

    config = manifest.get("config")
    _require(isinstance(config, dict), "MaxRL pilot manifest config must be present")
    _require(config.get("num_generations") == 16, "MaxRL pilot requires G=16")
    _require(
        config.get("vllm_importance_sampling_mode") == "token_truncate",
        "MaxRL pilot requires token_truncate importance sampling",
    )

    objective = manifest.get("objective")
    _require(isinstance(objective, dict), "MaxRL pilot objective metadata must be present")
    for key, expected in _EXPECTED_OBJECTIVE.items():
        _require(
            objective.get(key) == expected,
            f"MaxRL objective provenance mismatch for {key}",
        )
    _require(
        objective.get("trainer_composition") == _EXPECTED_TRAINER_COMPOSITION,
        "MaxRL objective trainer composition mismatch",
    )

    ledger_meta = manifest.get("signal_ledger")
    _require(
        isinstance(ledger_meta, dict) and ledger_meta.get("enabled") is True,
        "MaxRL signal ledger must be enabled",
    )
    _require(
        ledger_meta.get("step_semantics") == "generation_global_step",
        "MaxRL signal ledger step semantics mismatch",
    )
    _require(
        ledger_meta.get("actual_is_semantics") == "token_truncate",
        "MaxRL signal ledger must record token-level IS diagnostics",
    )

    ledger_dir = destination / str(ledger_meta.get("directory", "signal_ledger"))
    _require(ledger_dir.is_dir(), "missing MaxRL signal_ledger directory")
    ledger_files = sorted(ledger_dir.glob("*.jsonl"))
    _require(len(ledger_files) == 2, "MaxRL pilot requires exactly 2 rank ledger files")

    rows: list[dict] = []
    for path in ledger_files:
        rows.extend(_read_jsonl(path))
    _require(len(rows) == 640, "MaxRL 20-step pilot requires exactly 640 rollout rows")

    rank_counts: dict[int, int] = defaultdict(int)
    step_counts: dict[int, int] = defaultdict(int)
    groups: dict[tuple[int, int], list[dict]] = defaultdict(list)
    token_ratio_count = 0
    token_ratio_sum = 0.0
    token_ratio_sq_sum = 0.0

    for row in rows:
        step = row.get("generation_global_step")
        rank = row.get("rank")
        dataset_index = row.get("dataset_index")
        _require(isinstance(step, int) and not isinstance(step, bool), "invalid generation_global_step")
        _require(isinstance(rank, int) and not isinstance(rank, bool), "invalid rank")
        _require(isinstance(dataset_index, int) and not isinstance(dataset_index, bool), "invalid dataset_index")
        _require(rank in {0, 1}, "MaxRL pilot ledger rank must be 0 or 1")
        rank_counts[rank] += 1
        step_counts[step] += 1
        groups[(step, dataset_index)].append(row)

        count = row.get("actual_is_ratio_count")
        ratio_sum = row.get("actual_is_ratio_sum")
        ratio_sq_sum = row.get("actual_is_ratio_sq_sum")
        _require(
            isinstance(count, int) and not isinstance(count, bool) and count > 0,
            "token-level IS diagnostics require positive actual_is_ratio_count",
        )
        ratio_sum_value = _finite_number(ratio_sum, name="actual_is_ratio_sum")
        ratio_sq_sum_value = _finite_number(
            ratio_sq_sum, name="actual_is_ratio_sq_sum"
        )
        _require(
            ratio_sq_sum_value > 0.0,
            "token-level IS diagnostics require positive actual_is_ratio_sq_sum",
        )
        token_ratio_count += count
        token_ratio_sum += ratio_sum_value
        token_ratio_sq_sum += ratio_sq_sum_value

    _require(set(rank_counts) == {0, 1}, "MaxRL pilot must contain ranks 0 and 1")
    _require(
        rank_counts[0] == 320 and rank_counts[1] == 320,
        "MaxRL pilot requires 320 rollout rows per rank",
    )
    _require(set(step_counts) == set(range(20)), "MaxRL pilot steps must be exactly 0..19")
    _require(
        all(count == 32 for count in step_counts.values()),
        "MaxRL pilot requires exactly 32 rollout rows per generation step",
    )
    _require(len(groups) == 40, "MaxRL 20-step pilot requires exactly 40 prompt groups")

    max_advantage_error = 0.0
    for (step, dataset_index), group_rows in groups.items():
        _require(
            len(group_rows) == 16,
            f"MaxRL group ({step}, {dataset_index}) must contain exactly 16 rows",
        )
        per_rank = defaultdict(int)
        for row in group_rows:
            per_rank[int(row["rank"])] += 1
        _require(
            per_rank == {0: 8, 1: 8},
            f"MaxRL group ({step}, {dataset_index}) must contain 8 rows per rank",
        )

        rewards = [
            _finite_number(row.get("canonical_reward"), name="canonical_reward")
            for row in group_rows
        ]
        _require(
            all(reward in {0.0, 1.0} for reward in rewards),
            "MaxRL pilot canonical rewards must be binary",
        )
        successes = int(sum(rewards))
        for row, reward in zip(group_rows, rewards, strict=True):
            _require(row.get("group_size") == 16, "MaxRL group_size must equal 16")
            _require(
                row.get("group_successes") == successes,
                "MaxRL group_successes does not match canonical rewards",
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
                "MaxRL advantage identity mismatch for practical MaxRL-15",
            )

    _require(token_ratio_count > 0, "token-level IS diagnostics are missing")
    _require(token_ratio_sq_sum > 0.0, "token-level IS diagnostics are degenerate")
    aggregate_ess = (token_ratio_sum * token_ratio_sum) / token_ratio_sq_sum
    aggregate_ess_fraction = aggregate_ess / token_ratio_count
    _require(
        math.isfinite(aggregate_ess_fraction) and 0.0 < aggregate_ess_fraction <= 1.0 + 1e-9,
        "token-level IS diagnostics produced an invalid aggregate ESS/N",
    )

    return {
        "status": "PASS",
        "steps": 20,
        "rows": len(rows),
        "groups": len(groups),
        "rank_files": len(ledger_files),
        "group_size": 16,
        "max_advantage_error": max_advantage_error,
        "aggregate_token_is_ess_fraction": aggregate_ess_fraction,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Validate the frozen 20-step controlled MaxRL GPU pilot."
    )
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)
    report = validate_maxrl_pilot(args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
