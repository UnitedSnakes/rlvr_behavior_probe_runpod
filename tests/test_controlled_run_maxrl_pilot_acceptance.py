from __future__ import annotations

import json
from pathlib import Path

import pytest

import controlled_run.maxrl_pilot_acceptance as acceptance
from controlled_run.maxrl_pilot_acceptance import (
    CANONICAL_PI0_LINEAGE_ID,
    validate_maxrl_pilot,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "controlled_run/configs/grpo_qwen3_0_6b.yaml"


def _load_config() -> dict:
    import yaml

    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _advantage(reward: float, k: int) -> float:
    if k in {0, 16}:
        return 0.0
    return (16 - k) / k if reward == 1.0 else -1.0


def _ledger_row(*, step: int, rank: int, dataset_index: int, reward: float, k: int) -> dict:
    advantage = _advantage(reward, k)
    return {
        "generation_global_step": step,
        "rank": rank,
        "dataset_index": dataset_index,
        "correct": bool(reward),
        "terminated": True,
        "canonical_reward": reward,
        "completion_length": 8,
        "group_successes": k,
        "group_size": 16,
        "group_reward_std": 0.0,
        "advantage": advantage,
        "raw_log_rho": 0.0,
        "effective_log_rho": None,
        "importance_sampling_ratio": None,
        "upper_cap_masked": False,
        "token_delta_count": 8,
        "token_delta_mean": 0.0,
        "token_delta_std": 0.0,
        "token_delta_min": 0.0,
        "token_delta_max": 0.0,
        "token_delta_positive_fraction": 0.0,
        "token_ratio_sum": 8.0,
        "token_ratio_sq_sum": 8.0,
        "token_ratio_gt_clip_fraction": 0.0,
        "token_abs_delta_gt_1_fraction": 0.0,
        "trimmed_token_count_1pct": 8,
        "trimmed_raw_log_rho_1pct": 0.0,
        "actual_is_ratio_count": 8,
        "actual_is_ratio_mean": 1.0,
        "actual_is_ratio_std": 0.0,
        "actual_is_ratio_min": 1.0,
        "actual_is_ratio_max": 1.0,
        "actual_is_ratio_sum": 8.0,
        "actual_is_ratio_sq_sum": 8.0,
        "actual_is_log_ratio_mean": 0.0,
        "actual_is_ratio_at_upper_cap_fraction": 0.0,
    }


def _write_valid_pilot(output_dir: Path, *, steps: int = 20) -> None:
    config = _load_config()
    manifest = {
        "mode": "pilot",
        "scientific_use": False,
        "pilot_steps": steps,
        "config": config,
        "runtime_batch": {
            "world_size": 2,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "global_optimizer_batch_size": 32,
            "generation_batch_size": 32,
            "steps_per_generation": 4,
            "num_generations": 16,
            "unique_prompts_per_generation_batch": 2,
        },
        "pi0_manifest": {"kind": "pi0"},
        "pi0_lineage_id": CANONICAL_PI0_LINEAGE_ID,
        "gsm8k_dataset_sha": "gsm8k-sha",
        "prompt_length_audit": {"max_prompt_tokens": 12},
        "signal_ledger": {
            "enabled": True,
            "directory": "signal_ledger",
            "step_semantics": "generation_global_step",
            "raw_log_rho_semantics": "counterfactual_sequence_sum_of_token_logprob_differences",
            "actual_is_semantics": "token_truncate",
        },
        "core_diagnostics": [],
        "objective": {
            "objective_family": "MaxRL",
            "objective_intervention": "replace_group_advantages_only",
            "advantage_estimator": "practical_maxrl",
            "rollouts_per_prompt": 16,
            "effective_maxrl_order": 15,
            "all_failure_behavior": "zero_group_gradient",
            "maxrl_denominator_epsilon": 0.0,
            "trainer_composition": [
                "trl.GRPOTrainer",
                "PracticalMaxRLTrainer",
                "SignalLedgerGRPOTrainer",
            ],
            "grouping_semantics": "trl_global_reward_order_grouped_by_num_generations",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "maxrl_run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    ledger_dir = output_dir / "signal_ledger"
    ledger_dir.mkdir()
    handles = [
        (ledger_dir / f"signal_ledger_test_rank{rank}.jsonl").open("w", encoding="utf-8")
        for rank in range(2)
    ]
    try:
        for step in range(steps):
            groups = [
                (1000 + 2 * step, 1),
                (1001 + 2 * step, 0 if step % 2 == 0 else 16),
            ]
            for dataset_index, k in groups:
                rewards = [1.0] * k + [0.0] * (16 - k)
                for rollout_index, reward in enumerate(rewards):
                    rank = 0 if rollout_index < 8 else 1
                    row = _ledger_row(
                        step=step,
                        rank=rank,
                        dataset_index=dataset_index,
                        reward=reward,
                        k=k,
                    )
                    handles[rank].write(json.dumps(row) + "\n")
    finally:
        for handle in handles:
            handle.close()


def _rewrite_first_row(output_dir: Path, mutate) -> None:
    path = sorted((output_dir / "signal_ledger").glob("*.jsonl"))[0]
    lines = path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    mutate(row)
    lines[0] = json.dumps(row)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_validate_maxrl_pilot_accepts_exact_two_rank_twenty_step_structure(tmp_path):
    output_dir = tmp_path / "pilot"
    _write_valid_pilot(output_dir)

    report = validate_maxrl_pilot(output_dir)

    assert report["status"] == "PASS"
    assert report["steps"] == 20
    assert report["rows"] == 640
    assert report["groups"] == 40
    assert report["rank_files"] == 2
    assert report["group_size"] == 16
    assert report["max_advantage_error"] == pytest.approx(0.0)
    assert report["aggregate_token_is_ess_fraction"] == pytest.approx(1.0)


def test_validate_maxrl_pilot_fails_closed_on_wrong_advantage(tmp_path):
    output_dir = tmp_path / "pilot"
    _write_valid_pilot(output_dir)
    _rewrite_first_row(output_dir, lambda row: row.__setitem__("advantage", 14.0))

    with pytest.raises(ValueError, match="MaxRL advantage identity"):
        validate_maxrl_pilot(output_dir)


def test_validate_maxrl_pilot_fails_closed_on_wrong_lineage(tmp_path):
    output_dir = tmp_path / "pilot"
    _write_valid_pilot(output_dir)
    manifest_path = output_dir / "maxrl_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["pi0_lineage_id"] = "wrong"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="pi0_lineage_id"):
        validate_maxrl_pilot(output_dir)


def test_validate_maxrl_pilot_fails_closed_on_incomplete_group(tmp_path):
    output_dir = tmp_path / "pilot"
    _write_valid_pilot(output_dir)
    path = sorted((output_dir / "signal_ledger").glob("*.jsonl"))[0]
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[1:]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="640 rollout rows|16 rows"):
        validate_maxrl_pilot(output_dir)


def test_validate_maxrl_pilot_fails_closed_on_token_is_missing(tmp_path):
    output_dir = tmp_path / "pilot"
    _write_valid_pilot(output_dir)
    _rewrite_first_row(
        output_dir,
        lambda row: row.__setitem__("actual_is_ratio_count", None),
    )

    with pytest.raises(ValueError, match="token-level IS diagnostics"):
        validate_maxrl_pilot(output_dir)


def test_maxrl_pilot_acceptance_cli_prints_json_pass_report(tmp_path, capsys):
    output_dir = tmp_path / "pilot"
    _write_valid_pilot(output_dir)

    acceptance.main([str(output_dir)])

    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "PASS"
    assert report["rows"] == 640
    assert report["groups"] == 40
