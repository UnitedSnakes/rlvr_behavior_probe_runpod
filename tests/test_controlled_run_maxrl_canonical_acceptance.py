from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from controlled_run.maxrl_canonical_acceptance import (
    CANONICAL_MAXRL_EXECUTION_COMMIT,
    _validate_snapshot_contract,
)
from controlled_run.maxrl_pilot_acceptance import CANONICAL_PI0_LINEAGE_ID
from controlled_run.train_grpo import progress_step_map


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "controlled_run/configs/grpo_qwen3_0_6b.yaml"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_snapshot_contract_accepts_frozen_canonical_schedule(tmp_path):
    steps = 20
    schedule = progress_step_map(steps)
    _write_json(
        tmp_path / "policy_snapshot_schedule.json",
        {
            "max_steps": steps,
            "percentage_to_step": {str(k): v for k, v in schedule.items()},
            "pi0_lineage_id": CANONICAL_PI0_LINEAGE_ID,
        },
    )
    for pct, step in schedule.items():
        policy_dir = tmp_path / f"pi_{pct:03d}"
        policy_dir.mkdir(parents=True)
        (policy_dir / "config.json").write_text("{}", encoding="utf-8")
        _write_json(
            policy_dir / "policy_metadata.json",
            {
                "actual_step": step,
                "target_percentage": pct,
                "pi0_lineage_id": CANONICAL_PI0_LINEAGE_ID,
            },
        )

    observed = _validate_snapshot_contract(
        tmp_path,
        lineage=CANONICAL_PI0_LINEAGE_ID,
        expected_steps=steps,
    )

    assert observed == schedule


def test_snapshot_contract_fails_closed_on_wrong_step(tmp_path):
    steps = 20
    schedule = progress_step_map(steps)
    _write_json(
        tmp_path / "policy_snapshot_schedule.json",
        {
            "max_steps": steps,
            "percentage_to_step": {str(k): v for k, v in schedule.items()},
            "pi0_lineage_id": CANONICAL_PI0_LINEAGE_ID,
        },
    )
    for pct, step in schedule.items():
        policy_dir = tmp_path / f"pi_{pct:03d}"
        policy_dir.mkdir(parents=True)
        (policy_dir / "config.json").write_text("{}", encoding="utf-8")
        _write_json(
            policy_dir / "policy_metadata.json",
            {
                "actual_step": step,
                "target_percentage": pct,
                "pi0_lineage_id": CANONICAL_PI0_LINEAGE_ID,
            },
        )

    broken = tmp_path / "pi_025" / "policy_metadata.json"
    payload = json.loads(broken.read_text(encoding="utf-8"))
    payload["actual_step"] += 1
    _write_json(broken, payload)

    with pytest.raises(ValueError, match="step mismatch"):
        _validate_snapshot_contract(
            tmp_path,
            lineage=CANONICAL_PI0_LINEAGE_ID,
            expected_steps=steps,
        )


def test_canonical_execution_commit_is_frozen_to_authorized_launch_commit():
    assert (
        CANONICAL_MAXRL_EXECUTION_COMMIT
        == "981475795538eee391c7e86aa022ee609b539770"
    )


def test_current_grpo_config_remains_frozen_for_canonical_maxrl():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["seed"] == 42
    assert config["num_generations"] == 16
    assert config["generation_batch_size"] == 32
    assert config["loss_type"] == "dapo"
    assert config["vllm_importance_sampling_mode"] == "token_truncate"
