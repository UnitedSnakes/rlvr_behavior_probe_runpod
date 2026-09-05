from __future__ import annotations

import argparse
import json
from pathlib import Path

from controlled_run.config import load_config, validate_grpo_config
from controlled_run.maxrl import (
    MaxRLRewardBatchRecorder,
    make_practical_maxrl_trainer,
    practical_maxrl_metadata,
)
from controlled_run.train_grpo import _run_controlled_grpo


DEFAULT_CONFIG = Path("controlled_run/configs/grpo_qwen3_0_6b.yaml")
DEFAULT_OUTPUT_DIR = Path("controlled_run_outputs/maxrl")
MAXRL_MANIFEST_NAME = "maxrl_run_manifest.json"


def build_maxrl_objective_metadata(
    config: dict,
    *,
    config_validator=validate_grpo_config,
) -> dict[str, object]:
    """Freeze the practical MaxRL intervention attached to a matched GRPO recipe."""
    config_validator(config)
    metadata = practical_maxrl_metadata(group_size=int(config["num_generations"]))
    return {
        "objective_family": "MaxRL",
        "objective_intervention": "replace_group_advantages_only",
        **metadata,
        "trainer_composition": [
            "trl.GRPOTrainer",
            "PracticalMaxRLTrainer",
            "SignalLedgerGRPOTrainer",
        ],
        "grouping_semantics": "trl_global_reward_order_grouped_by_num_generations",
    }


def _maxrl_trainer_transform(base_trainer_class, *, recorder: MaxRLRewardBatchRecorder):
    return make_practical_maxrl_trainer(base_trainer_class, recorder=recorder)


def run_maxrl(
    *,
    config_path: Path,
    pi0_dir: Path,
    output_dir: Path,
    mode: str,
    pilot_steps: int | None = None,
) -> dict:
    config_path = Path(config_path)
    config = load_config(config_path)
    validate_grpo_config(config)
    objective = build_maxrl_objective_metadata(config)

    return _run_controlled_grpo(
        config_path=config_path,
        pi0_dir=Path(pi0_dir),
        output_dir=Path(output_dir),
        mode=mode,
        pilot_steps=pilot_steps,
        recorder_factory=MaxRLRewardBatchRecorder,
        trainer_transform=_maxrl_trainer_transform,
        manifest_filename=MAXRL_MANIFEST_NAME,
        manifest_extra={"objective": objective},
        result_extra={"objective": objective},
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run exact-lineage controlled Qwen3 practical MaxRL on GSM8K."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pi0-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mode", choices=("pilot", "canonical"), required=True)
    parser.add_argument("--pilot-steps", type=int, default=None)
    args = parser.parse_args(argv)

    result = run_maxrl(
        config_path=args.config,
        pi0_dir=args.pi0_dir,
        output_dir=args.output_dir,
        mode=args.mode,
        pilot_steps=args.pilot_steps,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
