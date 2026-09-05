from __future__ import annotations

import argparse
import json
from pathlib import Path

from controlled_run.config import (
    GRPO_1XA100_REPLICATION_ALLOWED_SEEDS,
    build_grpo_1xa100_replication_config,
    load_config,
    validate_grpo_1xa100_replication_config,
    validate_grpo_1xa100_replication_runtime_batch,
)
from controlled_run.maxrl import MaxRLRewardBatchRecorder
from controlled_run.train_grpo import DEFAULT_CONFIG, _run_controlled_grpo
from controlled_run.train_grpo_replication import replication_metadata
from controlled_run.train_maxrl import (
    MAXRL_MANIFEST_NAME,
    _maxrl_trainer_transform,
    build_maxrl_objective_metadata,
)


def run_maxrl_replication(
    *,
    config_path: Path,
    pi0_dir: Path,
    output_dir: Path,
    seed: int,
    pilot_steps: int | None = None,
) -> dict:
    seed = int(seed)
    if seed not in GRPO_1XA100_REPLICATION_ALLOWED_SEEDS:
        raise ValueError(
            f"seed must be one of {GRPO_1XA100_REPLICATION_ALLOWED_SEEDS}"
        )

    canonical = load_config(Path(config_path))
    replication_config = build_grpo_1xa100_replication_config(
        canonical,
        seed=seed,
    )
    objective = build_maxrl_objective_metadata(
        replication_config,
        config_validator=validate_grpo_1xa100_replication_config,
    )
    metadata = replication_metadata(seed)
    mode = "pilot" if pilot_steps is not None else "replication"

    return _run_controlled_grpo(
        config_path=Path(config_path),
        pi0_dir=Path(pi0_dir),
        output_dir=Path(output_dir),
        mode=mode,
        pilot_steps=pilot_steps,
        recorder_factory=MaxRLRewardBatchRecorder,
        trainer_transform=_maxrl_trainer_transform,
        manifest_filename=MAXRL_MANIFEST_NAME,
        manifest_extra={"objective": objective, "replication": metadata},
        result_extra={"objective": objective, "replication": metadata},
        config_transform=lambda canonical_config: build_grpo_1xa100_replication_config(
            canonical_config,
            seed=seed,
        ),
        config_validator=validate_grpo_1xa100_replication_config,
        runtime_batch_validator=validate_grpo_1xa100_replication_runtime_batch,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen 1xA100 practical MaxRL replication lane."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pi0-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--seed",
        type=int,
        choices=GRPO_1XA100_REPLICATION_ALLOWED_SEEDS,
        required=True,
    )
    parser.add_argument(
        "--pilot-steps",
        type=int,
        default=None,
        help="20-500 step engineering pilot; omit for a full replication run.",
    )
    args = parser.parse_args(argv)

    result = run_maxrl_replication(
        config_path=args.config,
        pi0_dir=args.pi0_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        pilot_steps=args.pilot_steps,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
