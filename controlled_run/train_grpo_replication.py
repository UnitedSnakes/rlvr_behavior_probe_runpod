from __future__ import annotations

import argparse
import json
from pathlib import Path

from controlled_run.config import (
    GRPO_1XA100_REPLICATION_ALLOWED_SEEDS,
    build_grpo_1xa100_replication_config,
    validate_grpo_1xa100_replication_config,
    validate_grpo_1xa100_replication_runtime_batch,
)
from controlled_run.train_grpo import DEFAULT_CONFIG, _run_controlled_grpo


def replication_metadata(seed: int) -> dict:
    return {
        "suite": "a100_seed42_43_44_h1_h2_replication",
        "training_seed": int(seed),
        "hardware_contract": "1x NVIDIA A100 80GB; exactly one visible CUDA device",
        "execution_topology": "single_process_single_gpu",
        "changed_from_seed42_a40_canonical": [
            "training_seed",
            "world_size",
            "per_device_train_batch_size",
            "gpu_sku",
        ],
        "preserved_batch_geometry": {
            "global_optimizer_batch_size": 32,
            "generation_batch_size": 32,
            "steps_per_generation": 4,
            "num_generations": 16,
            "unique_prompts_per_generation_batch": 2,
        },
        "interpretation": (
            "Seeds 42-44 within this A100 suite are hardware/topology matched. "
            "The original A40 seed42 run remains the canonical discovery run; "
            "cross-platform seed42 is a hardware/topology bridge."
        ),
    }


def run_grpo_replication(
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

    mode = "pilot" if pilot_steps is not None else "replication"
    metadata = replication_metadata(seed)

    return _run_controlled_grpo(
        config_path=Path(config_path),
        pi0_dir=Path(pi0_dir),
        output_dir=Path(output_dir),
        mode=mode,
        pilot_steps=pilot_steps,
        manifest_extra={"replication": metadata},
        result_extra={"replication": metadata},
        config_transform=lambda canonical: build_grpo_1xa100_replication_config(
            canonical,
            seed=seed,
        ),
        config_validator=validate_grpo_1xa100_replication_config,
        runtime_batch_validator=validate_grpo_1xa100_replication_runtime_batch,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen 1xA100 GRPO replication lane."
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

    result = run_grpo_replication(
        config_path=args.config,
        pi0_dir=args.pi0_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        pilot_steps=args.pilot_steps,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
