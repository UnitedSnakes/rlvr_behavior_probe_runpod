from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback

from controlled_run.checkpointing import PI0_MANIFEST_NAME, load_pi0_manifest
from controlled_run.config import load_config, validate_grpo_config
from controlled_run.data import assert_prompt_token_limit, build_gsm8k_rl_rows
from controlled_run.provenance import resolve_hf_revision, sha256_file, write_json
from controlled_run.rewards import gsm8k_binary_reward


DEFAULT_CONFIG = Path("controlled_run/configs/grpo_qwen3_0_6b.yaml")
DEFAULT_OUTPUT_DIR = Path("controlled_run_outputs/grpo")


def _load_grpo_classes():
    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as error:
        raise RuntimeError(
            "TRL is required for controlled GRPO training. Install "
            "controlled_run/requirements-a40.in in the isolated A40 environment."
        ) from error
    return GRPOConfig, GRPOTrainer, TrainerCallback


def progress_step_map(
    total_steps: int,
    percentages=range(5, 101, 5),
) -> dict[int, int]:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")

    step_to_percentage: dict[int, int] = {}
    for raw_percentage in percentages:
        percentage = int(raw_percentage)
        if not 0 < percentage <= 100:
            raise ValueError("snapshot percentages must lie in (0, 100]")
        step = max(1, min(total_steps, int(round(total_steps * percentage / 100))))
        step_to_percentage[step] = percentage

    return {
        percentage: step
        for step, percentage in sorted(step_to_percentage.items())
    }


def verify_pi0_for_grpo(pi0_dir: Path) -> dict:
    directory = Path(pi0_dir)
    manifest = load_pi0_manifest(directory)
    return {
        "manifest": manifest,
        "lineage_id": sha256_file(directory / PI0_MANIFEST_NAME),
    }


def build_grpo_arguments(
    config: dict,
    output_dir: Path,
    *,
    max_steps: int | None = None,
):
    validate_grpo_config(config)
    GRPOConfig, _, _ = _load_grpo_classes()

    kwargs = {
        "output_dir": str(Path(output_dir)),
        "num_train_epochs": config["num_train_epochs"],
        "num_generations": config["num_generations"],
        "temperature": config["temperature"],
        "top_p": config["top_p"],
        "top_k": config["top_k"],
        "repetition_penalty": config["repetition_penalty"],
        "max_completion_length": config["max_completion_length"],
        "mask_truncated_completions": config["mask_truncated_completions"],
        "learning_rate": config["learning_rate"],
        "lr_scheduler_type": config["lr_scheduler_type"],
        "warmup_ratio": config["warmup_ratio"],
        "optim": config["optim"],
        "max_grad_norm": config["max_grad_norm"],
        "bf16": config["bf16"],
        "gradient_checkpointing": config["gradient_checkpointing"],
        "use_vllm": config["use_vllm"],
        "vllm_mode": config["vllm_mode"],
        "vllm_gpu_memory_utilization": config["vllm_gpu_memory_utilization"],
        "vllm_max_model_length": config["vllm_max_model_length"],
        "beta": config["beta"],
        "epsilon": config["epsilon"],
        "num_iterations": config["num_iterations"],
        "loss_type": config["loss_type"],
        "scale_rewards": config["scale_rewards"],
        "per_device_train_batch_size": config["per_device_train_batch_size"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "generation_batch_size": config["generation_batch_size"],
        "vllm_importance_sampling_correction": config[
            "vllm_importance_sampling_correction"
        ],
        "vllm_importance_sampling_mode": config["vllm_importance_sampling_mode"],
        "vllm_importance_sampling_cap": config["vllm_importance_sampling_cap"],
        "save_strategy": "steps",
        "save_steps": 0.25,
        "report_to": "none",
        "seed": config["seed"],
        "data_seed": config["seed"],
    }
    if max_steps is not None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive when supplied")
        kwargs["max_steps"] = int(max_steps)
        kwargs["save_strategy"] = "no"

    return GRPOConfig(**kwargs)


class PolicySnapshotCallback(TrainerCallback):
    def __init__(
        self,
        output_dir: Path,
        tokenizer,
        pi0_lineage_id: str,
    ):
        self.output_dir = Path(output_dir)
        self.tokenizer = tokenizer
        self.pi0_lineage_id = str(pi0_lineage_id)
        self._saved_percentages: set[int] = set()
        self._schedule: dict[int, int] | None = None

    def _ensure_schedule(self, state) -> dict[int, int]:
        if self._schedule is None:
            self._schedule = progress_step_map(int(state.max_steps))
            write_json(
                self.output_dir / "policy_snapshot_schedule.json",
                {
                    "max_steps": int(state.max_steps),
                    "percentage_to_step": {
                        str(key): value for key, value in self._schedule.items()
                    },
                    "pi0_lineage_id": self.pi0_lineage_id,
                },
            )
        return self._schedule

    def on_train_begin(self, args, state, control, **kwargs):
        if getattr(state, "is_world_process_zero", False):
            self._ensure_schedule(state)
        return control

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if not getattr(state, "is_world_process_zero", False):
            return control
        if model is None:
            raise ValueError("PolicySnapshotCallback requires model at step end")

        schedule = self._ensure_schedule(state)
        current_step = int(state.global_step)
        for percentage, target_step in schedule.items():
            if target_step != current_step or percentage in self._saved_percentages:
                continue

            destination = self.output_dir / f"pi_{percentage:03d}"
            model.save_pretrained(str(destination))
            self.tokenizer.save_pretrained(str(destination))
            write_json(
                destination / "policy_metadata.json",
                {
                    "actual_step": current_step,
                    "target_percentage": percentage,
                    "pi0_lineage_id": self.pi0_lineage_id,
                },
            )
            self._saved_percentages.add(percentage)
        return control


def validate_pilot_steps(mode: str, pilot_steps: int | None) -> int | None:
    if mode not in {"pilot", "canonical"}:
        raise ValueError("mode must be 'pilot' or 'canonical'")
    if mode == "canonical":
        if pilot_steps is not None:
            raise ValueError("canonical mode does not allow a max-step override")
        return None
    if pilot_steps is None or not 20 <= int(pilot_steps) <= 50:
        raise ValueError("pilot mode requires --pilot-steps between 20 and 50")
    return int(pilot_steps)


def _write_grpo_manifest(
    destination: Path,
    *,
    mode: str,
    config: dict,
    pi0_verification: dict,
    gsm8k_sha: str,
    prompt_audit: dict,
    pilot_steps: int | None,
) -> None:
    write_json(
        destination / "grpo_run_manifest.json",
        {
            "mode": mode,
            "scientific_use": mode == "canonical",
            "pilot_steps": pilot_steps,
            "config": config,
            "pi0_manifest": pi0_verification["manifest"],
            "pi0_lineage_id": pi0_verification["lineage_id"],
            "gsm8k_dataset_sha": gsm8k_sha,
            "prompt_length_audit": prompt_audit,
            "core_diagnostics": [
                "reward",
                "frac_reward_zero_std",
                "completions/clipped_ratio",
                "completions/max_terminated_length",
                "entropy",
                "grad_norm",
                "clip_ratio",
                "vllm_sampling_logp_difference",
                "importance_sampling_ratio",
            ],
        },
    )


def run_grpo(
    *,
    config_path: Path,
    pi0_dir: Path,
    output_dir: Path,
    mode: str,
    pilot_steps: int | None = None,
) -> dict:
    pilot_steps = validate_pilot_steps(mode, pilot_steps)

    # Lineage verification is deliberately the first operation that touches pi_0.
    pi0_verification = verify_pi0_for_grpo(pi0_dir)

    config = load_config(Path(config_path))
    validate_grpo_config(config)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(Path(pi0_dir)))
    gsm8k_sha = resolve_hf_revision(
        config["dataset_name"],
        revision="main",
        repo_type="dataset",
    )
    raw_dataset = load_dataset(
        config["dataset_name"],
        revision=gsm8k_sha,
        split=config["dataset_split"],
    )
    rows = build_gsm8k_rl_rows(raw_dataset)
    prompt_audit = assert_prompt_token_limit(
        rows,
        tokenizer,
        max_tokens=config["max_prompt_tokens"],
    )
    write_json(destination / "prompt_length_audit.json", prompt_audit)
    _write_grpo_manifest(
        destination,
        mode=mode,
        config=config,
        pi0_verification=pi0_verification,
        gsm8k_sha=gsm8k_sha,
        prompt_audit=prompt_audit,
        pilot_steps=pilot_steps,
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(Path(pi0_dir)),
        dtype=torch.bfloat16,
        attn_implementation=config["attn_implementation"],
    )
    train_dataset = Dataset.from_list(rows)
    trainer_output = destination / "trainer"
    args = build_grpo_arguments(
        config,
        trainer_output,
        max_steps=pilot_steps,
    )
    _, GRPOTrainer, _ = _load_grpo_classes()

    callbacks = []
    if mode == "canonical":
        callbacks.append(
            PolicySnapshotCallback(
                output_dir=destination,
                tokenizer=tokenizer,
                pi0_lineage_id=pi0_verification["lineage_id"],
            )
        )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=gsm8k_binary_reward,
        args=args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    trainer.train()

    return {
        "mode": mode,
        "scientific_use": mode == "canonical",
        "pilot_steps": pilot_steps,
        "pi0_lineage_id": pi0_verification["lineage_id"],
        "gsm8k_dataset_sha": gsm8k_sha,
        "prompt_length_audit": prompt_audit,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run exact-lineage controlled Qwen3 GRPO on GSM8K."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--pi0-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mode", choices=("pilot", "canonical"), required=True)
    parser.add_argument("--pilot-steps", type=int, default=None)
    args = parser.parse_args(argv)

    result = run_grpo(
        config_path=args.config,
        pi0_dir=args.pi0_dir,
        output_dir=args.output_dir,
        mode=args.mode,
        pilot_steps=args.pilot_steps,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
