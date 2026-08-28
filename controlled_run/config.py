from __future__ import annotations

from pathlib import Path

import yaml

from controlled_run.constants import BASE_MODEL, GSM8K_DATASET, SEED, SFT_DATASET


SFT_INVARIANTS = {
    "model_name": BASE_MODEL,
    "dataset_name": SFT_DATASET,
    "dataset_config": "default",
    "subset_size": 10_000,
    "num_train_epochs": 2,
    "max_length": 16384,
    "bf16": True,
    "attn_implementation": "flash_attention_2",
    "gradient_checkpointing": True,
    "packing": True,
    "packing_strategy": "bfd",
    "completion_only_loss": True,
    "optim": "adamw_torch_fused",
    "learning_rate": 2e-5,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    "weight_decay": 0.01,
    "global_batch_size": 64,
    "seed": SEED,
}

GRPO_INVARIANTS = {
    "dataset_name": GSM8K_DATASET,
    "dataset_split": "train",
    "num_train_epochs": 1,
    "reward": "binary_final_answer_correctness",
    "num_generations": 16,
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 0,
    "repetition_penalty": 1.0,
    "max_prompt_tokens": 512,
    "max_completion_length": 1024,
    "mask_truncated_completions": True,
    "vllm_max_model_length": 1536,
    "learning_rate": 1e-6,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.10,
    "optim": "adamw_torch_fused",
    "max_grad_norm": 1.0,
    "bf16": True,
    "attn_implementation": "flash_attention_2",
    "gradient_checkpointing": True,
    "use_vllm": True,
    "vllm_mode": "colocate",
    "vllm_gpu_memory_utilization": 0.30,
    "beta": 0.0,
    "epsilon": 0.2,
    "num_iterations": 1,
    "loss_type": "dapo",
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "generation_batch_size": 32,
    "vllm_importance_sampling_correction": True,
    "vllm_importance_sampling_mode": "sequence_mask",
    "vllm_importance_sampling_cap": 3.0,
    "seed": SEED,
}


def load_config(path: Path) -> dict:
    path = Path(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")
    return payload


def _validate_exact(config: dict, expected: dict, label: str) -> None:
    for key, value in expected.items():
        if key not in config:
            raise ValueError(f"{label} config is missing required field {key!r}")
        if config[key] != value:
            raise ValueError(
                f"{label} config field {key!r} must be {value!r}, "
                f"got {config[key]!r}"
            )


def validate_sft_config(config: dict) -> None:
    _validate_exact(config, SFT_INVARIANTS, "SFT")
    for key in ("per_device_train_batch_size", "gradient_accumulation_steps"):
        value = config.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"SFT config field {key!r} must be a positive integer")


def validate_sft_runtime_batch(
    config: dict,
    *,
    world_size: int,
    canonical: bool,
) -> dict:
    validate_sft_config(config)
    if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size <= 0:
        raise ValueError("SFT world_size must be a positive integer")

    per_device = int(config["per_device_train_batch_size"])
    grad_accum = int(config["gradient_accumulation_steps"])
    effective = per_device * world_size * grad_accum
    target = int(config["global_batch_size"])

    if canonical and world_size != 2:
        raise ValueError(
            "Canonical controlled SFT requires exactly 2 GPUs; "
            f"WORLD_SIZE={world_size}"
        )
    if canonical and effective != target:
        raise ValueError(
            "Canonical controlled SFT requires global batch size 64; "
            f"got {per_device} x {world_size} x {grad_accum} = {effective}"
        )

    return {
        "world_size": world_size,
        "per_device_train_batch_size": per_device,
        "gradient_accumulation_steps": grad_accum,
        "global_batch_size": effective,
    }


def validate_grpo_config(config: dict) -> None:
    _validate_exact(config, GRPO_INVARIANTS, "GRPO")

    if config.get("scale_rewards") not in {True, False, "group", "batch", "none"}:
        raise ValueError(
            "GRPO scale_rewards must be one of True, False, 'group', 'batch', or 'none'"
        )

    if config["vllm_max_model_length"] != (
        config["max_prompt_tokens"] + config["max_completion_length"]
    ):
        raise ValueError(
            "GRPO vllm_max_model_length must equal max_prompt_tokens + "
            "max_completion_length"
        )

    if config["generation_batch_size"] % config["num_generations"] != 0:
        raise ValueError(
            "GRPO generation_batch_size must be divisible by num_generations"
        )

    if config["generation_batch_size"] // config["num_generations"] < 2:
        raise ValueError(
            "Controlled behavior study requires at least two independent prompts "
            "per generation batch"
        )
