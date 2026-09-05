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
    "assistant_terminal_token": "<|endoftext|>",
    "assistant_terminal_token_id": 151643,
    "seed": SEED,
}

GRPO_INVARIANTS = {
    "dataset_name": GSM8K_DATASET,
    "dataset_config": "main",
    "dataset_split": "train",
    "canonical_world_size": 2,
    "global_optimizer_batch_size": 32,
    "num_train_epochs": 1,
    "reward": "binary_terminated_final_answer_correctness",
    "num_generations": 16,
    "temperature": 0.8,
    "top_p": 1.0,
    "top_k": 0,
    "repetition_penalty": 1.0,
    "max_prompt_tokens": 512,
    "max_completion_length": 2048,
    "mask_truncated_completions": False,
    "vllm_max_model_length": 2560,
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
    "scale_rewards": "group",
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "generation_batch_size": 32,
    "vllm_importance_sampling_correction": True,
    "vllm_importance_sampling_mode": "token_truncate",
    "vllm_importance_sampling_cap": 3.0,
    "seed": SEED,
}


GRPO_1XA100_REPLICATION_ALLOWED_SEEDS = (42, 43, 44)
GRPO_1XA100_REPLICATION_OVERRIDES = {
    "canonical_world_size": 1,
    "per_device_train_batch_size": 8,
    "gradient_checkpointing": False,
}


GRPO_A40_REPLICATION_ALLOWED_SEEDS = (43, 44)


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


def _validate_grpo_structure(config: dict, *, label: str) -> None:
    if config["vllm_max_model_length"] != (
        config["max_prompt_tokens"] + config["max_completion_length"]
    ):
        raise ValueError(
            f"{label} vllm_max_model_length must equal max_prompt_tokens + "
            "max_completion_length"
        )

    if config["generation_batch_size"] % config["num_generations"] != 0:
        raise ValueError(
            f"{label} generation_batch_size must be divisible by num_generations"
        )

    if config["generation_batch_size"] // config["num_generations"] < 2:
        raise ValueError(
            "Controlled behavior study requires at least two independent prompts "
            "per generation batch"
        )


def validate_grpo_config(config: dict) -> None:
    _validate_exact(config, GRPO_INVARIANTS, "GRPO")
    _validate_grpo_structure(config, label="GRPO")


def build_grpo_a40_replication_config(
    canonical_config: dict,
    *,
    seed: int,
) -> dict:
    """Derive a matched A40 replication config; only seed may differ."""
    validate_grpo_config(canonical_config)
    seed = int(seed)
    if seed not in GRPO_A40_REPLICATION_ALLOWED_SEEDS:
        raise ValueError(
            "A40 replication seed must be one of "
            f"{GRPO_A40_REPLICATION_ALLOWED_SEEDS}; got {seed}"
        )
    config = dict(canonical_config)
    config["seed"] = seed
    validate_grpo_a40_replication_config(config)
    return config


def validate_grpo_a40_replication_config(config: dict) -> None:
    seed = config.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("A40 replication seed must be an integer")
    if seed not in GRPO_A40_REPLICATION_ALLOWED_SEEDS:
        raise ValueError(
            "A40 replication seed must be one of "
            f"{GRPO_A40_REPLICATION_ALLOWED_SEEDS}; got {seed}"
        )
    expected = dict(GRPO_INVARIANTS)
    expected["seed"] = seed
    _validate_exact(config, expected, "A40 GRPO replication")
    _validate_grpo_structure(config, label="A40 GRPO replication")


def validate_grpo_a40_replication_runtime_batch(
    config: dict,
    *,
    world_size: int,
) -> dict:
    validate_grpo_a40_replication_config(config)
    return _validate_grpo_runtime_geometry(
        config,
        world_size=world_size,
        expected_world_size=2,
        label="A40 GRPO replication",
    )


def build_grpo_1xa100_replication_config(
    canonical_config: dict,
    *,
    seed: int,
) -> dict:
    validate_grpo_config(canonical_config)
    seed = int(seed)
    if seed not in GRPO_1XA100_REPLICATION_ALLOWED_SEEDS:
        raise ValueError(
            "1xA100 replication seed must be one of "
            f"{GRPO_1XA100_REPLICATION_ALLOWED_SEEDS}; got {seed}"
        )
    config = dict(canonical_config)
    config.update(GRPO_1XA100_REPLICATION_OVERRIDES)
    config["seed"] = seed
    validate_grpo_1xa100_replication_config(config)
    return config


def validate_grpo_1xa100_replication_config(config: dict) -> None:
    seed = config.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("1xA100 replication seed must be an integer")
    if seed not in GRPO_1XA100_REPLICATION_ALLOWED_SEEDS:
        raise ValueError(
            "1xA100 replication seed must be one of "
            f"{GRPO_1XA100_REPLICATION_ALLOWED_SEEDS}; got {seed}"
        )
    expected = dict(GRPO_INVARIANTS)
    expected.update(GRPO_1XA100_REPLICATION_OVERRIDES)
    expected["seed"] = seed
    _validate_exact(config, expected, "1xA100 GRPO replication")
    _validate_grpo_structure(config, label="1xA100 GRPO replication")


def _validate_grpo_runtime_geometry(
    config: dict,
    *,
    world_size: int,
    expected_world_size: int,
    label: str,
) -> dict:
    if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size <= 0:
        raise ValueError(f"{label} world_size must be a positive integer")
    if world_size != expected_world_size:
        raise ValueError(
            f"{label} requires WORLD_SIZE={expected_world_size}; got {world_size}"
        )

    per_device = int(config["per_device_train_batch_size"])
    grad_accum = int(config["gradient_accumulation_steps"])
    global_optimizer_batch = per_device * world_size * grad_accum
    target_global_optimizer_batch = int(config["global_optimizer_batch_size"])
    if global_optimizer_batch != target_global_optimizer_batch:
        raise ValueError(
            f"{label} global optimizer batch mismatch; got "
            f"{per_device} x {world_size} x {grad_accum} = "
            f"{global_optimizer_batch}, expected {target_global_optimizer_batch}"
        )

    per_step_global_batch = per_device * world_size
    generation_batch = int(config["generation_batch_size"])
    if generation_batch % per_step_global_batch != 0:
        raise ValueError(
            f"{label} generation_batch_size must be divisible by the global "
            "per-step training batch"
        )
    steps_per_generation = generation_batch // per_step_global_batch

    num_generations = int(config["num_generations"])
    if generation_batch % num_generations != 0:
        raise ValueError(
            f"{label} generation_batch_size must be divisible by num_generations"
        )
    unique_prompts = generation_batch // num_generations

    if steps_per_generation != 4:
        raise ValueError(
            f"{label} requires steps_per_generation=4; got {steps_per_generation}"
        )
    if unique_prompts != 2:
        raise ValueError(
            f"{label} requires exactly 2 unique prompts per generation batch; "
            f"got {unique_prompts}"
        )

    return {
        "world_size": world_size,
        "per_device_train_batch_size": per_device,
        "gradient_accumulation_steps": grad_accum,
        "global_optimizer_batch_size": global_optimizer_batch,
        "generation_batch_size": generation_batch,
        "steps_per_generation": steps_per_generation,
        "num_generations": num_generations,
        "unique_prompts_per_generation_batch": unique_prompts,
    }


def validate_grpo_runtime_batch(config: dict, *, world_size: int) -> dict:
    validate_grpo_config(config)
    if world_size != 2:
        raise ValueError(
            "Controlled GRPO requires exactly 2 GPUs; "
            f"WORLD_SIZE={world_size}"
        )
    return _validate_grpo_runtime_geometry(
        config,
        world_size=world_size,
        expected_world_size=2,
        label="Controlled GRPO",
    )


def validate_grpo_1xa100_replication_runtime_batch(
    config: dict,
    *,
    world_size: int,
) -> dict:
    validate_grpo_1xa100_replication_config(config)
    return _validate_grpo_runtime_geometry(
        config,
        world_size=world_size,
        expected_world_size=1,
        label="1xA100 GRPO replication",
    )
