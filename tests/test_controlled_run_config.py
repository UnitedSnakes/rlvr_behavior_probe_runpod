from pathlib import Path

from controlled_run.config import load_config, validate_grpo_config, validate_sft_config
from controlled_run.constants import CONTROLLED_SYSTEM_PROMPT


ROOT = Path(__file__).resolve().parents[1]


def test_controlled_prompt_requires_boxed_final_answer():
    assert "Reason step by step" in CONTROLLED_SYSTEM_PROMPT
    assert r"\boxed{}" in CONTROLLED_SYSTEM_PROMPT
    assert "Qwen2.5" not in CONTROLLED_SYSTEM_PROMPT


def test_sft_config_matches_precommitted_recipe():
    cfg = load_config(ROOT / "controlled_run/configs/sft_qwen3_0_6b.yaml")
    validate_sft_config(cfg)

    assert cfg["model_name"] == "Qwen/Qwen3-0.6B-Base"
    assert cfg["num_train_epochs"] == 2
    assert cfg["max_length"] == 2048
    assert cfg["learning_rate"] == 2e-5
    assert cfg["global_batch_size"] == 64
    assert cfg["per_device_train_batch_size"] == 1
    assert cfg["gradient_accumulation_steps"] == 32


def test_grpo_config_matches_precommitted_behavior_study_recipe():
    cfg = load_config(ROOT / "controlled_run/configs/grpo_qwen3_0_6b.yaml")
    validate_grpo_config(cfg)

    assert cfg["num_generations"] == 16
    assert cfg["temperature"] == 0.8
    assert cfg["max_prompt_tokens"] == 512
    assert cfg["max_completion_length"] == 1024
    assert cfg["mask_truncated_completions"] is True
    assert cfg["vllm_max_model_length"] == 1536
    assert cfg["per_device_train_batch_size"] == 8
    assert cfg["gradient_accumulation_steps"] == 4
    assert cfg["generation_batch_size"] == 32
    assert cfg["generation_batch_size"] // cfg["num_generations"] == 2
    assert cfg["learning_rate"] == 1e-6
    assert cfg["beta"] == 0.0
    assert cfg["loss_type"] == "dapo"
    assert cfg["scale_rewards"] == "group"
    assert cfg["vllm_importance_sampling_correction"] is True
    assert cfg["vllm_importance_sampling_mode"] == "sequence_mask"
    assert cfg["vllm_importance_sampling_cap"] == 3.0
