from __future__ import annotations


CONTROLLED_SYSTEM_PROMPT = (
    "You are a helpful assistant. Reason step by step and put your final answer "
    "within \\boxed{}."
)

BASE_MODEL = "Qwen/Qwen3-0.6B-Base"
SFT_DATASET = "open-r1/OpenR1-Math-220k"
GSM8K_DATASET = "openai/gsm8k"
SEED = 42
