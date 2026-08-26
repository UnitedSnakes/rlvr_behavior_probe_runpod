# Exact system prompt stated on the SFT model card.
SYSTEM_PROMPT = (
    "You are a helpful assistant. Think step by step before responding to the user's query. "
    "Your thought process should be enclosed between <think> and </think> tags. "
    "Once your thought process is complete, write a response which should end in the final "
    "answer enclosed in \\boxed{}."
)

# Use one canonical tokenizer/chat template for BOTH checkpoints and backends.
TOKENIZER_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
TOKENIZER_REVISION = "main"
