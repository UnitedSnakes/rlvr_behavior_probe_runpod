import json

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from controlled_run.config import load_config
from controlled_run.data import build_gsm8k_rl_rows
from controlled_run.rewards import gsm8k_binary_reward


POLICY = "controlled_run_outputs/sft_corrected/pi_0"
CONFIG = "controlled_run/configs/grpo_qwen3_0_6b.yaml"
GSM8K_SHA = "740312add88f781978c0658806c59bc2815b9866"
IDX = 7


def main():
    cfg = load_config(CONFIG)
    tok = AutoTokenizer.from_pretrained(POLICY)

    raw = load_dataset(
        "openai/gsm8k",
        "main",
        revision=GSM8K_SHA,
        split="train[:8]",
    )
    row = build_gsm8k_rl_rows(raw)[IDX]

    rendered = tok.apply_chat_template(
        row["prompt"],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tok(
        rendered,
        add_special_tokens=False,
    )["input_ids"]

    llm = LLM(
        model=POLICY,
        tokenizer=POLICY,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.50,
        max_model_len=cfg["vllm_max_model_length"],
    )

    params = SamplingParams(
        n=16,
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        top_k=cfg["top_k"],
        repetition_penalty=cfg["repetition_penalty"],
        max_tokens=cfg["max_completion_length"],
        seed=cfg["seed"] * 100_000 + IDX,
    )

    result = llm.generate(
        [TokensPrompt(prompt_token_ids=prompt_ids)],
        sampling_params=params,
        use_tqdm=False,
    )[0]

    texts = [o.text for o in result.outputs]
    rewards = gsm8k_binary_reward(texts, row["answer"])

    records = []

    print("QUESTION:", row["prompt"][-1]["content"])
    print("GOLD:", row["answer"])

    for i, (o, reward) in enumerate(zip(result.outputs, rewards)):
        rec = {
            "rollout": i,
            "finish_reason": o.finish_reason,
            "stop_reason": o.stop_reason,
            "length": len(o.token_ids),
            "correct": bool(reward),
            "boxed": "\\boxed{" in o.text,
            "text": o.text,
        }
        records.append(rec)

        print("\n" + "=" * 100)
        print(
            f"[{i}] finish={o.finish_reason} "
            f"len={len(o.token_ids)} "
            f"boxed={rec['boxed']} correct={bool(reward)}"
        )
        print("--- TAIL ---")
        print(o.text[-1200:])

    with open(
        "controlled_run_outputs/sft_corrected/"
        "prompt7_termination_diagnostic.json",
        "w",
    ) as f:
        json.dump(
            {
                "question": row["prompt"][-1]["content"],
                "gold": row["answer"],
                "rollouts": records,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
