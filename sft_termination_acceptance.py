import json
import statistics
from pathlib import Path

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
OUTPUT = Path(
    "controlled_run_outputs/sft_corrected/"
    "termination_acceptance_128.json"
)


def main():
    cfg = load_config(CONFIG)
    tok = AutoTokenizer.from_pretrained(POLICY)

    raw = load_dataset(
        "openai/gsm8k",
        "main",
        revision=GSM8K_SHA,
        split="train[:8]",
    )
    rows = build_gsm8k_rl_rows(raw)

    llm = LLM(
        model=POLICY,
        tokenizer=POLICY,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.50,
        max_model_len=cfg["vllm_max_model_length"],
    )

    all_items = []

    for dataset_index, row in enumerate(rows):
        rendered = tok.apply_chat_template(
            row["prompt"],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tok(
            rendered,
            add_special_tokens=False,
        )["input_ids"]

        params = SamplingParams(
            n=cfg["num_generations"],
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            top_k=cfg["top_k"],
            repetition_penalty=cfg["repetition_penalty"],
            max_tokens=cfg["max_completion_length"],
            seed=cfg["seed"] * 100_000 + dataset_index,
        )

        result = llm.generate(
            [TokensPrompt(prompt_token_ids=prompt_ids)],
            sampling_params=params,
            use_tqdm=False,
        )[0]

        texts = [o.text for o in result.outputs]
        rewards = gsm8k_binary_reward(texts, row["answer"])

        for rollout_idx, (o, reward) in enumerate(
            zip(result.outputs, rewards)
        ):
            all_items.append({
                "dataset_index": dataset_index,
                "rollout": rollout_idx,
                "finish_reason": o.finish_reason,
                "stop_reason": o.stop_reason,
                "length": len(o.token_ids),
                "correct": bool(reward),
                "boxed": "\\boxed{" in o.text,
            })

        print(
            f"[{dataset_index}] "
            f"correct={sum(rewards):.0f}/16 "
            f"stopped="
            f"{sum(o.finish_reason != 'length' for o in result.outputs)}/16"
        )

    n = len(all_items)
    lengths = [x["length"] for x in all_items]

    clipped = sum(
        x["finish_reason"] == "length"
        for x in all_items
    )
    stopped = n - clipped
    correct = sum(x["correct"] for x in all_items)
    boxed = sum(x["boxed"] for x in all_items)
    stop_correct = sum(
        x["finish_reason"] != "length" and x["correct"]
        for x in all_items
    )
    clip_correct = sum(
        x["finish_reason"] == "length" and x["correct"]
        for x in all_items
    )

    def pct(x):
        return 100.0 * x / n

    sorted_lengths = sorted(lengths)

    summary = {
        "n": n,
        "natural_stop": stopped,
        "natural_stop_pct": pct(stopped),
        "length_clipped": clipped,
        "length_clipped_pct": pct(clipped),
        "boxed": boxed,
        "boxed_pct": pct(boxed),
        "correct": correct,
        "correct_pct": pct(correct),
        "stop_correct": stop_correct,
        "stop_correct_pct": pct(stop_correct),
        "clip_correct": clip_correct,
        "clip_correct_pct": pct(clip_correct),
        "length_p50": statistics.median(lengths),
        "length_p75": sorted_lengths[int(0.75 * (n - 1))],
        "length_p90": sorted_lengths[int(0.90 * (n - 1))],
    }

    OUTPUT.write_text(
        json.dumps(
            {
                "policy": POLICY,
                "gsm8k_sha": GSM8K_SHA,
                "sampling": {
                    "num_generations": cfg["num_generations"],
                    "temperature": cfg["temperature"],
                    "top_p": cfg["top_p"],
                    "top_k": cfg["top_k"],
                    "repetition_penalty": cfg["repetition_penalty"],
                    "max_completion_length":
                        cfg["max_completion_length"],
                    "seed": cfg["seed"],
                },
                "summary": summary,
                "rollouts": all_items,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== FINAL ACCEPTANCE ===")
    print("n:", n)
    print(
        f"natural stop:   {stopped}/{n} "
        f"= {pct(stopped):.2f}%"
    )
    print(
        f"length clipped: {clipped}/{n} "
        f"= {pct(clipped):.2f}%"
    )
    print(
        f"boxed:          {boxed}/{n} "
        f"= {pct(boxed):.2f}%"
    )
    print(
        f"correct:        {correct}/{n} "
        f"= {pct(correct):.2f}%"
    )
    print(
        f"stop+correct:   {stop_correct}/{n} "
        f"= {pct(stop_correct):.2f}%"
    )
    print(
        f"clip+correct:   {clip_correct}/{n} "
        f"= {pct(clip_correct):.2f}%"
    )
    print(
        "length p50/p75/p90:",
        summary["length_p50"],
        summary["length_p75"],
        summary["length_p90"],
    )
    print("saved:", OUTPUT)


if __name__ == "__main__":
    main()
