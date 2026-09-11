import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import json
from collections import Counter
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

TS = 151667
TE = 151668
CAPS = [2048, 4096, 8192]
MAX_CAP = 8192

OUT = Path(
    "controlled_run_outputs/sft_corrected/"
    "forced_think_start_8192.json"
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
        gpu_memory_utilization=0.70,
        max_model_len=cfg["max_prompt_tokens"] + MAX_CAP,
        seed=cfg["seed"],
    )

    trajectories = []

    for idx, row in enumerate(rows):
        rendered = tok.apply_chat_template(
            row["prompt"],
            tokenize=False,
            add_generation_prompt=True,
        )

        # Single intervention:
        # make inference prefix exactly match SFT assistant start.
        rendered += "<think>\n"

        prompt_ids = tok(
            rendered,
            add_special_tokens=False,
        )["input_ids"]

        assert prompt_ids[-2:] == [TS, 198], prompt_ids[-10:]

        params = SamplingParams(
            n=cfg["num_generations"],
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            top_k=cfg["top_k"],
            repetition_penalty=cfg["repetition_penalty"],
            max_tokens=MAX_CAP,
            seed=cfg["seed"] * 100_000 + idx,
        )

        result = llm.generate(
            [TokensPrompt(prompt_token_ids=prompt_ids)],
            sampling_params=params,
            use_tqdm=False,
        )[0]

        for r, o in enumerate(result.outputs):
            text = o.text
            reward = gsm8k_binary_reward(
                [text],
                row["answer"],
            )[0]

            trajectories.append({
                "dataset_index": idx,
                "rollout": r,
                "finish_reason": o.finish_reason,
                "stop_reason": o.stop_reason,
                "token_ids": list(o.token_ids),
                "text": text,
                "correct": bool(reward),
            })

        print(
            f"[{idx}] "
            f"stop={sum(o.finish_reason != 'length' for o in result.outputs)}/16 "
            f"close_think={sum(TE in o.token_ids for o in result.outputs)}/16"
        )

    print("\n=== FORCED <think> START ===")

    for cap in CAPS:
        stop = 0
        closed = 0
        correct = 0
        closed_and_stop = 0
        stopped_open = 0

        for t in trajectories:
            ids = t["token_ids"]

            # If the 8192 generation naturally stopped before cap,
            # retain that stop. Otherwise virtually truncate at cap.
            natural_before_cap = (
                t["finish_reason"] != "length"
                and len(ids) <= cap
            )

            prefix = ids[:cap]

            if natural_before_cap:
                stop += 1

            has_close = TE in prefix
            if has_close:
                closed += 1

            # correctness under virtual cap
            text = tok.decode(
                prefix,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            reward = gsm8k_binary_reward(
                [text],
                rows[t["dataset_index"]]["answer"],
            )[0]

            correct += bool(reward)

            if natural_before_cap and has_close:
                closed_and_stop += 1

            if natural_before_cap and not has_close:
                stopped_open += 1

        n = len(trajectories)

        print(f"\ncap={cap}")
        print(f"  natural stop: {stop}/{n} = {100*stop/n:.2f}%")
        print(f"  closed think: {closed}/{n} = {100*closed/n:.2f}%")
        print(f"  correct:      {correct}/{n} = {100*correct/n:.2f}%")
        print(f"  stop+closed:  {closed_and_stop}/{n}")
        print(f"  stop+open:    {stopped_open}/{n}")

    first_tokens = Counter(
        t["token_ids"][0]
        for t in trajectories
        if t["token_ids"]
    )

    print("\n=== FIRST GENERATED TOKEN AFTER FORCED <think> ===")
    for tid, n in first_tokens.most_common(10):
        print(n, tid, repr(tok.decode([tid], skip_special_tokens=False)))

    OUT.write_text(
        json.dumps(
            {
                "intervention": "append <think>\\n to generation prompt",
                "trajectories": trajectories,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nsaved:", OUT)


if __name__ == "__main__":
    main()
