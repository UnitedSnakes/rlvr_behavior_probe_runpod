import os

# Must be set before importing vLLM.
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from controlled_run.config import load_config
from controlled_run.data import build_gsm8k_rl_rows
from controlled_run.rewards import gsm8k_binary_reward
from probe.scoring import extract_numeric_answer


POLICY = "controlled_run_outputs/sft_corrected/pi_0"
CONFIG = "controlled_run/configs/grpo_qwen3_0_6b.yaml"
GSM8K_SHA = "740312add88f781978c0658806c59bc2815b9866"

CAPS = [2048, 4096, 8192]
MAX_CAP = max(CAPS)

OUTPUT = Path(
    "controlled_run_outputs/sft_corrected/"
    "horizon_curve_2048_4096_8192.json"
)


def percentile(xs, q):
    xs = sorted(xs)
    return xs[int(q * (len(xs) - 1))]


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

    prompts = []

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

        print(
            f"prompt {dataset_index}: "
            f"{len(prompt_ids)} tokens"
        )
        assert len(prompt_ids) <= cfg["max_prompt_tokens"]

        prompts.append(prompt_ids)

    max_model_len = cfg["max_prompt_tokens"] + MAX_CAP
    print("max_model_len:", max_model_len)

    llm = LLM(
        model=POLICY,
        tokenizer=POLICY,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.70,
        max_model_len=max_model_len,
        seed=cfg["seed"],
    )

    trajectories = []

    for dataset_index, (row, prompt_ids) in enumerate(
        zip(rows, prompts)
    ):
        params = SamplingParams(
            n=cfg["num_generations"],
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            top_k=cfg["top_k"],
            repetition_penalty=cfg["repetition_penalty"],
            max_tokens=MAX_CAP,
            seed=cfg["seed"] * 100_000 + dataset_index,
        )

        result = llm.generate(
            [TokensPrompt(prompt_token_ids=prompt_ids)],
            sampling_params=params,
            use_tqdm=False,
        )[0]

        stopped = sum(
            o.finish_reason != "length"
            for o in result.outputs
        )

        print(
            f"[{dataset_index}] "
            f"8192-stop={stopped}/16"
        )

        for rollout_index, o in enumerate(result.outputs):
            trajectories.append({
                "dataset_index": dataset_index,
                "rollout": rollout_index,
                "gold": row["answer"],
                "finish_reason_8192": o.finish_reason,
                "stop_reason_8192": o.stop_reason,
                "token_ids": list(o.token_ids),
                "text_8192": o.text,
            })

    summaries = {}
    virtual_rows = []

    for cap in CAPS:
        items = []

        for traj in trajectories:
            ids = traj["token_ids"]

            if len(ids) > cap:
                prefix_ids = ids[:cap]
                finish = "length"
            else:
                prefix_ids = ids
                finish = traj["finish_reason_8192"]

            text = tok.decode(
                prefix_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            reward = gsm8k_binary_reward(
                [text],
                traj["gold"],
            )[0]

            pred, raw_answer, method = extract_numeric_answer(text)

            item = {
                "cap": cap,
                "dataset_index": traj["dataset_index"],
                "rollout": traj["rollout"],
                "finish_reason": finish,
                "length": len(prefix_ids),
                "correct": bool(reward),
                "boxed": "\\boxed{" in text,
                "extraction_method": method,
                "prediction": pred,
                "raw_answer": raw_answer,
            }
            items.append(item)
            virtual_rows.append(item)

        n = len(items)
        clipped = sum(
            x["finish_reason"] == "length"
            for x in items
        )
        stopped = n - clipped
        boxed = sum(x["boxed"] for x in items)
        correct = sum(x["correct"] for x in items)
        stop_correct = sum(
            x["finish_reason"] != "length"
            and x["correct"]
            for x in items
        )
        clip_correct = sum(
            x["finish_reason"] == "length"
            and x["correct"]
            for x in items
        )

        lengths = [x["length"] for x in items]

        summary = {
            "n": n,
            "natural_stop": stopped,
            "natural_stop_pct": 100 * stopped / n,
            "length_clipped": clipped,
            "length_clipped_pct": 100 * clipped / n,
            "boxed": boxed,
            "boxed_pct": 100 * boxed / n,
            "correct": correct,
            "correct_pct": 100 * correct / n,
            "stop_correct": stop_correct,
            "stop_correct_pct": 100 * stop_correct / n,
            "clip_correct": clip_correct,
            "clip_correct_pct": 100 * clip_correct / n,
            "length_p50": statistics.median(lengths),
            "length_p75": percentile(lengths, 0.75),
            "length_p90": percentile(lengths, 0.90),
            "extraction_methods": dict(Counter(
                x["extraction_method"]
                for x in items
            )),
        }

        summaries[str(cap)] = summary

    # How many trajectories clipped at 2048 recover later?
    at = {}
    for x in virtual_rows:
        at[
            (
                x["cap"],
                x["dataset_index"],
                x["rollout"],
            )
        ] = x

    clipped_2048 = [
        key[1:]
        for key, x in at.items()
        if key[0] == 2048
        and x["finish_reason"] == "length"
    ]

    recovered = {}

    for cap in [4096, 8192]:
        stop = 0
        boxed = 0
        correct = 0

        for dataset_index, rollout in clipped_2048:
            x = at[(cap, dataset_index, rollout)]
            stop += x["finish_reason"] != "length"
            boxed += x["boxed"]
            correct += x["correct"]

        recovered[str(cap)] = {
            "among_2048_clipped": len(clipped_2048),
            "natural_stop": stop,
            "boxed": boxed,
            "correct": correct,
        }

    by_prompt = {}

    for cap in CAPS:
        d = defaultdict(list)
        for x in virtual_rows:
            if x["cap"] == cap:
                d[x["dataset_index"]].append(x)

        by_prompt[str(cap)] = {
            str(i): {
                "stop": sum(
                    x["finish_reason"] != "length"
                    for x in xs
                ),
                "boxed": sum(x["boxed"] for x in xs),
                "correct": sum(x["correct"] for x in xs),
            }
            for i, xs in sorted(d.items())
        }

    result = {
        "policy": POLICY,
        "gsm8k_sha": GSM8K_SHA,
        "determinism": {
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "engine_seed": cfg["seed"],
        },
        "sampling": {
            "num_generations": cfg["num_generations"],
            "temperature": cfg["temperature"],
            "top_p": cfg["top_p"],
            "top_k": cfg["top_k"],
            "repetition_penalty": cfg["repetition_penalty"],
            "question_seed_scheme":
                "seed*100000+dataset_index",
        },
        "summaries": summaries,
        "recovery_from_2048": recovered,
        "by_prompt": by_prompt,
        "virtual_rollouts": virtual_rows,
        "trajectories_8192": trajectories,
    }

    OUTPUT.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("\n=== MATCHED HORIZON CURVE ===")

    for cap in CAPS:
        s = summaries[str(cap)]
        print(
            f"\ncap={cap}"
            f"\n  stop:    {s['natural_stop']}/128 "
            f"= {s['natural_stop_pct']:.2f}%"
            f"\n  clipped: {s['length_clipped']}/128 "
            f"= {s['length_clipped_pct']:.2f}%"
            f"\n  boxed:   {s['boxed']}/128 "
            f"= {s['boxed_pct']:.2f}%"
            f"\n  correct: {s['correct']}/128 "
            f"= {s['correct_pct']:.2f}%"
            f"\n  stop+correct: "
            f"{s['stop_correct']}/128 "
            f"= {s['stop_correct_pct']:.2f}%"
            f"\n  p50/p75/p90: "
            f"{s['length_p50']} / "
            f"{s['length_p75']} / "
            f"{s['length_p90']}"
        )

    print("\n=== RECOVERY OF 2048-CLIPPED TRAJECTORIES ===")
    for cap in [4096, 8192]:
        r = recovered[str(cap)]
        print(
            f"by {cap}: "
            f"stop={r['natural_stop']}/"
            f"{r['among_2048_clipped']}, "
            f"boxed={r['boxed']}/"
            f"{r['among_2048_clipped']}, "
            f"correct={r['correct']}/"
            f"{r['among_2048_clipped']}"
        )

    print("\nsaved:", OUTPUT)


if __name__ == "__main__":
    main()
