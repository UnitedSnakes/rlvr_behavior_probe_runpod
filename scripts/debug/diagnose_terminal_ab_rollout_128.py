from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from controlled_run.data import build_gsm8k_rl_rows
from controlled_run.rewards import gsm8k_binary_reward

N_PROMPTS = 16
N_ROLLOUTS = 8
MAX_TOKENS = 2048

EOT = 151643
IM_END = 151645

OLD_INDICES = [
    204, 912, 1143, 1828,
    2006, 2253, 5238, 6074,
]


def choose_indices(n_rows):
    rng = random.Random(4242)

    pool = [
        i for i in range(n_rows)
        if i not in OLD_INDICES
    ]

    extra = rng.sample(
        pool,
        N_PROMPTS - len(OLD_INDICES),
    )

    return OLD_INDICES + extra


def percentile(xs, p):
    return float(np.percentile(xs, p)) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)

    revisions = json.loads(
        Path(
            "data/controlled_run/manifests/source_revisions.json"
        ).read_text()
    )

    gsm_sha = revisions["gsm8k_dataset"]["sha"]

    ds = load_dataset(
        "openai/gsm8k",
        "main",
        split="train",
        revision=gsm_sha,
    )

    rows = build_gsm8k_rl_rows(ds)
    indices = choose_indices(len(rows))

    print("indices:", indices)

    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=0.50,
        max_model_len=2560,
    )

    records = []

    for idx in indices:
        row = rows[idx]

        prompt_text = tok.apply_chat_template(
            row["prompt"],
            tokenize=False,
            add_generation_prompt=True,
        )

        prompt_ids = tok(
            prompt_text,
            add_special_tokens=False,
        )["input_ids"]

        params = SamplingParams(
            n=N_ROLLOUTS,
            temperature=0.8,
            top_p=0.95,
            top_k=0,
            repetition_penalty=1.0,
            max_tokens=MAX_TOKENS,
            seed=42 * 100000 + idx,
            skip_special_tokens=False,
        )

        result = llm.generate(
            [TokensPrompt(prompt_token_ids=prompt_ids)],
            params,
            use_tqdm=False,
        )[0]

        print(f"\n=== {args.name} dataset_index={idx} ===")

        for j, out in enumerate(result.outputs):
            ids = list(out.token_ids)
            text = out.text

            has_box = "\\boxed" in text

            reward = gsm8k_binary_reward(
                [text],
                row["answer"],
            )[0]

            eot_pos = (
                ids.index(EOT)
                if EOT in ids else None
            )

            im_end_pos = (
                ids.index(IM_END)
                if IM_END in ids else None
            )

            rec = {
                "name": args.name,
                "dataset_index": idx,
                "rollout": j,
                "length": len(ids),
                "finish_reason": out.finish_reason,
                "stop_reason": out.stop_reason,
                "has_box": has_box,
                "correct": bool(reward),
                "eot_pos": eot_pos,
                "im_end_pos": im_end_pos,
            }

            records.append(rec)

            print(
                f"{j}: "
                f"len={len(ids):4d} "
                f"finish={out.finish_reason:6s} "
                f"box={int(has_box)} "
                f"correct={int(bool(reward))}"
            )

    args.out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.out.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    total = len(records)

    stopped = [
        r for r in records
        if r["finish_reason"] == "stop"
    ]

    clipped = [
        r for r in records
        if r["finish_reason"] == "length"
    ]

    boxed = [
        r for r in records
        if r["has_box"]
    ]

    correct = [
        r for r in records
        if r["correct"]
    ]

    stop_box = [
        r for r in records
        if r["finish_reason"] == "stop"
        and r["has_box"]
    ]

    stop_correct = [
        r for r in records
        if r["finish_reason"] == "stop"
        and r["correct"]
    ]

    clipped_correct = [
        r for r in records
        if r["finish_reason"] == "length"
        and r["correct"]
    ]

    lengths = [r["length"] for r in records]

    print("\n" + "=" * 80)
    print("SUMMARY", args.name)
    print("rollouts:", total)

    print(
        "natural stop:",
        len(stopped),
        len(stopped) / total,
    )

    print(
        "clipped:",
        len(clipped),
        len(clipped) / total,
    )

    print(
        "boxed:",
        len(boxed),
        len(boxed) / total,
    )

    print(
        "correct:",
        len(correct),
        len(correct) / total,
    )

    print(
        "stop + boxed:",
        len(stop_box),
        len(stop_box) / total,
    )

    print(
        "stop + correct:",
        len(stop_correct),
        len(stop_correct) / total,
    )

    print(
        "clipped + correct:",
        len(clipped_correct),
        len(clipped_correct) / total,
    )

    print("length p50:", percentile(lengths, 50))
    print("length p75:", percentile(lengths, 75))
    print("length p90:", percentile(lengths, 90))

    print(
        "EOT appeared:",
        sum(r["eot_pos"] is not None for r in records),
    )

    print(
        "IM_END appeared:",
        sum(r["im_end_pos"] is not None for r in records),
    )


if __name__ == "__main__":
    main()
