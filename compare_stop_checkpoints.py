import json
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DATA = Path("data/controlled_run/generated/sft_10k_records.jsonl")

MODELS = [
    ("base", "Qwen/Qwen3-0.6B-Base"),
    ("epoch1", "controlled_run_outputs/sft/trainer/checkpoint-56"),
    ("epoch2", "controlled_run_outputs/sft/trainer/checkpoint-112"),
    ("pi0", "controlled_run_outputs/sft/pi_0"),
]

IM_END = 151645
EOT = 151643
N = 20
MAX_LEN = 3000
SEED = 42


def choose_examples(tok):
    rows = [json.loads(x) for x in DATA.open() if x.strip()]
    random.Random(SEED).shuffle(rows)

    chosen = []

    for r in rows:
        full = tok.apply_chat_template(
            r["prompt"] + r["completion"],
            tokenize=False,
            add_generation_prompt=False,
        )
        ids = tok(full, add_special_tokens=False)["input_ids"]

        positions = [i for i, x in enumerate(ids) if x == IM_END]
        if not positions:
            continue

        prefix = ids[:positions[-1]]

        if len(prefix) <= MAX_LEN:
            chosen.append((r.get("source_index"), prefix))

        if len(chosen) == N:
            break

    return chosen


def pct(xs, p):
    return float(np.percentile(xs, p))


def evaluate(name, path, examples):
    print(f"\n{'='*80}")
    print(name, path)

    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=torch.bfloat16,
    ).cuda().eval()

    probs = []
    ranks = []

    with torch.inference_mode():
        for source_index, ids in examples:
            x = torch.tensor([ids], device="cuda")

            logits = model(input_ids=x).logits[0, -1].float()
            p = torch.softmax(logits, dim=-1)

            prob = p[IM_END].item()
            rank = int((logits > logits[IM_END]).sum().item()) + 1

            eot_prob = p[EOT].item()
            eot_rank = int((logits > logits[EOT]).sum().item()) + 1

            probs.append(prob)
            ranks.append(rank)

            print(
                f"source={source_index:>6} "
                f"P(im_end)={prob:.6f} rank={rank:>6} "
                f"P(eot)={eot_prob:.6f} rank={eot_rank:>6}"
            )

    print("\nSUMMARY")
    print("mean P :", float(np.mean(probs)))
    print("p25 P  :", pct(probs, 25))
    print("p50 P  :", pct(probs, 50))
    print("p75 P  :", pct(probs, 75))
    print("mean rank:", float(np.mean(ranks)))
    print("median rank:", float(np.median(ranks)))

    del model
    torch.cuda.empty_cache()

    return probs


def main():
    tok = AutoTokenizer.from_pretrained(
        "controlled_run_outputs/sft/pi_0"
    )

    examples = choose_examples(tok)
    print("examples:", len(examples))
    print("indices:", [x[0] for x in examples])

    all_results = {}

    for name, path in MODELS:
        all_results[name] = evaluate(name, path, examples)

    print("\n" + "="*80)
    print("PAIRED SUMMARY")

    for name, xs in all_results.items():
        print(
            f"{name:8s} "
            f"mean={np.mean(xs):.6f} "
            f"median={np.median(xs):.6f}"
        )


if __name__ == "__main__":
    main()
