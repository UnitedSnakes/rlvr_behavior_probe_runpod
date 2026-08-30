import json
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA = Path("data/controlled_run/generated/sft_10k_records.jsonl")

MODELS = [
    ("base", "Qwen/Qwen3-0.6B-Base"),
    (
        "A_current_56",
        "controlled_run_outputs/sft_terminal_ab/current_56/final",
    ),
    (
        "B_native_56",
        "controlled_run_outputs/sft_terminal_ab/native_eos_56/final",
    ),
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


def evaluate(name, path, examples):
    print("\n" + "=" * 90)
    print(name, path)

    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=torch.bfloat16,
    ).cuda().eval()

    im_probs = []
    eot_probs = []
    im_ranks = []
    eot_ranks = []

    with torch.inference_mode():
        for source_index, ids in examples:
            x = torch.tensor([ids], device="cuda")
            logits = model(input_ids=x).logits[0, -1].float()
            p = torch.softmax(logits, dim=-1)

            pim = p[IM_END].item()
            peot = p[EOT].item()

            rim = int((logits > logits[IM_END]).sum()) + 1
            reot = int((logits > logits[EOT]).sum()) + 1

            im_probs.append(pim)
            eot_probs.append(peot)
            im_ranks.append(rim)
            eot_ranks.append(reot)

            print(
                f"source={source_index:>6} "
                f"P(im_end)={pim:.6f} r={rim:>6} "
                f"P(eot)={peot:.6f} r={reot:>6}"
            )

    out = {
        "im_mean": float(np.mean(im_probs)),
        "im_median": float(np.median(im_probs)),
        "im_rank_median": float(np.median(im_ranks)),
        "eot_mean": float(np.mean(eot_probs)),
        "eot_median": float(np.median(eot_probs)),
        "eot_rank_median": float(np.median(eot_ranks)),
    }

    print("\nSUMMARY")
    for k, v in out.items():
        print(f"{k:16s}: {v}")

    del model
    torch.cuda.empty_cache()

    return out


def main():
    tok = AutoTokenizer.from_pretrained(
        "controlled_run_outputs/sft/pi_0"
    )
    examples = choose_examples(tok)

    results = {}
    for name, path in MODELS:
        results[name] = evaluate(name, path, examples)

    print("\n" + "=" * 90)
    print("FINAL COMPARISON")
    for name, r in results.items():
        print(
            f"{name:14s} "
            f"EOT median={r['eot_median']:.6f} "
            f"EOT rank={r['eot_rank_median']:.1f} | "
            f"IM_END median={r['im_median']:.6f} "
            f"IM_END rank={r['im_rank_median']:.1f}"
        )


if __name__ == "__main__":
    main()
