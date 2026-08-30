import json
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DATA = Path("data/controlled_run/generated/sft_10k_records.jsonl")
MODEL = "controlled_run_outputs/sft/pi_0"

IM_END = 151645
EOS = 151643
N = 20
MAX_LEN = 3000
SEED = 42


def to_ids(x):
    # Transformers 5.x compatibility
    if hasattr(x, "ids"):
        return list(x.ids)
    if isinstance(x, list) and x and hasattr(x[0], "ids"):
        return list(x[0].ids)
    if isinstance(x, dict):
        x = x["input_ids"]
    return list(x)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    rows = [json.loads(x) for x in DATA.open() if x.strip()]
    random.Random(SEED).shuffle(rows)

    chosen = []

    for r in rows:
        messages = r["prompt"] + r["completion"]

        full = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        ids = tok(full, add_special_tokens=False)["input_ids"]

        # 找模板附加的最后一个 <|im_end|>
        positions = [i for i, x in enumerate(ids) if x == IM_END]
        if not positions:
            continue

        endpos = positions[-1]

        # teacher forcing 输入只到 im_end 前一个 token
        prefix = ids[:endpos]

        if len(prefix) <= MAX_LEN:
            chosen.append((r, prefix))
        if len(chosen) == N:
            break

    print("examples:", len(chosen))

    results = []

    with torch.inference_mode():
        for i, (r, ids) in enumerate(chosen):
            x = torch.tensor([ids], device="cuda")

            logits = model(input_ids=x).logits[0, -1].float()
            probs = torch.softmax(logits, dim=-1)

            p_im = probs[IM_END].item()
            p_eos = probs[EOS].item()

            rank_im = int((logits > logits[IM_END]).sum().item()) + 1
            top = torch.topk(probs, 5)

            top5 = [
                (
                    int(tid),
                    tok.decode([int(tid)], skip_special_tokens=False),
                    float(p),
                )
                for p, tid in zip(top.values, top.indices)
            ]

            results.append(p_im)

            print("\n" + "=" * 90)
            print(
                f"[{i}] source_index={r.get('source_index')} "
                f"context_tokens={len(ids)}"
            )
            print(
                f"P(<|im_end|>)={p_im:.6f} "
                f"rank={rank_im} "
                f"P(<|endoftext|>)={p_eos:.6f}"
            )
            print("top5:")
            for tid, text, p in top5:
                print(f"  {tid:6d} {p:.6f} {text!r}")

    results.sort()

    if results:
        print("\n=== SUMMARY ===")
        print("min :", results[0])
        print("p25 :", results[len(results)//4])
        print("p50 :", results[len(results)//2])
        print("p75 :", results[(3*len(results))//4])
        print("max :", results[-1])


if __name__ == "__main__":
    main()
