from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from controlled_run.data import build_gsm8k_rl_rows

INDICES = [204, 912, 1143, 1828, 2006, 2253, 5238, 6074]

N = 4
MAX_TOKENS = 2048
IM_END = 151645
EOT = 151643


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

    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=0.50,
        max_model_len=2560,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)

    records = []

    for idx in INDICES:
        text = tok.apply_chat_template(
            rows[idx]["prompt"],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tok(
            text,
            add_special_tokens=False,
        )["input_ids"]

        params = SamplingParams(
            n=N,
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

            eot_pos = ids.index(EOT) if EOT in ids else None
            im_pos = ids.index(IM_END) if IM_END in ids else None

            box_char = out.text.find("\\boxed")
            if box_char >= 0:
                box_tok = len(
                    tok(
                        out.text[:box_char],
                        add_special_tokens=False,
                    )["input_ids"]
                )
            else:
                box_tok = None

            rec = {
                "name": args.name,
                "dataset_index": idx,
                "rollout": j,
                "length": len(ids),
                "finish_reason": out.finish_reason,
                "stop_reason": out.stop_reason,
                "eot_pos": eot_pos,
                "im_end_pos": im_pos,
                "boxed_token_approx": box_tok,
            }
            records.append(rec)

            print(
                f"{j}: len={len(ids):4d} "
                f"finish={out.finish_reason:6s} "
                f"eot={str(eot_pos):>5} "
                f"im_end={str(im_pos):>5} "
                f"boxed≈{box_tok}"
            )

    with args.out.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    stopped = [
        r for r in records
        if r["finish_reason"] == "stop"
    ]
    clipped = [
        r for r in records
        if r["finish_reason"] == "length"
    ]

    lengths = sorted(r["length"] for r in records)

    def percentile(xs, p):
        if not xs:
            return None
        return xs[round((len(xs) - 1) * p)]

    print("\n" + "=" * 80)
    print("SUMMARY", args.name)
    print("rollouts:", len(records))
    print("natural stop:", len(stopped))
    print("length clipped:", len(clipped))
    print("natural-stop ratio:", len(stopped) / len(records))
    print("clipped ratio:", len(clipped) / len(records))
    print("length p50:", percentile(lengths, .50))
    print("length p75:", percentile(lengths, .75))
    print("length p90:", percentile(lengths, .90))
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
