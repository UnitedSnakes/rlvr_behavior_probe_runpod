import json
import random
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from controlled_run.data import build_gsm8k_rl_rows

MODEL = "controlled_run_outputs/sft/pi_0"
MAX_COMPLETION = 8192
MAX_MODEL_LEN = 8704
N_PROMPTS = 8
N_ROLLOUTS = 4
SEED = 42

OUT = Path("controlled_run_outputs/pi0_length_audit_8192.jsonl")


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)

    ds = load_dataset("openai/gsm8k", "main", split="train")
    rows = build_gsm8k_rl_rows(ds)

    rng = random.Random(SEED)
    indices = sorted(rng.sample(range(len(rows)), N_PROMPTS))
    print("dataset indices:", indices)

    llm = LLM(
        model=MODEL,
        tokenizer=MODEL,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.50,
        max_model_len=MAX_MODEL_LEN,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    records = []

    with OUT.open("w") as f:
        for idx in indices:
            prompt_text = tok.apply_chat_template(
                rows[idx]["prompt"],
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
                max_tokens=MAX_COMPLETION,
                seed=SEED * 100000 + idx,
                skip_special_tokens=False,
            )

            result = llm.generate(
                [TokensPrompt(prompt_token_ids=prompt_ids)],
                params,
                use_tqdm=False,
            )[0]

            print(f"\n=== dataset_index={idx} ===")

            for j, out in enumerate(result.outputs):
                ids = list(out.token_ids)
                text = out.text

                box_char = text.find("\\boxed")
                if box_char >= 0:
                    box_tok = len(
                        tok(
                            text[:box_char],
                            add_special_tokens=False,
                        )["input_ids"]
                    )
                else:
                    box_tok = None

                rec = {
                    "dataset_index": idx,
                    "rollout": j,
                    "length": len(ids),
                    "finish_reason": out.finish_reason,
                    "stop_reason": out.stop_reason,
                    "boxed_token_approx": box_tok,
                }
                records.append(rec)
                f.write(json.dumps(rec) + "\n")

                print(
                    f"{j}: len={len(ids):4d} "
                    f"finish={out.finish_reason} "
                    f"boxed≈{box_tok}"
                )

    lengths = sorted(x["length"] for x in records)
    stopped = [x for x in records if x["finish_reason"] == "stop"]
    clipped = [x for x in records if x["finish_reason"] == "length"]
    boxed = [x for x in records if x["boxed_token_approx"] is not None]

    def pct(xs, p):
        if not xs:
            return None
        xs = sorted(xs)
        return xs[round((len(xs) - 1) * p)]

    print("\n=== SUMMARY ===")
    print("rollouts:", len(records))
    print("natural stop:", len(stopped))
    print("length clipped:", len(clipped))
    print("clipped ratio:", len(clipped) / len(records))
    print("length p50:", pct(lengths, .50))
    print("length p75:", pct(lengths, .75))
    print("length p90:", pct(lengths, .90))
    print("boxed present:", len(boxed))

    if boxed:
        bp = [x["boxed_token_approx"] for x in boxed]
        print("boxed position p50:", pct(bp, .50))
        print("boxed position p90:", pct(bp, .90))


if __name__ == "__main__":
    main()
