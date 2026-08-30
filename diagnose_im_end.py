from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from controlled_run.data import build_gsm8k_rl_rows

MODEL = "controlled_run_outputs/sft/pi_0"
IM_END = 151645
EOS = 151643


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)

    ds = load_dataset("openai/gsm8k", "main", split="train")
    rows = build_gsm8k_rl_rows(ds)

    llm = LLM(
        model=MODEL,
        tokenizer=MODEL,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.50,
        max_model_len=2560,
    )

    for idx in [0, 1]:
        text = tok.apply_chat_template(
            rows[idx]["prompt"],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tok(text, add_special_tokens=False)["input_ids"]

        params = SamplingParams(
            n=16,
            temperature=0.8,
            top_p=0.95,
            top_k=0,
            repetition_penalty=1.0,
            max_tokens=2048,
            seed=4200000 + idx,
            skip_special_tokens=False,
        )

        result = llm.generate(
            [TokensPrompt(prompt_token_ids=prompt_ids)],
            params,
            use_tqdm=False,
        )[0]

        print(f"\n=== question {idx} ===")

        hit = 0
        continued = 0

        for j, out in enumerate(result.outputs):
            ids = list(out.token_ids)

            im_pos = ids.index(IM_END) if IM_END in ids else None
            eos_pos = ids.index(EOS) if EOS in ids else None

            if im_pos is not None:
                hit += 1
                if im_pos < len(ids) - 1:
                    continued += 1

            print(
                f"{j:2d}: len={len(ids):4d} "
                f"im_end={im_pos} eos={eos_pos} "
                f"finish_reason={out.finish_reason}"
            )

            if im_pos is not None:
                lo = max(0, im_pos - 8)
                hi = min(len(ids), im_pos + 20)
                print(
                    "    around im_end:",
                    repr(tok.decode(ids[lo:hi], skip_special_tokens=False)),
                )

        print(
            f"im_end appeared: {hit}/16; "
            f"generation continued after im_end: {continued}/16"
        )


if __name__ == "__main__":
    main()
