import json
import re

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from controlled_run.data import build_gsm8k_rl_rows


POLICY = "controlled_run_outputs/sft_corrected/pi_0"
DIAG = (
    "controlled_run_outputs/sft_corrected/"
    "prompt7_termination_diagnostic.json"
)
GSM8K_SHA = "740312add88f781978c0658806c59bc2815b9866"

EOT = 151643
IM_END = 151645
IDX = 7


def boxed_ends(text):
    """Return character offsets immediately after each balanced \\boxed{...}."""
    ends = []

    for m in re.finditer(r"\\boxed\s*", text):
        i = m.end()
        if i >= len(text) or text[i] != "{":
            continue

        depth = 0
        for j in range(i, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    ends.append(j + 1)
                    break

    return ends


def main():
    with open(DIAG) as f:
        d = json.load(f)

    raw = load_dataset(
        "openai/gsm8k",
        "main",
        revision=GSM8K_SHA,
        split="train[:8]",
    )
    row = build_gsm8k_rl_rows(raw)[IDX]

    tok = AutoTokenizer.from_pretrained(POLICY)
    prompt_text = tok.apply_chat_template(
        row["prompt"],
        tokenize=False,
        add_generation_prompt=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        POLICY,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to("cuda")
    model.eval()

    def probe(prefix):
        ids = tok(
            prefix,
            add_special_tokens=False,
        )["input_ids"]

        x = torch.tensor(
            [ids],
            dtype=torch.long,
            device="cuda",
        )

        with torch.inference_mode():
            logits = model(
                x,
                use_cache=False,
            ).logits[0, -1].float()

        probs = torch.softmax(logits, dim=-1)

        p_eot = probs[EOT].item()
        p_im = probs[IM_END].item()
        rank_eot = int(
            (logits > logits[EOT]).sum().item()
        ) + 1

        top = torch.topk(probs, 5)

        top5 = [
            (
                int(token_id),
                float(prob),
                repr(tok.decode([int(token_id)])),
            )
            for prob, token_id in zip(
                top.values,
                top.indices,
            )
        ]

        return p_eot, rank_eot, p_im, top5

    print("=== GENERATED-STATE EOT PROBE ===")

    for r in d["rollouts"]:
        text = r["text"]
        ends = boxed_ends(text)

        # Most important cases:
        # 1. stopped outputs
        # 2. length-clipped outputs that reached boxed answer
        if r["finish_reason"] == "stop":
            p, rank, pim, top5 = probe(
                prompt_text + text
            )

            print(
                f"\n[{r['rollout']}] STOPPED "
                f"len={r['length']} correct={r['correct']}"
            )
            print(
                f"full endpoint: "
                f"P(EOT)={p:.6f} "
                f"rank={rank} "
                f"P(im_end)={pim:.3g}"
            )
            print("top5:", top5)

        if r["finish_reason"] == "length" and ends:
            for k, end in enumerate(ends):
                p, rank, pim, top5 = probe(
                    prompt_text + text[:end]
                )

                after = " ".join(
                    text[end:end + 180].split()
                )

                print(
                    f"\n[{r['rollout']}] CLIPPED "
                    f"BOX#{k + 1} "
                    f"len={r['length']} "
                    f"correct={r['correct']}"
                )
                print(
                    f"immediately after box: "
                    f"P(EOT)={p:.6f} "
                    f"rank={rank} "
                    f"P(im_end)={pim:.3g}"
                )
                print("next text:", repr(after))
                print("top5:", top5)


if __name__ == "__main__":
    main()
