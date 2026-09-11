import json
import statistics
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = [
    ("base", "Qwen/Qwen3-0.6B-Base"),
    ("epoch1", "controlled_run_outputs/sft_corrected/trainer/checkpoint-56"),
    ("epoch2", "controlled_run_outputs/sft_corrected/trainer/checkpoint-112"),
    ("pi0", "controlled_run_outputs/sft_corrected/pi_0"),
]

SFT = "data/controlled_run/generated/sft_10k_records.jsonl"

THINK_START = 151667
THINK_END = 151668
EOT = 151643

INDICES = [
    0, 500, 1000, 1500, 2000,
    2500, 3000, 3500, 4000, 4500,
    5000, 5500, 6000, 6500, 7000,
    7500, 8000, 8500, 9000, 9500,
]

with open(SFT) as f:
    rows = [json.loads(line) for line in f]


def rank(logits, token_id):
    return int((logits > logits[token_id]).sum().item()) + 1


for name, path in MODELS:
    print("\n" + "=" * 80)
    print(name, path)

    tok = AutoTokenizer.from_pretrained(path)

    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to("cuda")
    model.eval()

    think_end_ps = []
    think_end_ranks = []
    think_start_ps = []
    think_start_ranks = []

    for idx in INDICES:
        row = rows[idx]

        # ----- full training sequence -----
        rendered = tok.apply_chat_template(
            row["prompt"] + row["completion"],
            tokenize=False,
            add_generation_prompt=False,
        )
        ids = tok(
            rendered,
            add_special_tokens=False,
        )["input_ids"]

        te = [i for i, t in enumerate(ids) if t == THINK_END]
        ts = [i for i, t in enumerate(ids) if t == THINK_START]

        if len(te) != 1 or len(ts) != 1:
            print(idx, "SKIP", len(ts), len(te))
            continue

        # ----- </think> endpoint -----
        pos = te[0]
        x = torch.tensor(
            [ids[:pos]],
            device="cuda",
            dtype=torch.long,
        )

        with torch.inference_mode():
            logits = model(
                x,
                use_cache=False,
            ).logits[0, -1].float()

        probs = torch.softmax(logits, -1)

        think_end_ps.append(probs[THINK_END].item())
        think_end_ranks.append(rank(logits, THINK_END))

        # ----- <think> at assistant start -----
        pos = ts[0]
        x = torch.tensor(
            [ids[:pos]],
            device="cuda",
            dtype=torch.long,
        )

        with torch.inference_mode():
            logits = model(
                x,
                use_cache=False,
            ).logits[0, -1].float()

        probs = torch.softmax(logits, -1)

        think_start_ps.append(probs[THINK_START].item())
        think_start_ranks.append(rank(logits, THINK_START))

    print("\n<think> START")
    print("mean P:", statistics.mean(think_start_ps))
    print("median P:", statistics.median(think_start_ps))
    print("median rank:", statistics.median(think_start_ranks))
    print("rank1:", sum(r == 1 for r in think_start_ranks), "/", len(think_start_ranks))

    print("\n</think> END")
    print("mean P:", statistics.mean(think_end_ps))
    print("median P:", statistics.median(think_end_ps))
    print("median rank:", statistics.median(think_end_ranks))
    print("rank1:", sum(r == 1 for r in think_end_ranks), "/", len(think_end_ranks))

    del model
    torch.cuda.empty_cache()
