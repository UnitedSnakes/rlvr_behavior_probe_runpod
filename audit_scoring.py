import json
import random

random.seed(42)

def load_rollouts(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            for r in q["rollouts"]:
                rows.append({
                    "qid": q["qid"],
                    "question": q["question"],
                    "gold": q["gold_value"],
                    **r,
                })
    return rows


for name in ["sft", "rl"]:
    rows = load_rollouts(f"results_2048_batched/{name}_raw.jsonl")

    correct = [x for x in rows if x["correct"]]
    wrong = [x for x in rows if not x["correct"]]

    samples = (
        [("CORRECT", x) for x in random.sample(correct, 5)]
        + [("WRONG", x) for x in random.sample(wrong, 5)]
    )

    for label, x in samples:
        print("\n" + "=" * 100)
        print(name.upper(), label)
        print("qid:", x["qid"])
        print("question:", x["question"])
        print("gold:", x["gold"])
        print("pred:", x["pred_value"])
        print("method:", x["extract_method"])
        print("\nRESPONSE TAIL:")
        print(x["text"][-2000:])