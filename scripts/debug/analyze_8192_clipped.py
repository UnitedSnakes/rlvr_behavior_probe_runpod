import json
import re
from collections import Counter, defaultdict

from transformers import AutoTokenizer

from probe.scoring import extract_numeric_answer, numeric_equal, _to_number


PATH = (
    "controlled_run_outputs/sft_corrected/"
    "horizon_curve_2048_4096_8192.json"
)
POLICY = "controlled_run_outputs/sft_corrected/pi_0"

tok = AutoTokenizer.from_pretrained(POLICY)

RESTART_PATTERNS = {
    "alternatively": r"\balternatively\b",
    "wait": r"\bwait\b",
    "reconsider": r"\breconsider\b",
    "another_approach": r"\banother (?:approach|way|method)\b",
    "check_again": r"\b(?:check|verify|recheck) (?:again|this)\b",
    "maybe": r"\bmaybe\b",
    "however": r"\bhowever\b",
    "let_me": r"\blet me\b",
    "actually": r"\bactually\b",
}


def balanced_box_ends(text):
    out = []
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
                    out.append((m.start(), j + 1))
                    break
    return out


def strict_answer_events(text, gold):
    """Only boxed/final-phrase answers. Never last-number fallback."""
    events = []

    # Boxed answers.
    for start, end in balanced_box_ends(text):
        pred, raw, method = extract_numeric_answer(text[start:end])
        if method == "boxed":
            events.append({
                "kind": "boxed",
                "start": start,
                "end": end,
                "pred": pred,
                "correct": numeric_equal(pred, gold),
            })

    # Explicit final-answer language.
    pattern = re.compile(
        r"(?:final\s+answer|answer\s+is|therefore|thus)"
        r"\s*[:=]?\s*\$?\s*"
        r"([+-]?(?:\d[\d,]*)(?:\.\d+)?"
        r"(?:/\d+(?:\.\d+)?)?)",
        re.I,
    )

    for m in pattern.finditer(text):
        pred = _to_number(m.group(1))
        events.append({
            "kind": "final_phrase",
            "start": m.start(),
            "end": m.end(),
            "pred": pred,
            "correct": numeric_equal(pred, gold),
        })

    return sorted(events, key=lambda x: x["end"])


def max_ngram_repeat(ids, n):
    if len(ids) < n:
        return 0

    counts = Counter(
        tuple(ids[i:i+n])
        for i in range(len(ids) - n + 1)
    )
    return max(counts.values(), default=0)


with open(PATH) as f:
    data = json.load(f)

clipped = [
    x for x in data["trajectories_8192"]
    if x["finish_reason_8192"] == "length"
]

print("8192 clipped:", len(clipped))

records = []

for x in clipped:
    text = x["text_8192"]
    ids = x["token_ids"]
    gold = _to_number(str(x["gold"]))

    events = strict_answer_events(text, gold)
    correct_events = [e for e in events if e["correct"]]

    restart_counts = {
        name: len(re.findall(pattern, text, re.I))
        for name, pattern in RESTART_PATTERNS.items()
    }
    restart_total = sum(restart_counts.values())

    repeat32 = max_ngram_repeat(ids, 32)
    repeat64 = max_ngram_repeat(ids, 64)

    think_close = text.find("</think>")
    has_think_close = think_close >= 0

    if correct_events:
        first = correct_events[0]
        first_correct_token = len(tok(
            text[:first["end"]],
            add_special_tokens=False,
        )["input_ids"])
    else:
        first_correct_token = None

    loop_like = repeat32 >= 4 or repeat64 >= 3

    if not has_think_close:
        category = "never_closed_think"
    elif correct_events:
        category = "answered_then_continued"
    elif loop_like:
        category = "repetition_loop"
    elif restart_total >= 3:
        category = "self_restart"
    else:
        category = "long_unresolved"

    records.append({
        "dataset_index": x["dataset_index"],
        "rollout": x["rollout"],
        "category": category,
        "has_think_close": has_think_close,
        "strict_answers": len(events),
        "strict_correct_answers": len(correct_events),
        "first_correct_token": first_correct_token,
        "restart_total": restart_total,
        "restart_counts": restart_counts,
        "repeat32": repeat32,
        "repeat64": repeat64,
    })


print("\n=== PRIMARY CLASSIFICATION ===")
print(Counter(x["category"] for x in records))

print("\n=== THINK STATE ===")
print(Counter(
    "closed" if x["has_think_close"] else "never_closed"
    for x in records
))

print("\n=== STRICT ANSWER BEFORE 8192 ===")
print(
    "any strict answer:",
    sum(x["strict_answers"] > 0 for x in records),
    "/", len(records),
)
print(
    "correct strict answer:",
    sum(x["strict_correct_answers"] > 0 for x in records),
    "/", len(records),
)

positions = sorted(
    x["first_correct_token"]
    for x in records
    if x["first_correct_token"] is not None
)
if positions:
    print("first correct answer token positions:", positions)


print("\n=== RESTART / LOOP SIGNALS ===")
print(
    "restart >= 3:",
    sum(x["restart_total"] >= 3 for x in records),
)
print(
    "repeat32 >= 4:",
    sum(x["repeat32"] >= 4 for x in records),
)
print(
    "repeat64 >= 3:",
    sum(x["repeat64"] >= 3 for x in records),
)

totals = Counter()
for x in records:
    totals.update(x["restart_counts"])
print("restart markers:", totals)


print("\n=== BY PROMPT ===")
by_prompt = defaultdict(list)
for x in records:
    by_prompt[x["dataset_index"]].append(x)

for idx in sorted(by_prompt):
    xs = by_prompt[idx]
    print(
        idx,
        "n=", len(xs),
        "categories=", dict(Counter(x["category"] for x in xs)),
    )


print("\n=== PER TRAJECTORY ===")
for x in records:
    print(
        f"p{x['dataset_index']} r{x['rollout']:02d} "
        f"{x['category']:24s} "
        f"strict_correct={x['strict_correct_answers']} "
        f"first_correct_tok={x['first_correct_token']} "
        f"restart={x['restart_total']} "
        f"rep32={x['repeat32']} rep64={x['repeat64']} "
        f"think={'Y' if x['has_think_close'] else 'N'}"
    )


out = (
    "controlled_run_outputs/sft_corrected/"
    "analysis_8192_clipped.json"
)
with open(out, "w") as f:
    json.dump(records, f, indent=2)

print("\nsaved:", out)
