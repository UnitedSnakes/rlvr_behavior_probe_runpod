from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


TOKENIZER = "Qwen/Qwen2.5-1.5B-Instruct"


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def comb_pass(c: int, K: int, k: int) -> float:
    """Empirical pass@k by uniformly subsampling k of the K observed rollouts."""
    if c <= 0:
        return 0.0
    if k > K:
        raise ValueError("k cannot exceed K")
    if K - c < k:
        return 1.0
    return 1.0 - math.comb(K - c, k) / math.comb(K, k)


def question_counts(rows, strict_boxed=False):
    out = {}
    for row in rows:
        c = 0
        for r in row["rollouts"]:
            ok = bool(r["correct"])
            if strict_boxed:
                ok = ok and r["extract_method"] == "boxed"
            c += int(ok)
        out[int(row["qid"])] = {
            "c": c,
            "K": int(row["n_rollouts"]),
            "question": row["question"],
        }
    return out


def bootstrap_passk_diff(sft, rl, k, n_boot=5000, seed=123):
    qids = sorted(set(sft) & set(rl))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        sample = rng.choice(qids, size=len(qids), replace=True)
        ds = []
        for qid in sample:
            a, b = sft[qid], rl[qid]
            ds.append(comb_pass(b["c"], b["K"], k) - comb_pass(a["c"], a["K"], k))
        vals.append(np.mean(ds))
    return (
        float(np.mean(vals)),
        float(np.percentile(vals, 2.5)),
        float(np.percentile(vals, 97.5)),
    )


def summarize_model(name, rows, tok, cap):
    methods = Counter()
    lens = []
    correct = 0
    strict_correct = 0

    for row in rows:
        for r in row["rollouts"]:
            methods[r["extract_method"]] += 1
            correct += int(bool(r["correct"]))
            strict_correct += int(bool(r["correct"]) and r["extract_method"] == "boxed")
            lens.append(len(tok.encode(r["text"], add_special_tokens=False)))

    arr = np.asarray(lens)
    near = int((arr >= cap - 4).sum())

    print(f"\n{name}")
    print("-" * len(name))
    print(f"rollouts:                 {len(arr)}")
    print(f"lenient sample accuracy:  {correct/len(arr):.3%}")
    print(f"boxed-only accuracy:      {strict_correct/len(arr):.3%}")
    print(f"extraction methods:       {methods}")
    print(f"boxed rate:               {methods['boxed']/len(arr):.3%}")
    print(f"mean tokens:              {arr.mean():.1f}")
    print(f"median tokens:            {np.median(arr):.1f}")
    print(f"p90 tokens:               {np.percentile(arr, 90):.1f}")
    print(f">= cap-4 ({cap-4}) tokens: {near}/{len(arr)} = {near/len(arr):.1%}")
    print(f"max tokens:               {arr.max()}")


def compare(sft_rows, rl_rows, strict_boxed=False, n_boot=5000):
    sft = question_counts(sft_rows, strict_boxed=strict_boxed)
    rl = question_counts(rl_rows, strict_boxed=strict_boxed)
    qids = sorted(set(sft) & set(rl))
    K = sft[qids[0]]["K"]

    label = "BOXED-ONLY" if strict_boxed else "LENIENT"
    print(f"\n{label} BEHAVIORAL COMPARISON")
    print("=" * (len(label) + 22))

    # sample accuracy
    s_acc = np.mean([sft[q]["c"] / K for q in qids])
    r_acc = np.mean([rl[q]["c"] / K for q in qids])
    print(f"sample accuracy: SFT={s_acc:.3%} RL={r_acc:.3%} delta={r_acc-s_acc:+.3%}")

    # empirical subsampled pass@k curve from the same K rollouts
    ks = sorted(set([1, 2, 4, K]))
    print("\nEmpirical subsampled pass@k curve:")
    for k in ks:
        ps = np.mean([comb_pass(sft[q]["c"], K, k) for q in qids])
        pr = np.mean([comb_pass(rl[q]["c"], K, k) for q in qids])
        mean, lo, hi = bootstrap_passk_diff(sft, rl, k, n_boot=n_boot)
        print(
            f"  k={k:<2d} SFT={ps:.3%} RL={pr:.3%} "
            f"delta={pr-ps:+.3%}  bootstrap95=[{lo:+.3%}, {hi:+.3%}]"
        )

    sharpening = expansion = contraction = regression = unchanged = 0
    for q in qids:
        cs, cr = sft[q]["c"], rl[q]["c"]
        if cs == 0 and cr > 0:
            expansion += 1
        if cs > 0 and cr == 0:
            contraction += 1
        if cs > 0 and cr > cs:
            sharpening += 1
        elif cr < cs:
            regression += 1
        elif cs == cr:
            unchanged += 1

    print("\nQuestion-level counts:")
    print(f"  sharpening:             {sharpening}")
    print(f"  observed expansion:     {expansion}")
    print(f"  observed contraction:   {contraction}")
    print(f"  regression:             {regression}")
    print(f"  unchanged:              {unchanged}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result-dir", default="results_1024")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--bootstrap", type=int, default=5000)
    args = p.parse_args()

    rd = Path(args.result_dir)
    sft_rows = load_jsonl(rd / "sft_raw.jsonl")
    rl_rows = load_jsonl(rd / "rl_raw.jsonl")

    tok = AutoTokenizer.from_pretrained(TOKENIZER)

    summarize_model("SFT", sft_rows, tok, args.max_new_tokens)
    summarize_model("RL", rl_rows, tok, args.max_new_tokens)

    compare(sft_rows, rl_rows, strict_boxed=False, n_boot=args.bootstrap)
    compare(sft_rows, rl_rows, strict_boxed=True, n_boot=args.bootstrap)

    print(
        "\nInterpretation rule:\n"
        "  First require low near-cap rates and similar conclusions under lenient and boxed-only scoring.\n"
        "  Then inspect whether the RL-SFT gap is large at k=1 but shrinks toward k=8.\n"
        "  That is the behavioral signature expected from probability sharpening rather than broad observed-coverage expansion."
    )


if __name__ == "__main__":
    main()
