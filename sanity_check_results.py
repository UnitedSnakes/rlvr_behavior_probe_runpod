from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pass_at_k(n: int, c: int, k: int) -> float:
    if not 1 <= k <= n:
        raise ValueError(f"Need 1 <= k <= n, got k={k}, n={n}")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def audit(alias: str, rows, source):
    print(f"\n[{alias.upper()}]")
    qids = [int(r["qid"]) for r in rows]
    counts = Counter(qids)

    assert len(qids) == len(source), f"{alias}: expected {len(source)} rows, got {len(qids)}"
    assert len(set(qids)) == len(qids), f"{alias}: duplicate qids: {[q for q, n in counts.items() if n > 1]}"
    assert set(qids) == set(source), f"{alias}: qid set does not match source questions"

    Ks = set()
    all_correct = []

    for row in rows:
        qid = int(row["qid"])
        src = source[qid]

        assert row["question"] == src["question"], f"{alias} qid={qid}: question text mismatch"
        assert str(row["gold"]).strip() == str(src["gold"]).strip(), f"{alias} qid={qid}: gold mismatch"

        rollouts = row["rollouts"]
        K = int(row["n_rollouts"])
        Ks.add(K)

        assert len(rollouts) == K, f"{alias} qid={qid}: {len(rollouts)} rollouts, metadata says {K}"
        assert sorted(int(r["rollout"]) for r in rollouts) == list(range(K)), f"{alias} qid={qid}: bad rollout indices"

        flags = [bool(r["correct"]) for r in rollouts]
        assert sum(flags) == int(row["n_correct"]), f"{alias} qid={qid}: n_correct mismatch"
        all_correct.extend(flags)

    assert len(Ks) == 1, f"{alias}: inconsistent rollout counts: {Ks}"
    K = next(iter(Ks))

    sample_acc = mean(all_correct)
    ks = sorted(set(k for k in (1, 2, 4, K) if k <= K))
    curve = {
        k: mean(pass_at_k(K, int(row["n_correct"]), k) for row in rows)
        for k in ks
    }

    assert math.isclose(curve[1], sample_acc, abs_tol=1e-12), (
        f"{alias}: pass@1 != sample accuracy: {curve[1]} vs {sample_acc}"
    )
    assert all(curve[a] <= curve[b] + 1e-12 for a, b in zip(ks, ks[1:])), (
        f"{alias}: pass@k is not monotone: {curve}"
    )

    literal_boxed = sum("\\boxed" in r["text"] for row in rows for r in row["rollouts"])
    parser_boxed = sum(r["extract_method"] == "boxed" for row in rows for r in row["rollouts"])
    total = len(all_correct)

    print(f"[PASS] {len(rows)} unique questions")
    print(f"[PASS] {K} rollouts per question, rollout indices 0..{K-1}")
    print("[PASS] question text and gold exactly match data/gsm8k_subset.jsonl")
    print("[PASS] stored n_correct matches rollout-level correctness flags")
    print(f"[PASS] pass@1 == sample accuracy == {sample_acc:.4f}")
    print("[PASS] pass@k monotone: " + ", ".join(f"k={k}: {curve[k]:.4f}" for k in ks))
    print(f"[INFO] literal \\boxed present: {literal_boxed}/{total} = {literal_boxed/total:.1%}")
    print(f"[INFO] parser extract_method='boxed': {parser_boxed}/{total} = {parser_boxed/total:.1%}")
    if literal_boxed != parser_boxed:
        print("[WARN] The parser is not recognizing every literal \\boxed answer. Inspect probe/scoring.py.")

    return {int(r["qid"]): r for r in rows}


def show_manual_samples(sft, rl, qids=(0, 7, 19), n_rollouts=2):
    print("\n[MANUAL ROLLOUT↔QUESTION CHECK]")
    for qid in qids:
        if qid not in sft or qid not in rl:
            continue
        print("\n" + "=" * 88)
        print(f"qid={qid}\nQUESTION: {sft[qid]['question']}")
        for alias, rows in (("SFT", sft), ("RL", rl)):
            print(f"\n{alias}:")
            for r in rows[qid]["rollouts"][:n_rollouts]:
                compact = " ".join(r["text"].split())
                print(
                    f"  rollout={r['rollout']} pred={r['pred_value']} correct={r['correct']}\n"
                    f"  {compact[:700]}"
                )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result-dir", default="results_2048_batched")
    p.add_argument("--question-file", default="data/gsm8k_subset.jsonl")
    args = p.parse_args()

    source_rows = read_jsonl(Path(args.question_file))
    source = {int(r["qid"]): r for r in source_rows}
    assert len(source) == len(source_rows), "source question file has duplicate qids"

    rd = Path(args.result_dir)
    sft_rows = read_jsonl(rd / "sft_raw.jsonl")
    rl_rows = read_jsonl(rd / "rl_raw.jsonl")

    sft = audit("sft", sft_rows, source)
    rl = audit("rl", rl_rows, source)
    show_manual_samples(sft, rl)

    print("\nAll structural/pass@k checks passed. Read the printed samples to finish the semantic mapping check.")


if __name__ == "__main__":
    main()
