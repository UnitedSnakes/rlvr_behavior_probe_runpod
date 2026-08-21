from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from transformers import AutoTokenizer

TOKENIZER = "Qwen/Qwen2.5-1.5B-Instruct"


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pass_at_k(c: int, K: int, k: int) -> float:
    if c <= 0:
        return 0.0
    if k > K:
        raise ValueError(f"k={k} cannot exceed K={K}")
    if K - c < k:
        return 1.0
    return 1.0 - math.comb(K - c, k) / math.comb(K, k)


def sample_accuracy(rows) -> float:
    correct = sum(int(r["correct"]) for row in rows for r in row["rollouts"])
    total = sum(len(row["rollouts"]) for row in rows)
    return correct / total


def pass_curve(rows, ks):
    vals = []
    for k in ks:
        per_question = []
        for row in rows:
            c = sum(int(r["correct"]) for r in row["rollouts"])
            K = len(row["rollouts"])
            per_question.append(pass_at_k(c, K, k))
        vals.append(sum(per_question) / len(per_question))
    return vals


def mean_response_tokens(rows, tokenizer) -> float:
    lengths = [
        len(tokenizer.encode(r["text"], add_special_tokens=False))
        for row in rows
        for r in row["rollouts"]
    ]
    return sum(lengths) / len(lengths)


def save_pass_at_k(result_dir: Path, figure_dir: Path):
    sft = load_jsonl(result_dir / "sft_raw.jsonl")
    rl = load_jsonl(result_dir / "rl_raw.jsonl")
    K = len(sft[0]["rollouts"])
    ks = [k for k in (1, 2, 4, 8) if k <= K]

    sft_curve = pass_curve(sft, ks)
    rl_curve = pass_curve(rl, ks)

    plt.figure(figsize=(6.4, 4.2))
    plt.plot(ks, sft_curve, marker="o", label="SFT")
    plt.plot(ks, rl_curve, marker="o", label="Final RLVR")
    plt.xlabel("k")
    plt.ylabel("pass@k")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.xticks(ks)
    plt.title("RLVR advantage narrows as k increases")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "pass_at_k.png", dpi=180)
    plt.close()

    print("pass@k")
    for k, s, r in zip(ks, sft_curve, rl_curve):
        print(f"  k={k}: SFT={s:.3%}, RL={r:.3%}, delta={r-s:+.3%}")


def trajectory_paths(root: Path):
    return [
        ("SFT", root / "results_2048_batched" / "sft_raw.jsonl"),
        ("100", root / "trajectory" / "step-100" / "rl_raw.jsonl"),
        ("300", root / "trajectory" / "step-300" / "rl_raw.jsonl"),
        ("500", root / "trajectory" / "step-500" / "rl_raw.jsonl"),
        ("1000", root / "trajectory" / "step-1000" / "rl_raw.jsonl"),
        ("final", root / "results_2048_batched" / "rl_raw.jsonl"),
    ]


def save_accuracy_trajectory(root: Path, figure_dir: Path):
    entries = trajectory_paths(root)
    labels = [label for label, _ in entries]
    acc = [sample_accuracy(load_jsonl(path)) for _, path in entries]

    plt.figure(figsize=(6.4, 4.2))
    plt.plot(range(len(labels)), acc, marker="o")
    plt.xticks(range(len(labels)), labels)
    plt.xlabel("Checkpoint")
    plt.ylabel("Sample accuracy")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.title("Sample accuracy across post-training checkpoints")
    plt.tight_layout()
    plt.savefig(figure_dir / "trajectory_accuracy.png", dpi=180)
    plt.close()

    print("sample accuracy trajectory")
    for label, value in zip(labels, acc):
        print(f"  {label}: {value:.3%}")


def save_length_trajectory(root: Path, figure_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    entries = trajectory_paths(root)
    labels = [label for label, _ in entries]
    means = [mean_response_tokens(load_jsonl(path), tokenizer) for _, path in entries]

    plt.figure(figsize=(6.4, 4.2))
    plt.plot(range(len(labels)), means, marker="o")
    plt.xticks(range(len(labels)), labels)
    plt.xlabel("Checkpoint")
    plt.ylabel("Mean response length (tokens)")
    plt.title("Most response-length change happens early")
    plt.tight_layout()
    plt.savefig(figure_dir / "trajectory_length.png", dpi=180)
    plt.close()

    print("mean response length")
    for label, value in zip(labels, means):
        print(f"  {label}: {value:.1f} tokens")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--figure-dir", default="figures")
    args = p.parse_args()

    root = Path(args.root)
    figure_dir = root / args.figure_dir
    figure_dir.mkdir(parents=True, exist_ok=True)

    save_pass_at_k(root / "results_2048_batched", figure_dir)
    save_accuracy_trajectory(root, figure_dir)
    save_length_trajectory(root, figure_dir)

    print(f"\nSaved figures to {figure_dir}/")


if __name__ == "__main__":
    main()
