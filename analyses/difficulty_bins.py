#!/usr/bin/env python3
"""Analyze RL gains by SFT success-rate bin.

The bins are defined once from the SFT baseline in results_2048_batched:
    0/8, 1-2/8, 3-5/8, 6-8/8

For each RL checkpoint, the script reports the mean change in per-question
success rate within each bin and writes a single figure showing how those gains
change over training.

Run from the repository root:
    python analyses/difficulty_bins.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


BIN_SPECS = [
    ("0/8", 0, 0),
    ("1-2/8", 1, 2),
    ("3-5/8", 3, 5),
    ("6-8/8", 6, 8),
]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_counts(path: Path) -> dict[int, dict]:
    rows = read_jsonl(path)
    by_qid: dict[int, dict] = {}

    for row in rows:
        qid = int(row["qid"])
        if qid in by_qid:
            raise ValueError(f"Duplicate qid {qid} in {path}")

        rollouts = row["rollouts"]
        n = len(rollouts)
        c = sum(bool(r["correct"]) for r in rollouts)

        if int(row["n_rollouts"]) != n:
            raise ValueError(
                f"qid {qid}: n_rollouts={row['n_rollouts']} but found {n} rollouts"
            )
        if int(row["n_correct"]) != c:
            raise ValueError(
                f"qid {qid}: n_correct={row['n_correct']} but rollout flags sum to {c}. "
                "Re-run rescoring/summarization first."
            )

        by_qid[qid] = {
            "qid": qid,
            "question": row.get("question", ""),
            "n": n,
            "c": c,
            "rate": c / n,
        }

    return by_qid


def bin_for_count(c: int, n: int) -> str:
    if n != 8:
        raise ValueError(
            f"This analysis currently assumes 8 rollouts/question; found n={n}."
        )
    for label, lo, hi in BIN_SPECS:
        if lo <= c <= hi:
            return label
    raise ValueError(f"Unexpected count {c}/{n}")


def checkpoint_sort_key(path: Path) -> int:
    m = re.fullmatch(r"step-(\d+)", path.name)
    if not m:
        raise ValueError(f"Unexpected checkpoint directory: {path}")
    return int(m.group(1))


def validate_same_questions(reference: dict[int, dict], other: dict[int, dict], name: str) -> None:
    if set(reference) != set(other):
        missing = sorted(set(reference) - set(other))
        extra = sorted(set(other) - set(reference))
        raise ValueError(f"{name}: qid mismatch; missing={missing}, extra={extra}")


def summarize_checkpoint(
    label: str,
    sft: dict[int, dict],
    rl: dict[int, dict],
) -> tuple[list[dict], list[dict]]:
    validate_same_questions(sft, rl, label)

    per_question = []
    for qid in sorted(sft):
        s = sft[qid]
        r = rl[qid]
        if s["n"] != r["n"]:
            raise ValueError(f"{label}, qid {qid}: SFT/RL rollout-count mismatch")

        gain = r["rate"] - s["rate"]
        per_question.append(
            {
                "checkpoint": label,
                "qid": qid,
                "sft_correct": s["c"],
                "rl_correct": r["c"],
                "n_rollouts": s["n"],
                "sft_rate": s["rate"],
                "rl_rate": r["rate"],
                "gain": gain,
                "bin": bin_for_count(s["c"], s["n"]),
            }
        )

    summary = []
    for bin_label, _, _ in BIN_SPECS:
        xs = [x for x in per_question if x["bin"] == bin_label]
        if not xs:
            continue

        gains = [x["gain"] for x in xs]
        summary.append(
            {
                "checkpoint": label,
                "bin": bin_label,
                "n_questions": len(xs),
                "mean_sft_rate": sum(x["sft_rate"] for x in xs) / len(xs),
                "mean_rl_rate": sum(x["rl_rate"] for x in xs) / len(xs),
                "mean_gain": sum(gains) / len(gains),
                "improved": sum(g > 0 for g in gains),
                "regressed": sum(g < 0 for g in gains),
                "unchanged": sum(g == 0 for g in gains),
            }
        )

    return summary, per_question


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict]) -> None:
    print("\nGain by fixed SFT difficulty bin")
    print("=" * 78)
    print(
        f"{'checkpoint':<12} {'SFT bin':<9} {'n':>3} "
        f"{'SFT':>8} {'RL':>8} {'gain':>9} {'+/−/=':>9}"
    )
    print("-" * 78)
    for row in rows:
        signs = f"{row['improved']}/{row['regressed']}/{row['unchanged']}"
        print(
            f"{row['checkpoint']:<12} {row['bin']:<9} {row['n_questions']:>3} "
            f"{100 * row['mean_sft_rate']:>7.1f}% "
            f"{100 * row['mean_rl_rate']:>7.1f}% "
            f"{100 * row['mean_gain']:>+8.1f}pp {signs:>9}"
        )


def make_plot(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_order = []
    for row in rows:
        if row["checkpoint"] not in checkpoint_order:
            checkpoint_order.append(row["checkpoint"])

    bin_labels = [x[0] for x in BIN_SPECS]
    x = list(range(len(bin_labels)))

    fig, ax = plt.subplots(figsize=(8, 5))
    for checkpoint in checkpoint_order:
        by_bin = {
            row["bin"]: row
            for row in rows
            if row["checkpoint"] == checkpoint
        }
        y = [
            100 * by_bin[b]["mean_gain"] if b in by_bin else float("nan")
            for b in bin_labels
        ]
        ax.plot(x, y, marker="o", label=checkpoint)

    ax.axhline(0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel("SFT correct rollouts out of 8")
    ax.set_ylabel("Mean RL − SFT success rate (percentage points)")
    ax.set_title("Where do RL gains occur as a function of SFT difficulty?")
    ax.legend(title="RL checkpoint")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--final-dir",
        default="results_2048_batched",
        help="Directory containing the fixed SFT baseline and final RL results.",
    )
    parser.add_argument("--trajectory-dir", default="trajectory")
    parser.add_argument(
        "--figure",
        default="figures/difficulty_gain_by_sft_bin.png",
    )
    parser.add_argument(
        "--summary-csv",
        default="analyses/difficulty_bins.csv",
    )
    parser.add_argument(
        "--per-question-csv",
        default="analyses/difficulty_bins_per_question.csv",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    final_dir = root / args.final_dir
    trajectory_dir = root / args.trajectory_dir

    sft = load_counts(final_dir / "sft_raw.jsonl")
    if len(sft) != 30:
        print(f"[warning] Expected 30 SFT questions, found {len(sft)}")

    # Define difficulty once from this fixed SFT sample, then keep the same bins
    # for every checkpoint.
    all_summary: list[dict] = []
    all_per_question: list[dict] = []

    checkpoint_dirs = sorted(
        [p for p in trajectory_dir.glob("step-*") if p.is_dir()],
        key=checkpoint_sort_key,
    )

    for checkpoint_dir in checkpoint_dirs:
        label = checkpoint_dir.name.replace("step-", "step ")

        # The repo stores SFT next to each checkpoint. Check that it is the same
        # fixed SFT baseline rather than silently redefining difficulty.
        checkpoint_sft = load_counts(checkpoint_dir / "sft_raw.jsonl")
        validate_same_questions(sft, checkpoint_sft, f"{label} SFT")
        for qid in sft:
            if sft[qid]["c"] != checkpoint_sft[qid]["c"]:
                raise ValueError(
                    f"{label}: SFT baseline differs at qid {qid}. "
                    "Use one fixed SFT rollout set for binning."
                )

        rl = load_counts(checkpoint_dir / "rl_raw.jsonl")
        summary, per_question = summarize_checkpoint(label, sft, rl)
        all_summary.extend(summary)
        all_per_question.extend(per_question)

    final_rl = load_counts(final_dir / "rl_raw.jsonl")
    summary, per_question = summarize_checkpoint("final", sft, final_rl)
    all_summary.extend(summary)
    all_per_question.extend(per_question)

    write_csv(root / args.summary_csv, all_summary)
    write_csv(root / args.per_question_csv, all_per_question)
    make_plot(all_summary, root / args.figure)
    print_summary(all_summary)

    bin_sizes = {
        label: sum(bin_for_count(x["c"], x["n"]) == label for x in sft.values())
        for label, _, _ in BIN_SPECS
    }
    print("\nFixed SFT bin sizes:")
    print("  " + ", ".join(f"{k}: {v}" for k, v in bin_sizes.items()))
    print(f"\nWrote: {root / args.summary_csv}")
    print(f"Wrote: {root / args.per_question_csv}")
    print(f"Wrote: {root / args.figure}")


if __name__ == "__main__":
    main()
