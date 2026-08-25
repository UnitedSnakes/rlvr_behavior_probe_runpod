#!/usr/bin/env python3
"""Merged-bin follow-up for RL gains by fixed SFT difficulty.

Bins are defined once from results_2048_batched/sft_raw.jsonl:
    0-2/8, 3-5/8, 6-8/8

For each RL checkpoint, report mean RL-SFT gain and a question-level bootstrap
95% interval. The goal is to reduce the instability of the tiny 0/8 and 1-2/8
bins from the exploratory four-bin analysis.

Run from repo root:
    python analyses/difficulty_bins_merged.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


BIN_SPECS = [
    ("0-2/8", 0, 2),
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
    by_qid = {}

    for row in rows:
        qid = int(row["qid"])
        if qid in by_qid:
            raise ValueError(f"Duplicate qid {qid} in {path}")

        rollouts = row["rollouts"]
        n = len(rollouts)
        c = sum(bool(r["correct"]) for r in rollouts)

        if int(row["n_rollouts"]) != n:
            raise ValueError(f"{path}: qid {qid} has inconsistent n_rollouts")
        if int(row["n_correct"]) != c:
            raise ValueError(
                f"{path}: qid {qid} has n_correct={row['n_correct']} "
                f"but rollout flags sum to {c}"
            )

        by_qid[qid] = {
            "qid": qid,
            "n": n,
            "c": c,
            "rate": c / n,
        }

    return by_qid


def bin_for_count(c: int, n: int) -> str:
    if n != 8:
        raise ValueError(f"Expected 8 rollouts/question, found {n}")
    for label, lo, hi in BIN_SPECS:
        if lo <= c <= hi:
            return label
    raise ValueError(f"Unexpected count {c}/{n}")


def checkpoint_sort_key(path: Path) -> int:
    m = re.fullmatch(r"step-(\d+)", path.name)
    if not m:
        raise ValueError(f"Unexpected checkpoint directory: {path}")
    return int(m.group(1))


def validate_same_questions(a: dict[int, dict], b: dict[int, dict], name: str) -> None:
    if set(a) != set(b):
        raise ValueError(f"{name}: qid sets do not match")


def bootstrap_mean_ci(
    values: list[float],
    rng: np.random.Generator,
    n_boot: int = 10000,
) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    if len(x) == 1:
        return float(x[0]), float(x[0])

    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def summarize(
    checkpoint: str,
    sft: dict[int, dict],
    rl: dict[int, dict],
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[list[dict], list[dict]]:
    validate_same_questions(sft, rl, checkpoint)

    per_question = []
    for qid in sorted(sft):
        s = sft[qid]
        r = rl[qid]
        if s["n"] != r["n"]:
            raise ValueError(f"{checkpoint}: rollout count mismatch for qid {qid}")

        gain = r["rate"] - s["rate"]
        per_question.append(
            {
                "checkpoint": checkpoint,
                "qid": qid,
                "bin": bin_for_count(s["c"], s["n"]),
                "sft_correct": s["c"],
                "rl_correct": r["c"],
                "sft_rate": s["rate"],
                "rl_rate": r["rate"],
                "gain": gain,
            }
        )

    rows = []
    for label, _, _ in BIN_SPECS:
        xs = [x for x in per_question if x["bin"] == label]
        gains = [x["gain"] for x in xs]
        lo, hi = bootstrap_mean_ci(gains, rng, n_boot=n_boot)

        rows.append(
            {
                "checkpoint": checkpoint,
                "bin": label,
                "n_questions": len(xs),
                "mean_sft_rate": np.mean([x["sft_rate"] for x in xs]),
                "mean_rl_rate": np.mean([x["rl_rate"] for x in xs]),
                "mean_gain": np.mean(gains),
                "ci_low": lo,
                "ci_high": hi,
                "improved": sum(x["gain"] > 0 for x in xs),
                "regressed": sum(x["gain"] < 0 for x in xs),
                "unchanged": sum(x["gain"] == 0 for x in xs),
            }
        )

    return rows, per_question


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_rows(rows: list[dict]) -> None:
    print("\nMerged fixed-SFT difficulty bins")
    print("=" * 104)
    print(
        f"{'checkpoint':<12} {'bin':<8} {'n':>3} "
        f"{'SFT':>8} {'RL':>8} {'gain':>9} {'95% bootstrap CI':>24} {'+/−/=':>9}"
    )
    print("-" * 104)

    for r in rows:
        ci = f"[{100*r['ci_low']:+.1f}, {100*r['ci_high']:+.1f}]"
        signs = f"{r['improved']}/{r['regressed']}/{r['unchanged']}"
        print(
            f"{r['checkpoint']:<12} {r['bin']:<8} {r['n_questions']:>3} "
            f"{100*r['mean_sft_rate']:>7.1f}% "
            f"{100*r['mean_rl_rate']:>7.1f}% "
            f"{100*r['mean_gain']:>+8.1f}pp "
            f"{ci:>24} {signs:>9}"
        )


def make_final_plot(rows: list[dict], out_path: Path) -> None:
    final = [r for r in rows if r["checkpoint"] == "final"]
    labels = [x[0] for x in BIN_SPECS]
    by_bin = {r["bin"]: r for r in final}

    y = [100 * by_bin[b]["mean_gain"] for b in labels]
    yerr_low = [100 * (by_bin[b]["mean_gain"] - by_bin[b]["ci_low"]) for b in labels]
    yerr_high = [100 * (by_bin[b]["ci_high"] - by_bin[b]["mean_gain"]) for b in labels]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(labels))
    ax.errorbar(
        x,
        y,
        yerr=np.vstack([yerr_low, yerr_high]),
        fmt="o",
        capsize=4,
    )
    ax.axhline(0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("SFT correct rollouts out of 8")
    ax.set_ylabel("Final RL − SFT success rate (percentage points)")
    ax.set_title("Final RL gain by fixed SFT difficulty")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--final-dir", default="results_2048_batched")
    p.add_argument("--trajectory-dir", default="trajectory")
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--summary-csv",
        default="analyses/difficulty_bins_merged.csv",
    )
    p.add_argument(
        "--per-question-csv",
        default="analyses/difficulty_bins_merged_per_question.csv",
    )
    p.add_argument(
        "--figure",
        default="figures/difficulty_gain_merged_final.png",
    )
    args = p.parse_args()

    root = args.root.resolve()
    final_dir = root / args.final_dir
    traj_dir = root / args.trajectory_dir
    rng = np.random.default_rng(args.seed)

    sft = load_counts(final_dir / "sft_raw.jsonl")

    all_rows = []
    all_per_question = []

    checkpoint_dirs = sorted(
        [x for x in traj_dir.glob("step-*") if x.is_dir()],
        key=checkpoint_sort_key,
    )

    for d in checkpoint_dirs:
        label = d.name.replace("step-", "step ")
        checkpoint_sft = load_counts(d / "sft_raw.jsonl")
        validate_same_questions(sft, checkpoint_sft, f"{label} SFT")

        for qid in sft:
            if sft[qid]["c"] != checkpoint_sft[qid]["c"]:
                raise ValueError(f"{label}: fixed SFT baseline changed at qid {qid}")

        rl = load_counts(d / "rl_raw.jsonl")
        rows, per_q = summarize(label, sft, rl, rng, args.n_boot)
        all_rows.extend(rows)
        all_per_question.extend(per_q)

    final_rl = load_counts(final_dir / "rl_raw.jsonl")
    rows, per_q = summarize("final", sft, final_rl, rng, args.n_boot)
    all_rows.extend(rows)
    all_per_question.extend(per_q)

    write_csv(root / args.summary_csv, all_rows)
    write_csv(root / args.per_question_csv, all_per_question)
    make_final_plot(all_rows, root / args.figure)
    print_rows(all_rows)

    sizes = {
        label: sum(bin_for_count(v["c"], v["n"]) == label for v in sft.values())
        for label, _, _ in BIN_SPECS
    }
    print("\nFixed SFT bin sizes:")
    print("  " + ", ".join(f"{k}: {v}" for k, v in sizes.items()))
    print(f"\nWrote: {root / args.summary_csv}")
    print(f"Wrote: {root / args.per_question_csv}")
    print(f"Wrote: {root / args.figure}")


if __name__ == "__main__":
    main()
