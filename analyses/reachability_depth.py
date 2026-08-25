#!/usr/bin/env python3
"""Analyze how observed SFT reachability changes with sampling depth.

This analysis uses a deep SFT trajectory bank and a fixed RL reference run.
For each sampling budget K, it repeatedly draws a random ordering of each
question's SFT rollouts and asks whether at least one success appears in the
first K samples.

Run from the repository root, for example:
    python analyses/reachability_depth.py \
        --sft-file results_sft_256/sft_raw.jsonl \
        --rl-file results_2048_batched/rl_raw.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_K_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def validate_rows(rows: list[dict], source_name: str) -> dict[int, dict]:
    by_qid: dict[int, dict] = {}

    for row in rows:
        qid = int(row["qid"])
        if qid in by_qid:
            raise ValueError(f"Duplicate qid {qid} in {source_name}")

        rollouts = row["rollouts"]
        n_rollouts = len(rollouts)
        n_correct = sum(bool(rollout["correct"]) for rollout in rollouts)

        if int(row["n_rollouts"]) != n_rollouts:
            raise ValueError(
                f"{source_name}, qid {qid}: n_rollouts={row['n_rollouts']} "
                f"but found {n_rollouts} rollout records"
            )

        if int(row["n_correct"]) != n_correct:
            raise ValueError(
                f"{source_name}, qid {qid}: n_correct={row['n_correct']} "
                f"but rollout flags sum to {n_correct}"
            )

        by_qid[qid] = row

    return by_qid


def validate_same_questions(
    sft_by_qid: dict[int, dict],
    rl_by_qid: dict[int, dict],
) -> None:
    sft_qids = set(sft_by_qid)
    rl_qids = set(rl_by_qid)

    if sft_qids == rl_qids:
        return

    missing_in_rl = sorted(sft_qids - rl_qids)
    missing_in_sft = sorted(rl_qids - sft_qids)
    raise ValueError(
        "SFT/RL qid mismatch: "
        f"missing_in_rl={missing_in_rl}, missing_in_sft={missing_in_sft}"
    )


def zero_success_upper_bound(n_samples: int, alpha: float = 0.05) -> float:
    """Exact one-sided upper bound on Bernoulli success probability after 0/n."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")

    return 1.0 - alpha ** (1.0 / n_samples)


def build_per_question_rows(
    sft_rows: list[dict],
    rl_rows: list[dict],
) -> list[dict]:
    sft_by_qid = validate_rows(sft_rows, "SFT")
    rl_by_qid = validate_rows(rl_rows, "RL")
    validate_same_questions(sft_by_qid, rl_by_qid)

    output = []

    for qid in sorted(sft_by_qid):
        sft_row = sft_by_qid[qid]
        rl_row = rl_by_qid[qid]

        sft_n = int(sft_row["n_rollouts"])
        sft_correct = int(sft_row["n_correct"])
        rl_n = int(rl_row["n_rollouts"])
        rl_correct = int(rl_row["n_correct"])

        if sft_correct == 0:
            upper_bound = zero_success_upper_bound(sft_n)
        else:
            upper_bound = float("nan")

        output.append(
            {
                "qid": qid,
                "question": sft_row.get("question", ""),
                "gold": sft_row.get("gold", ""),
                "sft_successes_full": sft_correct,
                "sft_rollouts_full": sft_n,
                "sft_success_rate_full": sft_correct / sft_n,
                "rl_successes": rl_correct,
                "rl_rollouts": rl_n,
                "rl_success_rate": rl_correct / rl_n,
                "rl_observed_success": rl_correct > 0,
                "persistent_unobserved_sft_success": (
                    sft_correct == 0 and rl_correct > 0
                ),
                "sft_zero_success_95pct_upper_bound": upper_bound,
            }
        )

    return output


def summarize_samples(values: np.ndarray) -> tuple[float, float, float]:
    return (
        float(values.mean()),
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    )


def simulate_depth_curve(
    sft_rows: list[dict],
    rl_rows: list[dict],
    k_values: list[int],
    repeats: int,
    seed: int,
) -> list[dict]:
    if repeats <= 0:
        raise ValueError("repeats must be positive")

    sft_by_qid = validate_rows(sft_rows, "SFT")
    rl_by_qid = validate_rows(rl_rows, "RL")
    validate_same_questions(sft_by_qid, rl_by_qid)

    qids = sorted(sft_by_qid)
    if not qids:
        raise ValueError("No questions found")

    rollout_counts = {int(sft_by_qid[qid]["n_rollouts"]) for qid in qids}
    if len(rollout_counts) != 1:
        raise ValueError(
            "All SFT questions must have the same rollout count; "
            f"found {sorted(rollout_counts)}"
        )

    max_available = rollout_counts.pop()
    normalized_k_values = sorted(set(int(k) for k in k_values))

    if not normalized_k_values or normalized_k_values[0] <= 0:
        raise ValueError("k_values must contain positive integers")
    if normalized_k_values[-1] > max_available:
        raise ValueError(
            f"Requested K={normalized_k_values[-1]} but only "
            f"{max_available} SFT rollouts are available"
        )

    correctness = {
        qid: np.asarray(
            [bool(rollout["correct"]) for rollout in sft_by_qid[qid]["rollouts"]],
            dtype=np.int8,
        )
        for qid in qids
    }
    rl_success = {
        qid: int(rl_by_qid[qid]["n_correct"]) > 0
        for qid in qids
    }
    n_rl_success_questions = sum(rl_success.values())

    covered_by_k = {
        k: np.empty(repeats, dtype=np.int32)
        for k in normalized_k_values
    }
    unresolved_by_k = {
        k: np.empty(repeats, dtype=np.int32)
        for k in normalized_k_values
    }

    rng = np.random.default_rng(seed)

    for repeat_index in range(repeats):
        covered_counts = {k: 0 for k in normalized_k_values}
        unresolved_counts = {k: 0 for k in normalized_k_values}

        for qid in qids:
            permutation = rng.permutation(max_available)
            ordered_correctness = correctness[qid][permutation]
            cumulative_successes = np.cumsum(ordered_correctness)

            for k in normalized_k_values:
                observed_sft_success = cumulative_successes[k - 1] > 0

                if observed_sft_success:
                    covered_counts[k] += 1
                elif rl_success[qid]:
                    unresolved_counts[k] += 1

        for k in normalized_k_values:
            covered_by_k[k][repeat_index] = covered_counts[k]
            unresolved_by_k[k][repeat_index] = unresolved_counts[k]

    curve_rows = []

    for k in normalized_k_values:
        covered_mean, covered_low, covered_high = summarize_samples(covered_by_k[k])
        unresolved_mean, unresolved_low, unresolved_high = summarize_samples(
            unresolved_by_k[k]
        )

        curve_rows.append(
            {
                "k": k,
                "n_questions": len(qids),
                "n_rl_observed_success_questions": n_rl_success_questions,
                "repeats": repeats,
                "sft_covered_mean": covered_mean,
                "sft_covered_p2.5": covered_low,
                "sft_covered_p97.5": covered_high,
                "rl_success_unresolved_mean": unresolved_mean,
                "rl_success_unresolved_p2.5": unresolved_low,
                "rl_success_unresolved_p97.5": unresolved_high,
            }
        )

    return curve_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write to {path}")

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_plot(curve_rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    k_values = [row["k"] for row in curve_rows]
    covered = [row["sft_covered_mean"] for row in curve_rows]
    unresolved = [row["rl_success_unresolved_mean"] for row in curve_rows]

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(k_values, covered, marker="o", label="SFT questions with observed success")
    axis.plot(
        k_values,
        unresolved,
        marker="o",
        label="RL-success questions still unresolved by SFT",
    )
    axis.set_xscale("log", base=2)
    axis.set_xticks(k_values)
    axis.set_xticklabels([str(k) for k in k_values])
    axis.set_xlabel("SFT sampling budget K")
    axis.set_ylabel("Expected number of questions")
    axis.set_title("Observed reachability as SFT sampling depth increases")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def print_summary(curve_rows: list[dict], per_question_rows: list[dict]) -> None:
    print("\nSampling-depth reachability")
    print("=" * 80)
    print(
        f"{'K':>5} {'SFT covered':>20} "
        f"{'RL-success unresolved':>26}"
    )
    print("-" * 80)

    for row in curve_rows:
        covered = (
            f"{row['sft_covered_mean']:.2f} "
            f"[{row['sft_covered_p2.5']:.0f}, {row['sft_covered_p97.5']:.0f}]"
        )
        unresolved = (
            f"{row['rl_success_unresolved_mean']:.2f} "
            f"[{row['rl_success_unresolved_p2.5']:.0f}, "
            f"{row['rl_success_unresolved_p97.5']:.0f}]"
        )
        print(f"{row['k']:>5} {covered:>20} {unresolved:>26}")

    persistent = [
        row
        for row in per_question_rows
        if row["persistent_unobserved_sft_success"]
    ]

    print(
        "\nPersistent cases at full SFT depth "
        f"(SFT 0/full, RL > 0): {len(persistent)}"
    )

    if persistent:
        for row in persistent:
            print(
                f"  qid={row['qid']:02d} "
                f"SFT=0/{row['sft_rollouts_full']} "
                f"RL={row['rl_successes']}/{row['rl_rollouts']} "
                f"95% upper p_SFT={100 * row['sft_zero_success_95pct_upper_bound']:.2f}%"
            )


def parse_k_values(text: str) -> list[int]:
    try:
        values = [int(piece.strip()) for piece in text.split(",") if piece.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "K values must be comma-separated integers"
        ) from error

    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("K values must be positive integers")

    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sft-file",
        type=Path,
        default=Path("results_sft_256/sft_raw.jsonl"),
    )
    parser.add_argument(
        "--rl-file",
        type=Path,
        default=Path("results_2048_batched/rl_raw.jsonl"),
    )
    parser.add_argument(
        "--k-values",
        type=parse_k_values,
        default=DEFAULT_K_VALUES,
        help="Comma-separated SFT sampling budgets.",
    )
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--curve-csv",
        type=Path,
        default=Path("analyses/reachability_depth.csv"),
    )
    parser.add_argument(
        "--per-question-csv",
        type=Path,
        default=Path("analyses/reachability_depth_per_question.csv"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("figures/reachability_depth.png"),
    )
    args = parser.parse_args()

    sft_rows = read_jsonl(args.sft_file)
    rl_rows = read_jsonl(args.rl_file)

    per_question_rows = build_per_question_rows(sft_rows, rl_rows)
    curve_rows = simulate_depth_curve(
        sft_rows=sft_rows,
        rl_rows=rl_rows,
        k_values=args.k_values,
        repeats=args.repeats,
        seed=args.seed,
    )

    write_csv(args.curve_csv, curve_rows)
    write_csv(args.per_question_csv, per_question_rows)
    make_plot(curve_rows, args.figure)
    print_summary(curve_rows, per_question_rows)

    print(f"\nWrote: {args.curve_csv}")
    print(f"Wrote: {args.per_question_csv}")
    print(f"Wrote: {args.figure}")


if __name__ == "__main__":
    main()
