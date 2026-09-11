"""Compare GRPO learning-signal budget under three truncation policies.

Runs entirely offline on the rollouts already saved by
diagnose_p0_signal_budget.py. No GPU, no regeneration.

A GRPO group produces gradient only if its rewards have nonzero variance.
The three policies differ in how a truncated completion is treated:

  A  mask_truncated_completions=true   (current frozen config)
     truncated rollouts are dropped from the loss entirely.
     Variance is computed over survivors only.

  B  mask_truncated_completions=false, reward 0 on truncation
     every rollout contributes. Truncation becomes a learnable
     negative signal. Reward = correct AND terminated.

  C  mask_truncated_completions=false, reward = correctness as scored
     every rollout contributes, and a truncated rollout that still
     emitted a parseable correct answer keeps reward 1.
     This is the raw p0 interior and the upper bound on live rate.

The question is whether A's dead groups are a real property of the task
or an artifact of the masking rule.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

BIN_EDGES = [
    ("0", lambda p: p <= 0.0),
    ("(0, 1/4]", lambda p: 0.0 < p <= 0.25),
    ("(1/4, 1/2]", lambda p: 0.25 < p <= 0.50),
    ("(1/2, 3/4]", lambda p: 0.50 < p <= 0.75),
    ("(3/4, 1)", lambda p: 0.75 < p < 1.0),
    ("1", lambda p: p >= 1.0),
]


def assign_bin(p: float) -> str:
    for label, test in BIN_EDGES:
        if test(p):
            return label
    return "(3/4, 1)"


def live(rewards) -> bool:
    """A group has nonzero advantage iff its rewards are not all equal."""
    return len(set(rewards)) > 1


def pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rollouts",
        nargs="+",
        required=True,
        help="one or more rollouts_shard*.jsonl files",
    )
    args = parser.parse_args()

    records = []
    for path in args.rollouts:
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                records.append(json.loads(line))

    print(f"loaded {len(records)} prompt groups from {len(args.rollouts)} file(s)\n")

    rows = []
    flipped_total = 0
    n_rollouts = 0

    for rec in records:
        rolls = rec["rollouts"]
        n_rollouts += len(rolls)

        correct = [bool(r["correct"]) for r in rolls]
        trunc = [bool(r["truncated"]) for r in rolls]

        # correct-but-truncated: these flip 1 -> 0 under policy B
        flipped = sum(c and t for c, t in zip(correct, trunc))
        flipped_total += flipped

        survivors = [c for c, t in zip(correct, trunc) if not t]
        reward_a = survivors
        reward_b = [c and not t for c, t in zip(correct, trunc)]
        reward_c = correct

        rows.append({
            "p0_raw": sum(correct) / len(correct),
            "bin": assign_bin(sum(correct) / len(correct)),
            "n_survivors": len(survivors),
            "flipped": flipped,
            "live_a": bool(survivors) and live(reward_a),
            "live_b": live(reward_b),
            "live_c": live(reward_c),
            "p_b": sum(reward_b) / len(reward_b),
        })

    n = len(rows)

    print("=== OVERALL LIVE-GROUP RATE ===")
    for key, label in [
        ("live_a", "A  mask truncated (current)      "),
        ("live_b", "B  no mask, reward 0 on truncate "),
        ("live_c", "C  no mask, reward as scored     "),
    ]:
        k = sum(r[key] for r in rows)
        print(f"  {label} {k:>4}/{n} = {pct(k, n):.2f}%")

    print(f"\n  correct-but-truncated rollouts: {flipped_total}/{n_rollouts} = "
          f"{pct(flipped_total, n_rollouts):.2f}%")
    print("  (these lose reward 1 under policy B — the cost of the change)")

    print("\n=== LIVE RATE BY p0 BIN ===")
    print(f"  {'bin':<12} {'n':>4} {'A%':>8} {'B%':>8} {'C%':>8}   {'B-A':>7}")
    for label, _ in BIN_EDGES:
        group = [r for r in rows if r["bin"] == label]
        if not group:
            continue
        a = pct(sum(r["live_a"] for r in group), len(group))
        b = pct(sum(r["live_b"] for r in group), len(group))
        c = pct(sum(r["live_c"] for r in group), len(group))
        print(f"  {label:<12} {len(group):>4} {a:>7.2f}% {b:>7.2f}% {c:>7.2f}%   "
              f"{b - a:>+6.1f}")

    print("\n=== p0 SHIFT UNDER POLICY B ===")
    print("  (reward 0 on truncation lowers effective p0; check the bins survive)")
    shifted = Counter(assign_bin(r["p_b"]) for r in rows)
    original = Counter(r["bin"] for r in rows)
    print(f"  {'bin':<12} {'raw n':>7} {'policy B n':>11}")
    for label, _ in BIN_EDGES:
        print(f"  {label:<12} {original.get(label, 0):>7} {shifted.get(label, 0):>11}")

    interior_b = sum(1 for r in rows if 0.0 < r["p_b"] < 1.0)
    print(f"\n  interior under B: {interior_b}/{n} = {pct(interior_b, n):.2f}%")


if __name__ == "__main__":
    main()
