from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT = (
    REPO_ROOT
    / "analyses"
    / "canonical_maxrl_grpo_objective_comparison"
    / "objective_comparison.csv"
)
OUTPUT = REPO_ROOT / "figures" / "signal_vs_behavior.png"
BINS = ["0", "(0,.25]", "(.25,.5]", "(.5,.75]", "(.75,1)"]


def main() -> None:
    df = pd.read_csv(INPUT)
    df = df[df["bin"].astype(str).isin(BINS)].copy()
    df["bin"] = pd.Categorical(df["bin"].astype(str), categories=BINS, ordered=True)
    df = df.sort_values(["bin", "snapshot_pct"])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1), sharex=True)

    for bin_name in BINS:
        rows = df[df["bin"] == bin_name]
        line, = axes[0].plot(
            rows["snapshot_pct"],
            rows["maxrl_over_grpo_signal"],
            marker="o",
            markersize=3,
            linewidth=1.7,
            label=bin_name,
        )
        axes[1].plot(
            rows["snapshot_pct"],
            100.0 * rows["maxrl_minus_grpo_delta_C"],
            marker="o",
            markersize=3,
            linewidth=1.7,
            color=line.get_color(),
            label=bin_name,
        )

    axes[0].axhline(1.0, linewidth=1, linestyle="--", color="0.45")
    axes[1].axhline(0.0, linewidth=1, linestyle="--", color="0.45")

    axes[0].set_title("Where the realized RL signal moves")
    axes[0].set_ylabel("MaxRL / GRPO cumulative |A| per question")
    axes[1].set_title("Where correctness moves")
    axes[1].set_ylabel("MaxRL − GRPO Δ correctness (pp)")

    for ax in axes:
        ax.set_xlabel("Training progress (%)")
        ax.set_xlim(5, 100)
        ax.grid(alpha=0.2)

    axes[0].legend(title="pre-RL success p0", fontsize=8, title_fontsize=8)
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
