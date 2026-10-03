"""Candidate multi-seed figure for the camera-ready (paper_candidates/fig_multiseed_objective.{pdf,png}).

Rows: MaxRL/GRPO advantage-mass ratio; MaxRL - GRPO correctness contrast d_b (pp). Columns: bins 0 and (0,.25].
Colors keep the paper's bin identity (Okabe-Ito); seeds are told apart by marker and line style (shared legend).
Seed 42 (A40, discovery) is the paper's 20-snapshot trajectory (bundle export); seeds 43-45 are the box pairs at the
four evaluated steps. Each seed is its own line; the across-seed mean +/- 95 % t-interval (n = 3) is drawn at the endpoint.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "paper_candidates"
BUNDLE = Path.home() / "Downloads/attrib_draft_v5_source/data/objective_comparison.csv"
BIN_COLOR = {"0": "#CC79A7", "(0,.25]": "#0072B2"}
SEED_STYLE = {"seed43": ("o", "-"), "seed44": ("s", "--"), "seed45": ("^", "-.")}
STEPS = {"25": 934, "45": 1681, "65": 2428, "100": 3736}
SCHED = {5: 187, 10: 374, 15: 560, 20: 747, 25: 934, 30: 1121, 35: 1308, 40: 1494, 45: 1681, 50: 1868, 55: 2055,
         60: 2242, 65: 2428, 70: 2615, 75: 2802, 80: 2989, 85: 3176, 90: 3362, 95: 3549, 100: 3736}

plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "pdf.fonttype": 42})


def main() -> None:
    A = json.loads((HERE.parent / "results" / "analysis.json").read_text())
    rows = list(csv.DictReader(open(BUNDLE)))
    names = A["across"]["pairs"]
    fig, axes = plt.subplots(2, 2, figsize=(5.45, 3.9), sharex=True)
    for col, b in enumerate(("0", "(0,.25]")):
        color = BIN_COLOR[b]
        s42 = sorted((SCHED[int(r["snapshot_pct"])], float(r["maxrl_over_grpo_signal"]), 100 * float(r["maxrl_minus_grpo_delta_C"]))
                     for r in rows if r["bin"] == b)
        for row, (key, ylabel, ref) in enumerate((("ratio", r"MaxRL / GRPO advantage mass", 1.0),
                                                  ("d", r"$d_b$ for $C$ (pp)", 0.0))):
            ax = axes[row, col]
            ax.axhline(ref, color="#888888", lw=0.6, ls=":", zorder=1)
            ax.plot([s for s, *_ in s42], [v[0] if key == "ratio" else v[1] for _, *v in s42],
                    color="#555555", lw=0.9, ls=(0, (4, 2)), zorder=2)
            for n in names:
                p = A["pairs"][n]
                xs = [STEPS[k] for k in ("25", "45", "65", "100")]
                if key == "ratio":
                    ys = [p["mass"][str(STEPS[k])][b]["mass_ratio"] for k in ("25", "45", "65", "100")]
                else:
                    ys = [100 * p["dq_k16"][k]["C"]["d"][b] for k in ("25", "45", "65", "100")]
                mk, ls = SEED_STYLE[n]
                ax.plot(xs, ys, color=color, marker=mk, ls=ls, lw=1.1, ms=4, mec="white", mew=0.6, zorder=3)
            if key == "d":
                st = A["across"]["secondary"][f"d_C_{b}_100"]
                lo, hi = st["ci95"]
                ax.errorbar([3736 + 260], [st["mean"]], yerr=[[st["mean"] - lo], [hi - st["mean"]]], fmt="D",
                            color="#222222", ms=3.5, capsize=2.5, lw=0.9, zorder=4)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(f"$p_0$ bin {b}")
            if row == 1:
                ax.set_xlabel("Optimizer step")
            ax.set_xlim(0, 4250)
            ax.set_xticks([0, 934, 1681, 2428, 3736])
            ax.grid(axis="y", color="#E6E6E6", lw=0.5)
            ax.set_axisbelow(True)
    from matplotlib.lines import Line2D
    ink = "#444444"
    handles = [Line2D([], [], color=ink, marker=SEED_STYLE[n][0], ls=SEED_STYLE[n][1], lw=1.1, ms=4, mec="white",
                      mew=0.6, label=n.replace("seed", "seed ") + " (box)") for n in names]
    handles.append(Line2D([], [], color="#555555", lw=0.9, ls=(0, (4, 2)), label="seed 42 (A40, discovery)"))
    handles.append(Line2D([], [], color="#222222", marker="D", ls="none", ms=3.5,
                          label="mean of seeds 43–45, 95% t-interval"))
    handles = [handles[0], handles[3], handles[1], handles[4], handles[2]]  # column-major fill -> rows: box seeds / 42, mean
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0),
               handlelength=2.6, columnspacing=1.4)
    fig.tight_layout(h_pad=0.8, w_pad=1.2, rect=(0, 0, 1, 0.9))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_multiseed_objective.pdf")
    fig.savefig(OUT / "fig_multiseed_objective.png", dpi=200)
    print("written", OUT / "fig_multiseed_objective.pdf")


if __name__ == "__main__":
    main()
