"""Markdown/LaTeX renderers for RESULTS.md and paper_candidates/ (numbers come from JSON outputs only)."""

from __future__ import annotations

import json
from pathlib import Path

PP = 100.0
BINS = ["0", "(0,.25]", "(.25,.5]", "(.5,.75]", "(.75,1)"]
STEP_OF = {"25": 934, "45": 1681, "65": 2428, "100": 3736}


def f(x, d=2, sign=True):
    if x is None:
        return "—"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def ci(c, d=2):
    return "—" if c is None else f"[{c[0]:+.{d}f}, {c[1]:+.{d}f}]"


def seed42_F_md(F: dict) -> str:
    L = ["### F. Seed-42 discovery run, post-outcome descriptive analyses (A40)\n",
         "Labelled post-outcome: these were specified in `PREREG_RUNS.md` §F before any new-seed outcome existed, "
         "after the seed-42 results were known. Intervals are conditional sampling intervals (responses "
         "resampled within question, 3,000 draws), not training-seed variability.\n"]
    for metric in ("C", "R", "T"):
        L.append(f"**F1/F2 — d_b and q_b for {metric} (pp), with conditional 95 % sampling intervals**\n")
        L.append("| Step | Stat | " + " | ".join(BINS) + " |")
        L.append("|---|---|" + "---:|" * len(BINS))
        for pct in ("25", "45", "65", "100"):
            blk = F["F1_F2"][pct][metric]
            for s in ("d", "q"):
                L.append(f"| {STEP_OF[pct]} | {s} | " + " | ".join(
                    f"{f(PP*blk[b][s])} {ci([PP*x for x in blk[b][s+'_ci']])}" for b in BINS) + " |")
        L.append("")
    L.append("**F1 across all 20 snapshots — number of snapshots whose conditional interval excludes 0 (C)**\n")
    L.append("| Stat | " + " | ".join(BINS) + " |")
    L.append("|---|" + "---:|" * len(BINS))
    for s in ("d", "q"):
        row = []
        for b in BINS:
            pos = sum(1 for pct, blk in F["F1_F2"].items() if blk["C"][b][s + "_ci"][0] > 0)
            neg = sum(1 for pct, blk in F["F1_F2"].items() if blk["C"][b][s + "_ci"][1] < 0)
            row.append(f"{pos} above / {neg} below")
        L.append(f"| {s} | " + " | ".join(row) + " |")
    L.append("\nThese counts describe dependent snapshots of one pair of trajectories and are not independent trials.\n")
    L.append("**F3 — advantage mass (MaxRL/GRPO): mass ratio; mass-share ratio**\n")
    L.append("| Step | " + " | ".join(BINS) + " |")
    L.append("|---|" + "---:|" * len(BINS))
    for step, blk in F["F3"].items():
        L.append(f"| {step} | " + " | ".join(f"{blk[b]['mass_ratio']:.2f}; {blk[b]['share_ratio']:.2f}" for b in BINS) + " |")
    tot = {k: v["_total"] for k, v in F["F3"].items()}
    L.append("\nTotal |A| over all training groups up to the step (GRPO; MaxRL): " +
             "; ".join(f"{k}: {v['grpo']:.0f}; {v['maxrl']:.0f}" for k, v in tot.items()) + ".\n")
    L.append("**F4 — batch structure: share of optimizer steps with 0 / 1 / 2 live groups**\n")
    L.append("| Arm | All steps | " + " | ".join(f"steps with a panel question in {b}" for b in BINS) + " |")
    L.append("|---|---|" + "---|" * len(BINS))
    for arm in ("grpo", "maxrl"):
        bs = F["F4"][arm]
        o = bs["overall"]
        cells = [f"{o['0']:.3f} / {o['1']:.3f} / {o['2']:.3f}"]
        for b in BINS:
            v = bs["panel_steps_by_bin"][b]
            cells.append("—" if v is None else f"{v['0']:.3f} / {v['1']:.3f} / {v['2']:.3f}")
        L.append(f"| {arm.upper() if arm == 'grpo' else 'MaxRL'} | " + " | ".join(cells) + " |")
    L.append("\nReading rule (pre-registered): with identical weights, rollouts and loss handling, a one-live-group "
             "step gives loss gradients under the two estimators that differ only by a scalar. This is a statement "
             "about the estimators, not about the realized AdamW updates of the two runs.\n")
    L.append("**F5 — bin contents (opposite bank): mean baseline C, mean baseline T, share with baseline C ≥ 0.5**\n")
    L.append("| " + " | ".join(BINS) + " |")
    L.append("|" + "---:|" * len(BINS))
    L.append("| " + " | ".join(f"{PP*F['F5'][b]['C']:.1f} %, {PP*F['F5'][b]['T']:.1f} %, {PP*F['F5'][b]['C_ge_half']:.1f} %" for b in BINS) + " |")
    L.append("")
    if "clipping" in F:
        c = F["clipping"]["maxrl"]
        L.append(f"**Clipping (seed 42).** MaxRL, every 10th optimizer step only ({c['n_steps_logged']} logged steps): "
                 f"share clipped (grad norm > 1) {PP*c['share_clipped']:.1f} %, mean clip coefficient {c['mean']:.3f}, "
                 f"median {c['quantiles']['50']:.3f}, 5th percentile {c['quantiles']['5']:.3f}. GRPO: "
                 f"{F['clipping']['grpo']['note']}.\n")
    return "\n".join(L)


if __name__ == "__main__":
    import sys

    F = json.loads(Path(sys.argv[1]).read_text())
    Path(sys.argv[2]).write_text(seed42_F_md(F))
    print("written", sys.argv[2])
