"""Render camera_ready/BRIDGE_REPORT.md from the output of camera_ready.analysis.bridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PP = 100.0
NAMES = {"pi0": "π0 (both banks, 32/question)"}


def label(name: str) -> str:
    if name in NAMES:
        return NAMES[name]
    arm, pct = name.split("_")
    step = {"25": 934, "45": 1681, "65": 2428, "100": 3736}[pct]
    return f"{'GRPO' if arm == 'grpo' else 'MaxRL'} seed 42, step {step}"


def render(r: dict, status_note: str) -> str:
    L = []
    L.append("# Hardware bridge report: RTX 4080 SUPER (AutoDL) vs stored A40 evaluations\n")
    L.append(status_note + "\n")
    L.append("Pre-registration: `camera_ready/PREREG_BRIDGE.md` (commit `556a271`, pushed 2026-10-01 05:13:41 UTC, "
             "before any box evaluation output existed). Statistics: `camera_ready/analysis/bridge.py`.\n")
    L.append("**Scope.** The bridge compares generation and scoring of the same frozen weights on the two machines. "
             "It says nothing about whether *training* on this box behaves like training on A40; the main analysis "
             "never mixes A40-trained and box-trained runs whatever the verdict.\n")
    L.append(f"## Verdict: **{r['verdict']}**\n")
    g = r["gates"]
    L.append(f"Gates: (a) pooled Δ intervals inside ±1.5 pp: {g['a']}; (b) per-cell |Δ/SE| rule: {g['b']}; "
             f"(c) arm interaction for C and R: {g['c']}; (d) ≥ 4 RL checkpoints from two arms: {g['d']}; "
             f"GROSS FAIL: {g['gross']}; all nine checkpoints bridged: {g['complete']}.\n")
    L.append("## Per-checkpoint differences (box − A40), pp\n")
    L.append("| Checkpoint | ΔR (SE) | ΔT (SE) | ΔC (SE) | max \\|Δ/SE\\| |")
    L.append("|---|---:|---:|---:|---:|")
    for n in r["checkpoints"]:
        c = r["cells"][n]
        cells = " | ".join(f"{PP*c[m]['delta']:+.2f} ({PP*c[m]['se']:.2f})" for m in ("R", "T", "C"))
        L.append(f"| {label(n)} | {cells} | {r['max_abs_z_by_checkpoint'][n]:.2f} |")
    L.append("")
    L.append("## Pooled over checkpoints (equal weights), question-bootstrap 95 % interval (5,000 draws), pp\n")
    L.append("| Metric | Pooled Δ | 95 % interval |")
    L.append("|---|---:|---:|")
    for m in ("R", "T", "C"):
        v = r["pooled"][m]
        L.append(f"| {m} | {PP*v['delta']:+.2f} | [{PP*v['ci'][0]:+.2f}, {PP*v['ci'][1]:+.2f}] |")
    L.append("")
    steps = r["interaction"]["steps"]
    if steps:
        L.append(f"## Arm interaction Δ(MaxRL) − Δ(GRPO), pooled over steps {steps}, pp\n")
        L.append("| Metric | Interaction | 95 % interval |")
        L.append("|---|---:|---:|")
        for m in ("R", "T", "C"):
            v = r["interaction"][m]
            L.append(f"| {m} | {PP*v['delta']:+.2f} | [{PP*v['ci'][0]:+.2f}, {PP*v['ci'][1]:+.2f}] |")
        L.append("")
    else:
        L.append("Arm interaction: not computable yet (no step with both arms bridged).\n")
    L.append("## Not gating: completion length, cap-hit rate, per-bin Δ\n")
    L.append("| Checkpoint | Mean length A40 → box (tokens) | Cap-hit A40 → box (%) |")
    L.append("|---|---:|---:|")
    for n in r["checkpoints"]:
        d = r["descriptives"][n]
        L.append(f"| {label(n)} | {d['mean_len_a40']:.1f} → {d['mean_len_box']:.1f} | {PP*d['cap_a40']:.2f} → {PP*d['cap_box']:.2f} |")
    L.append("")
    bins = ["0", "(0,.25]", "(.25,.5]", "(.5,.75]", "(.75,1)"]
    L.append("Per-bin ΔC (pp) on the frozen cross-fit bins (symmetric average of the two directions):\n")
    L.append("| Checkpoint | " + " | ".join(bins) + " |")
    L.append("|---|" + "---:|" * len(bins))
    for n in r["checkpoints"]:
        pb = r["descriptives"][n]["per_bin_delta"]["C"]
        L.append(f"| {label(n)} | " + " | ".join(f"{PP*pb[b]:+.2f}" for b in bins) + " |")
    L.append("")
    if "pi0" in r["checkpoints"]:
        L.append("For π0 the bins are selected by the same A40 banks whose rates are subtracted, so per-bin π0 Δ "
                 "contains regression toward the mean by construction (low bins up, high bins down); it is shown "
                 "because it was pre-registered, not as a hardware signal. Per-bank π0 Δ (pp): " +
                 ", ".join(f"bank {h}: " + ", ".join(f"{m} {PP*v:+.2f}" for m, v in r['descriptives']['pi0']['per_bank_delta'][h].items())
                           for h in ("A", "B")) + ".\n")
    L.append("## Consequence (C4)\n")
    if r["verdict"] == "PASS":
        L.append("PASS: baselines and bins are the paper's stored A40 banks. A supplementary estimate pooling the "
                 "A40 discovery pair with the box pairs at K = 16 is reported, labelled as mixing hardware.\n")
    else:
        L.append("Not PASS: baselines come from the π0 banks regenerated on this box (A-bin uses regenerated bank B "
                 "and vice versa); bin membership stays the paper's frozen membership. No pooled estimate. The "
                 "discovery pair is shown separately, plus one sensitivity row using its checkpoints as re-evaluated "
                 "on this box where available.\n")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--note", default="")
    args = ap.parse_args(argv)
    r = json.loads(args.bridge.read_text())
    args.out.write_text(render(r, args.note))
    print("written", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
