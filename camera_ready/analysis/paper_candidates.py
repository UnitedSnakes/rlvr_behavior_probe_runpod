"""Candidate LaTeX tables for the camera-ready (camera_ready/paper_candidates/). Never overwrites paper files.

    python -m camera_ready.analysis.paper_candidates --analysis camera_ready/results/analysis.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "paper_candidates"
PP = 100.0
BINS = ["0", "(0,.25]", "(.25,.5]", "(.5,.75]", "(.75,1)"]
TEXBIN = {"0": "$0$", "(0,.25]": "$(0,.25]$", "(.25,.5]": "$(.25,.5]$", "(.5,.75]": "$(.5,.75]$", "(.75,1)": "$(.75,1)$"}


def s(x, d=2):
    return f"${x:+.{d}f}$".replace("+", "{+}") if x is not None else "---"


def interval(c, d=2):
    return "---" if c is None else f"$[{c[0]:+.{d}f}, {c[1]:+.{d}f}]$"


def primary_table(A: dict) -> str:
    acr = A["across"]
    names = acr["pairs"]
    rows = []
    for key, st in acr["primary"].items():
        stat, metric, b = key.split("_", 2)
        vals = " & ".join(s(v) for v in st["values"])
        rows.append(f"${stat}_b$ & ${metric}$ & {TEXBIN[b]} & {vals} & {s(st['mean'])} & {interval(st['ci95'])} & "
                    f"{s(st['discovery_value_k16'])} \\\\")
    cols = "lll" + "r" * len(names) + "rrr"
    head = "Stat & $X$ & Bin & " + " & ".join(f"Seed {n.replace('seed', '')}" for n in names) + " & Mean & 95\\% CI & Seed 42 (A40) \\\\"
    return "\n".join([
        "% Candidate table: primary cells, step 3736, MaxRL minus GRPO (pp). Generated; do not edit by hand.",
        f"% Endpoint K = {A['endpoint_k_primary']}; t-intervals across {len(names)} box-trained pairs; no multiplicity adjustment.",
        f"\\begin{{tabular}}{{{cols}}}", "\\toprule", head, "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", ""])


def endpoint_table(A: dict) -> str:
    """Multi-seed analogue of paper Table 2: means across box pairs with t-intervals (n >= 3) or values."""
    names = A["across"]["pairs"]
    lines = ["% Candidate multi-seed endpoint table (step 3736; d_b, q_b and Delta C at K = 16, the protocol evaluation). Generated.",
             "\\begin{tabular}{lrrrrr}", "\\toprule",
             "$p_0$ bin & $S_{\\mathrm M}/S_{\\mathrm G}$ & $\\Delta C_{\\mathrm G}$ & $\\Delta C_{\\mathrm M}$ & $d_b$ & $q_b$ \\\\",
             "\\midrule"]
    bank = A["baseline_bank"]
    for b in BINS:
        ratio = [A["pairs"][n]["mass"]["3736"][b]["mass_ratio"] for n in names]
        dg = [PP * A["pairs"][n]["delta"][bank]["grpo"]["100"]["C"][b] for n in names]
        dm = [PP * A["pairs"][n]["delta"][bank]["maxrl"]["100"]["C"][b] for n in names]
        d = A["across"]["secondary"][f"d_C_{b}_100"]
        q = A["across"]["secondary"][f"q_C_{b}_100"]
        fmt = lambda v: "/".join(f"{x:.2f}" for x in v)
        lines.append(f"{TEXBIN[b]} & {fmt(ratio)} & {fmt(dg)} & {fmt(dm)} & {s(d['mean'])} {interval(d['ci95'])} & "
                     f"{s(q['mean'])} {interval(q['ci95'])} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              f"% Columns 2-4 list per-seed values ({', '.join(names)}); d and q: mean [95% t-interval] (--- if n<3).", ""]
    return "\n".join(lines)


def preexposure_table(A: dict) -> str:
    names = A["across"]["pairs"]
    bank = A["baseline_bank"]
    lines = ["% Candidate pre-exposure table, GRPO arm, bin (0,.25], adjusted (pp). Generated.",
             "\\begin{tabular}{lrrrrrr}", "\\toprule",
             "Seed & \\multicolumn{2}{c}{25\\%} & \\multicolumn{2}{c}{45\\%} & \\multicolumn{2}{c}{65\\%} \\\\",
             " & $\\Delta C_U$ & $U-E$ & $\\Delta C_U$ & $U-E$ & $\\Delta C_U$ & $U-E$ \\\\", "\\midrule"]
    for n in names + ["discovery"]:
        res = A["pairs"][n] if n != "discovery" else A["discovery"]
        bk = bank if n != "discovery" else "a40"
        cells = []
        for pct in ("25", "45", "65"):
            c = res["pre_exposure"][bk][pct]["(0,.25]"]
            cells += [s(PP * c["unexposed"]), s(PP * c["gap_u_minus_e"])] if c.get("status") == "ok" else ["---", "---"]
        label = n.replace("seed", "") if n != "discovery" else "42 (A40)"
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", type=Path, required=True)
    args = ap.parse_args(argv)
    A = json.loads(args.analysis.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "table_primary_cells.tex").write_text(primary_table(A))
    (OUT / "table_endpoint_multiseed.tex").write_text(endpoint_table(A))
    (OUT / "table_preexposure_multiseed.tex").write_text(preexposure_table(A))
    print("written", sorted(p.name for p in OUT.iterdir()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
