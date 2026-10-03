"""Assemble camera_ready/RESULTS.md from JSON outputs plus the hand-maintained notes in camera_ready/results/notes/.

    python -m camera_ready.analysis.results --analysis camera_ready/results/analysis.json \
        --bridge camera_ready/results/bridge.json --F camera_ready/results/seed42_F.json --out camera_ready/RESULTS.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from camera_ready.analysis import render as R

NOTES = Path(__file__).resolve().parent.parent / "results" / "notes"


def note(name: str) -> str:
    p = NOTES / name
    return p.read_text() if p.is_file() else f"*({name} not written yet)*\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--bridge", type=Path, required=True)
    ap.add_argument("--F", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    A = json.loads(args.analysis.read_text())
    B = json.loads(args.bridge.read_text())
    F = json.loads(args.F.read_text())
    parts = [
        "# Camera-ready replication results\n",
        note("00_status.md"),
        "## 1. Bridge (evaluation compatibility)\n",
        f"Verdict: **{B['verdict']}** over {len(B['checkpoints'])} checkpoint(s). Full tables: `BRIDGE_REPORT.md`.\n",
        note("10_validity.md"),
        "## 2. Main sample: primary cells\n",
        R.results_tables(A),
        R.wording_sentences(A),
        R.mixed_table(A),
        "## 3. Secondary results\n",
        R.secondary_tables(A),
        R.panel_table(A),
        R.clipping_table(A),
        R.discovery_sensitivity(A),
        "## 4. Seed-42 discovery run: post-outcome analyses (F)\n",
        R.seed42_F_md(F),
        "## 5. Deviations and unresolved alerts\n",
        note("50_deviations.md"),
        "## 6. What in the paper needs to change\n",
        note("60_paper_changes.md"),
    ]
    args.out.write_text("\n".join(parts))
    print("written", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
