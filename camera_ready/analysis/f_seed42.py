"""Section F of PREREG_RUNS.md on the existing seed-42 A40 data.

Post-outcome descriptive analyses of the submitted discovery run.

    python -m camera_ready.analysis.f_seed42 --out camera_ready/results/seed42_F.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from camera_ready.analysis import core, pair as P

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
MAXRL_LOG = Path.home() / "rlvr_data/camera_ready_meta/rlvr-behavior-probe-maxrl-analysis-seed42-2026-09-05/logs/maxrl_canonical_seed42.log"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--draws", type=int, default=3000)
    args = ap.parse_args(argv)

    bank = core.load_p0_bank(DATA / "banks/a40_original")
    bins = core.frozen_bins(bank)
    pr = P.load_pair("seed42_a40", 42, DATA / "seed42_a40", pcts=sorted(core.PAPER_SCHEDULE))
    res: dict = {"label": "post-outcome descriptive analysis of the submitted discovery run (seed 42, A40)"}

    # F1 (all 20 snapshots) and F2 (point estimates, all snapshots; EVAL_STEPS are the reported rows)
    res["F1_F2"] = {}
    for pct in sorted(core.PAPER_SCHEDULE):
        point = P.contrasts(pr.snaps["maxrl"][pct], pr.snaps["grpo"][pct], bins)
        boot = P.bootstrap_dq(pr.snaps["maxrl"][pct], pr.snaps["grpo"][pct], bins, draws=args.draws)
        res["F1_F2"][pct] = {
            m: {
                b: {
                    "d": point[m]["d"][b],
                    "q": point[m]["q"][b],
                    "d_ci": boot[m][b]["d_ci"],
                    "q_ci": boot[m][b]["q_ci"],
                }
                for b in core.FIVE_BINS
            }
            | {"panel_gap": point[m]["panel_gap"]}
            for m in core.METRICS
        }
        print(f"F1 {pct}% done", flush=True)

    # F3
    res["F3"] = {str(k): v for k, v in P.mass_table(pr, bins, [core.PAPER_SCHEDULE[p] for p in core.EVAL_PCTS]).items()}
    # F4
    res["F4"] = {arm: P.batch_structure(pr.ledgers[arm], bins) for arm in P.ARMS}
    # F5 (A40 banks: the paper's baselines)
    res["F5"] = P.bin_contents(bins, bank)
    # Clipping: MaxRL seed 42 only, 1-in-10 logged steps; GRPO logs not kept
    if MAXRL_LOG.is_file():
        recs = P.parse_hf_trainer_stdout(MAXRL_LOG)
        res["clipping"] = {
            "maxrl": P.clipping_summary(P.clip_series(recs)) | {"note": "every 10th optimizer step only"},
            "grpo": {"note": "seed-42 GRPO training logs were not kept; not available"},
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1, default=float))
    print("written", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
