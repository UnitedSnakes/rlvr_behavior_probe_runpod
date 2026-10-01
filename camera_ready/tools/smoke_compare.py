"""D3/D4 smoke checks: box smoke ledger (seed 42, first N steps) vs the A40 seed-42 ledger.

    python smoke_compare.py --box-ledger <dir> --a40-ledger <dir> --objective grpo --steps 40 \
        [--box-step-log <jsonl>] [--a40-trainer-log <maxrl stdout log>]

Metrics over the groups of generation steps 0..N-1: mean R, T, C, completion length,
cap-hit rate (1 - T), share of live groups. Units = prompt groups (clustered SE:
SD of group means / sqrt(#groups)); the two runs are compared as independent samples,
alert if |diff| > 3 SE. Also checks the question order and the structural audit.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from camera_ready.analysis import core  # noqa: E402


def group_table(ledger: core.Ledger, steps: int) -> dict:
    keys = sorted(k for k in ledger.groups if k[0] < steps)
    g = [ledger.groups[k] for k in keys]
    return {
        "order": [k for k in keys],
        "R": np.array([x["R"] for x in g]),
        "T": np.array([x["T"] for x in g]),
        "C": np.array([x["C"] for x in g]),
        "len": np.array([x["mean_len"] for x in g]),
        "cap": np.array([x["cap_hits"] / x["n"] for x in g]),
        "live": np.array([float(x["live"]) for x in g]),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--box-ledger", type=Path, required=True)
    ap.add_argument("--a40-ledger", type=Path, required=True)
    ap.add_argument("--objective", choices=("grpo", "maxrl"), required=True)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--box-step-log", type=Path)
    ap.add_argument("--a40-trainer-log", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    box = core.load_ledger(args.box_ledger)
    a40 = core.load_ledger(args.a40_ledger)
    n_box_steps = len(box.by_step)
    steps = min(args.steps, n_box_steps)
    tb, ta = group_table(box, steps), group_table(a40, steps)
    res = {"objective": args.objective, "steps_compared": steps, "box_steps_present": n_box_steps}

    # question order: (step, dataset_index) of every group
    res["question_order_equal"] = [k for k in tb["order"]] == [k for k in ta["order"]]
    # partial structural audit on the steps present
    max_err = 0.0
    for key, g in box.groups.items():
        exp = core.expected_advantages(g["rewards"], args.objective)
        max_err = max(max_err, float(np.max(np.abs(g["advantages"] - exp))))
    res["box_audit"] = {
        "rows": box.rows,
        "groups": len(box.groups),
        "rows_per_step_rank_ok": all(v == 16 for v in box.step_rank_counts.values()),
        "nonfinite": box.nonfinite_numeric_fields,
        "max_advantage_error": max_err,
        "token_is_ess_fraction": box.token_ratio_sum ** 2 / box.token_ratio_sq_sum / box.token_ratio_count,
    }
    res["metrics"] = {}
    alerts = []
    for m in ("R", "T", "C", "len", "cap", "live"):
        xb, xa = tb[m], ta[m]
        se = math.sqrt(xb.var(ddof=1) / xb.size + xa.var(ddof=1) / xa.size)
        diff = float(xb.mean() - xa.mean())
        z = diff / se if se > 0 else 0.0
        res["metrics"][m] = {"box": float(xb.mean()), "a40": float(xa.mean()), "diff": diff, "se": se, "z": z}
        if abs(z) > 3:
            alerts.append(m)
    res["alerts_gt_3se"] = alerts

    if args.box_step_log and args.box_step_log.is_file():
        recs = {r["step"]: r for r in map(json.loads, open(args.box_step_log)) if "learning_rate" in r}
        res["box_lr"] = {s: recs[s].get("learning_rate") for s in sorted(recs) if s % 10 == 0}
        res["box_peak_mem_gib"] = max((r.get("cr_max_memory_allocated_gib") or 0) for r in recs.values())
        res["box_peak_reserved_gib"] = max((r.get("cr_max_memory_reserved_gib") or 0) for r in recs.values())
        res["box_s_per_step"] = float(np.mean([r["step_time"] for r in recs.values() if "step_time" in r]))
    if args.a40_trainer_log and args.a40_trainer_log.is_file():
        import ast
        import re

        text = args.a40_trainer_log.read_text(errors="replace")
        a40_recs = [ast.literal_eval(m.group(0)) for m in re.finditer(r"\{'loss': [^{}]*\}", text)]
        res["a40_lr"] = {10 * (k + 1): float(r["learning_rate"]) for k, r in enumerate(a40_recs[: steps // 10])}
        res["a40_s_per_step_first"] = [float(r["step_time"]) for r in a40_recs[: steps // 10]]
        if "box_lr" in res:
            res["lr_equal"] = all(
                math.isclose(float(res["box_lr"][s]), res["a40_lr"][s], rel_tol=2e-3) for s in res["a40_lr"] if s in res["box_lr"]
            )
    out = json.dumps(res, indent=1, default=float)
    if args.out:
        args.out.write_text(out)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
