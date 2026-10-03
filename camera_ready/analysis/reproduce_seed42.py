"""Gate A4: reproduce the submitted paper's numbers from the seed-42 raw data.

Checks (a) the rounded numbers listed in the camera-ready brief (Tables 1, 2, 4,
5 and Appendix A) and (b) exact agreement with the paper bundle exports
(objective_comparison.csv, exposure_adjusted_symmetric.csv, *_aggregate.csv)
at all snapshots. Exits non-zero on any mismatch.

    python -m camera_ready.analysis.reproduce_seed42 --bundle ~/Downloads/attrib_draft_v5_source
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from camera_ready.analysis import core

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"

PAPER = {
    "table1": {"initial": (34.91, 45.62, 50.45), "GRPO": (53.03, 77.25, 57.74), "MaxRL": (51.54, 74.73, 56.40)},
    "table4": {"A": (33, 89, 57, 56, 20, 1), "B": (37, 86, 59, 55, 19, 0)},
    "table2": {
        "mass_ratio": (2.21, 1.72, 0.99, 0.59, 0.46),
        "dC_grpo": (5.22, 9.99, 8.68, 4.96, 1.76),
        "dC_maxrl": (0.50, 9.56, 6.77, 4.34, 1.60),
        "d": (-4.71, -0.44, -1.91, -0.62, -0.16),
        "q": (-3.37, 0.91, -0.56, 0.72, 1.19),
    },
    "table5": {25: (6.25, 10.02, 3.77), 45: (7.42, 12.80, 5.38), 65: (10.90, 12.80, 1.90)},
    "appendixA": {"rows": 119552, "groups": 7472, "maxrl_adv_err": 1.59e-7, "maxrl_ess": 0.99805},
}


def load_pair(pair_dir: Path, pcts) -> dict:
    out = {}
    for arm in ("grpo", "maxrl"):
        out[arm] = {
            pct: core.load_snapshot(
                pair_dir / arm / "eval" / f"pi_{pct:03d}" / "snapshot_raw.jsonl",
                expected_seed=lambda i: 42 * 100_000 + i + 75_000,
            )
            for pct in pcts
        }
    return out


def prompt_tokens() -> dict[int, float]:
    raw = json.loads((HERE / "prompt_token_counts.json").read_text())
    return {int(k): float(v) for k, v in raw.items()}


class Checker:
    def __init__(self):
        self.failures = []
        self.n = 0

    def rounded(self, label, value, expected, digits=2):
        self.n += 1
        got = round(value, digits)
        ok = abs(got - expected) < 0.5 * 10 ** (-digits) + 1e-12
        if not ok:
            self.failures.append(f"{label}: got {value!r} (rounds to {got}), paper {expected}")
        return ok

    def exact(self, label, value, expected, tol=1e-9):
        self.n += 1
        if abs(float(value) - float(expected)) > tol:
            self.failures.append(f"{label}: got {value!r}, export {expected!r}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=Path.home() / "Downloads/attrib_draft_v5_source")
    parser.add_argument("--pair-dir", type=Path, default=DATA / "seed42_a40")
    parser.add_argument("--bank", type=Path, default=DATA / "banks/a40_original")
    parser.add_argument("--pcts", type=str, default=None, help="comma-separated subset of snapshots (default: all 20)")
    args = parser.parse_args(argv)
    ck = Checker()
    pcts = sorted(core.PAPER_SCHEDULE) if args.pcts is None else sorted(int(x) for x in args.pcts.split(","))

    bank = core.load_p0_bank(args.bank)
    bins = core.frozen_bins(bank)
    snaps = load_pair(args.pair_dir, pcts)

    # Table 4
    counts = core.bin_counts(bins)
    for d in core.DIRECTIONS:
        for b, exp in zip(core.BIN_ORDER, PAPER["table4"][d]):
            ck.rounded(f"T4 {d} {b}", counts[d][b], exp, 0)

    # Table 1
    init = core.initial_panel_rates(bank)
    for m, exp in zip(core.METRICS, PAPER["table1"]["initial"]):
        ck.rounded(f"T1 initial {m}", 100 * init[m], exp)
    for arm, key in (("grpo", "GRPO"), ("maxrl", "MaxRL")):
        for m, exp in zip(core.METRICS, PAPER["table1"][key]):
            ck.rounded(f"T1 {key} {m}", 100 * snaps[arm][100].panel_rate(m), exp)

    # Ledgers and Appendix A
    ledgers = {arm: core.load_ledger(args.pair_dir / arm / "ledger") for arm in ("grpo", "maxrl")}
    audits = {arm: core.structural_audit(ledgers[arm], arm) for arm in ledgers}
    for arm in ledgers:
        ck.rounded(f"AppA {arm} rows", audits[arm]["rows"], PAPER["appendixA"]["rows"], 0)
        ck.rounded(f"AppA {arm} groups", audits[arm]["groups"], PAPER["appendixA"]["groups"], 0)
    ck.n += 1
    if f"{audits['maxrl']['max_advantage_error']:.2e}" != "1.59e-07":
        ck.failures.append(f"AppA MaxRL adv error {audits['maxrl']['max_advantage_error']}")
    ck.rounded("AppA MaxRL ESS", audits["maxrl"]["aggregate_token_is_ess_fraction"], PAPER["appendixA"]["maxrl_ess"], 5)

    # Table 2 (endpoint) and exact agreement with objective_comparison.csv at all 20 snapshots
    export = list(csv.DictReader(open(args.bundle / "data/objective_comparison.csv")))
    centered = list(csv.DictReader(open(args.bundle / "data/centered_correctness.csv")))
    for pct in pcts:
        step = core.PAPER_SCHEDULE[pct]
        mass = {arm: core.mass_by_bin(ledgers[arm], bins, step) for arm in ledgers}
        delta = {
            arm: {m: core.delta_by_bin(snaps[arm][pct], bins, bank, m) for m in core.METRICS}
            for arm in ledgers
        }
        con = core.contrast_by_bin(snaps["maxrl"][pct], snaps["grpo"][pct], bins, "C")
        for row in (r for r in export if int(r["snapshot_pct"]) == pct):
            b = row["bin"]
            ck.exact(f"S_G {pct} {b}", mass["grpo"][b], row["grpo_signal"])
            ck.exact(f"S_M {pct} {b}", mass["maxrl"][b], row["maxrl_signal"])
            for m in core.METRICS:
                ck.exact(f"dX_G {m} {pct} {b}", delta["grpo"][m][b], row[f"grpo_delta_{m}"])
                ck.exact(f"dX_M {m} {pct} {b}", delta["maxrl"][m][b], row[f"maxrl_delta_{m}"])
            ck.exact(f"d_C {pct} {b}", con["d"][b], row["maxrl_minus_grpo_delta_C"])
        for row in (r for r in centered if int(r["snapshot_pct"]) == pct):
            ck.exact(f"q_C {pct} {row['bin']}", 100 * con["q"][row["bin"]], row["bin_minus_panel_contrast_pp"], 1e-7)
        if pct == 100:
            for k, b in enumerate(core.FIVE_BINS):
                ck.rounded(f"T2 mass ratio {b}", mass["maxrl"][b] / mass["grpo"][b], PAPER["table2"]["mass_ratio"][k])
                ck.rounded(f"T2 dC GRPO {b}", 100 * delta["grpo"]["C"][b], PAPER["table2"]["dC_grpo"][k])
                ck.rounded(f"T2 dC MaxRL {b}", 100 * delta["maxrl"]["C"][b], PAPER["table2"]["dC_maxrl"][k])
                ck.rounded(f"T2 d {b}", 100 * con["d"][b], PAPER["table2"]["d"][k])
                ck.rounded(f"T2 q {b}", 100 * con["q"][b], PAPER["table2"]["q"][k])
            ck.rounded("T2 panel gap", 100 * con["panel_gap"], -1.343, 3)

    # Aggregates at all snapshots vs *_aggregate.csv
    for arm, fname in (("grpo", "grpo_aggregate.csv"), ("maxrl", "maxrl_aggregate.csv")):
        for row in csv.DictReader(open(args.bundle / "data" / fname)):
            pct = int(row["snapshot_pct"])
            if pct != 0 and pct not in pcts:
                continue
            for m in core.METRICS:
                value = init[m] if pct == 0 else snaps[arm][pct].panel_rate(m)
                ck.exact(f"agg {arm} {pct} {m}", value, row[m])

    # Table 5 and exact agreement with exposure_adjusted_symmetric.csv
    expo = core_exposure = {}
    steps = ledgers["grpo"].exposure_steps()
    tokens = prompt_tokens()
    adj_export = list(csv.DictReader(open(args.bundle / "data/exposure_adjusted_symmetric.csv")))
    for pct in core.EXPOSURE_PCTS:
        rows = core.exposure_rows(snaps["grpo"][pct], bins, bank, steps, tokens, core.PAPER_SCHEDULE[pct])
        for m in core.METRICS:
            fit = core.adjusted_exposure(rows, m)
            for row in (r for r in adj_export if int(r["snapshot_pct"]) == pct):
                b = row["bin"]
                ck.exact(f"adj {m} E {pct} {b}", fit[b]["exposed"], row[f"adjusted_delta_{m}_exposed"])
                ck.exact(f"adj {m} U {pct} {b}", fit[b]["unexposed"], row[f"adjusted_delta_{m}_unexposed"])
            if m == "C":
                e, u, g = PAPER["table5"][pct]
                f = fit["(0,.25]"]
                ck.rounded(f"T5 {pct} E", 100 * f["exposed"], e)
                ck.rounded(f"T5 {pct} U", 100 * f["unexposed"], u)
                ck.rounded(f"T5 {pct} U-E", 100 * f["gap_u_minus_e"], g)
        expo[pct] = rows

    print(json.dumps({arm: {k: v for k, v in a.items() if k != "problems"} for arm, a in audits.items()}, indent=1))
    print(f"checks: {ck.n}, failures: {len(ck.failures)}")
    for f in ck.failures[:50]:
        print("  FAIL", f)
    print("SEED42 REPRODUCTION:", "PASS" if not ck.failures else "FAIL")
    return 0 if not ck.failures else 1


if __name__ == "__main__":
    sys.exit(main())
