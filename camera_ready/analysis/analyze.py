"""Main camera-ready analysis (PREREG_RUNS.md §7–§8) over any number of pairs.

    python -m camera_ready.analysis.analyze --pair seed43=camera_ready/data/seed43 [--pair seed44=...] \
        --bridge camera_ready/results/bridge.json --out camera_ready/results/analysis.json

Pair directories use the layout of camera_ready/analysis/pair.py. The discovery pair
(A40 seed 42) is always analysed separately from the main sample.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from camera_ready.analysis import core, pair as P, summarize as SM

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
PP = 100.0
PRIMARY = [(m, b, s) for m in ("C", "R") for b in ("0", "(0,.25]") for s in ("d", "q")]


def prompt_tokens() -> dict[int, float]:
    raw = json.loads((HERE / "prompt_token_counts.json").read_text())
    return {int(k): float(v) for k, v in raw.items()}


def analyse_pair(pr: P.Pair, bins, banks: dict, tokens, endpoint_k: int, bootstrap_draws: int) -> dict:
    out: dict = {"name": pr.name, "seed": pr.seed}
    steps = {pct: core.PAPER_SCHEDULE[pct] for pct in core.EVAL_PCTS}
    # panel rates
    out["panel"] = {
        arm: {str(pct): {m: pr.snaps[arm][pct].panel_rate(m) for m in core.METRICS} for pct in core.EVAL_PCTS}
        for arm in P.ARMS
    }
    # d, q at K=16 for all EVAL steps; endpoint K=64 if requested and available
    out["dq_k16"] = {}
    for pct in core.EVAL_PCTS:
        c = P.contrasts(pr.snaps["maxrl"][pct], pr.snaps["grpo"][pct], bins)
        out["dq_k16"][str(pct)] = {m: {"d": c[m]["d"], "q": c[m]["q"], "panel_gap": c[m]["panel_gap"]} for m in core.METRICS}
    if pr.endpoint64:
        c = P.contrasts(pr.endpoint64["maxrl"], pr.endpoint64["grpo"], bins)
        out["dq_k64_endpoint"] = {m: {"d": c[m]["d"], "q": c[m]["q"], "panel_gap": c[m]["panel_gap"]} for m in core.METRICS}
        out["panel_k64_endpoint"] = {arm: {m: pr.endpoint64[arm].panel_rate(m) for m in core.METRICS} for arm in P.ARMS}
    # within-pair conditional sampling intervals (F1 method) at the endpoint, for the K used
    end_m = pr.endpoint64["maxrl"] if endpoint_k == 64 else pr.snaps["maxrl"][100]
    end_g = pr.endpoint64["grpo"] if endpoint_k == 64 else pr.snaps["grpo"][100]
    out["endpoint_within_pair_intervals"] = P.bootstrap_dq(end_m, end_g, bins, draws=bootstrap_draws)
    out["endpoint_k"] = endpoint_k
    # Delta X per arm (both baseline choices)
    out["delta"] = {
        bname: {
            arm: {str(pct): P.deltas(pr.snaps[arm][pct], bins, bank) for pct in core.EVAL_PCTS} for arm in P.ARMS
        }
        for bname, bank in banks.items()
    }
    # mass
    if pr.ledgers:
        out["mass"] = {str(k): v for k, v in P.mass_table(pr, bins, list(steps.values())).items()}
        out["batch_structure"] = {arm: P.batch_structure(pr.ledgers[arm], bins) for arm in P.ARMS}
        out["audit"] = {arm: core.structural_audit(pr.ledgers[arm], arm) for arm in P.ARMS}
        out["pre_exposure"] = {bname: P.pre_exposure(pr, bins, bank, tokens) for bname, bank in banks.items()}
        clip = {}
        for arm in P.ARMS:
            if arm in pr.step_logs:
                clip[arm] = P.clipping_summary(P.clip_series(pr.step_logs[arm]))
        out["clipping"] = clip
        cw = P.clip_weighted_mass(pr, bins, list(steps.values()))
        if cw is not None:
            out["clip_weighted_mass"] = {str(k): v for k, v in cw.items()}
    return out


def primary_value(res: dict, metric: str, b: str, stat: str, endpoint_k: int) -> float:
    block = res["dq_k64_endpoint"] if endpoint_k == 64 else res["dq_k16"]["100"]
    return PP * block[metric][stat][b]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", action="append", default=[], help="name=dir for box pairs (seed parsed from name)")
    ap.add_argument("--discovery", type=Path, default=DATA / "seed42_a40")
    ap.add_argument("--a40-bank", type=Path, default=DATA / "banks/a40_original")
    ap.add_argument("--box-bank", type=Path, default=DATA / "bridge_box/pi0_bank")
    ap.add_argument("--bridge", type=Path, required=True, help="bridge.json from camera_ready.analysis.bridge")
    ap.add_argument("--discovery-box-evals", type=Path, default=DATA / "bridge_box")
    ap.add_argument("--draws", type=int, default=3000)
    ap.add_argument("--discovery-json", type=Path, default=None,
                    help="use a precomputed discovery result (written with --write-discovery-json) instead of recomputing")
    ap.add_argument("--write-discovery-json", type=Path, default=None,
                    help="compute only the discovery pair and write it to this file, then exit")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    tokens = prompt_tokens()
    a40_bank = core.load_p0_bank(args.a40_bank)
    bins = core.frozen_bins(a40_bank)
    banks = {"a40": a40_bank}
    if args.box_bank.is_dir():
        banks["box"] = core.load_p0_bank(args.box_bank)
    bridge = json.loads(args.bridge.read_text())
    verdict = bridge["verdict"]
    baseline_choice = "a40" if verdict == "PASS" else "box"
    result: dict = {"bridge_verdict": verdict, "baseline_bank": baseline_choice, "bins_from": "A40 original banks"}

    pairs = []
    for spec in args.pair:
        name, d = spec.split("=", 1)
        seed = int(name.replace("seed", ""))
        pairs.append(P.load_pair(name, seed, Path(d)))
    endpoint_k = 64 if pairs and all(pr.endpoint64 for pr in pairs) else 16
    result["endpoint_k_primary"] = endpoint_k
    result["pairs"] = {pr.name: analyse_pair(pr, bins, banks, tokens, endpoint_k, args.draws) for pr in pairs}

    if args.discovery_json is not None:
        result["discovery"] = json.loads(args.discovery_json.read_text())
    else:
        disc = P.load_pair("seed42_a40", 42, args.discovery)
        result["discovery"] = analyse_pair(disc, bins, {"a40": a40_bank}, tokens, 16, args.draws)
        if args.write_discovery_json is not None:
            args.write_discovery_json.write_text(json.dumps(result["discovery"], indent=1, default=float))
            print("discovery result written to", args.write_discovery_json)
            return 0

    # discovery checkpoints re-evaluated on the box (sensitivity row under NOT PASSED)
    sens = {}
    for pct in core.EVAL_PCTS:
        paths = {arm: args.discovery_box_evals / f"{arm}_seed42" / f"pi_{pct:03d}" / "snapshot_raw.jsonl" for arm in P.ARMS}
        if all(p.is_file() for p in paths.values()):
            snaps = {arm: core.load_snapshot(p, expected_seed=P.seed_rule(42)) for arm, p in paths.items()}
            c = P.contrasts(snaps["maxrl"], snaps["grpo"], bins)
            sens[str(pct)] = {m: {"d": c[m]["d"], "q": c[m]["q"]} for m in core.METRICS}
    result["discovery_box_reevaluated"] = sens

    # across-pair statistics
    names = [pr.name for pr in pairs]
    acr: dict = {"n_pairs": len(names), "pairs": names, "primary": {}, "secondary": {}}
    for m, b, s in PRIMARY:
        vals = [primary_value(result["pairs"][n], m, b, s, endpoint_k) for n in names]
        st = SM.across_pairs(vals)
        st["wording"] = SM.wording(st)
        st["discovery_value_k16"] = primary_value(result["discovery"], m, b, s, 16)
        st["discovery_vs_new_range"] = SM.inside_range(st["discovery_value_k16"], vals)
        st["per_pair_within_interval_pp"] = {
            n: [PP * x for x in result["pairs"][n]["endpoint_within_pair_intervals"][m][b][f"{s}_ci"]] for n in names
        }
        if endpoint_k == 64:
            st["k16_values"] = [primary_value(result["pairs"][n], m, b, s, 16) for n in names]
            st["k16_across"] = SM.across_pairs(st["k16_values"])
        acr["primary"][f"{s}_{m}_{b}"] = st
    for pct in core.EVAL_PCTS:
        for m in core.METRICS:
            for b in core.FIVE_BINS:
                for s in ("d", "q"):
                    vals = [PP * result["pairs"][n]["dq_k16"][str(pct)][m][s][b] for n in names]
                    acr["secondary"][f"{s}_{m}_{b}_{pct}"] = SM.across_pairs(vals)
    # reweighting realized
    def ratios(n, step):
        return {b: result["pairs"][n]["mass"][str(step)][b]["mass_ratio"] for b in ("0", "(0,.25]")}
    acr["reweighting_realized_at_3736"] = bool(names) and all(all(v > 1 for v in ratios(n, 3736).values()) for n in names)
    acr["reweighting_above1_all_eval_steps"] = bool(names) and all(
        all(v > 1 for v in ratios(n, core.PAPER_SCHEDULE[p]).values()) for n in names for p in core.EVAL_PCTS
    )
    # pre-exposure
    pe = {}
    for pct in core.EXPOSURE_PCTS:
        u = [PP * result["pairs"][n]["pre_exposure"][baseline_choice][pct]["(0,.25]"].get("unexposed", math.nan) for n in names]
        g = [PP * result["pairs"][n]["pre_exposure"][baseline_choice][pct]["(0,.25]"].get("gap_u_minus_e", math.nan) for n in names]
        pe[str(pct)] = {"dC_U": u, "U_minus_E": g, "U_minus_E_across": SM.across_pairs(g)}
    acr["pre_exposure_(0,.25]"] = pe
    acr["pre_exposure_replicated"] = bool(names) and all(
        all(v > 0 for v in pe[str(p)]["dC_U"]) for p in core.EXPOSURE_PCTS
    )
    # supplementary mixed estimate only under PASS
    if verdict == "PASS":
        mixed = {}
        for m, b, s in PRIMARY:
            vals = [primary_value(result["pairs"][n], m, b, s, 16) for n in names] + [primary_value(result["discovery"], m, b, s, 16)]
            mixed[f"{s}_{m}_{b}"] = SM.across_pairs(vals)
        acr["mixed_supplement_k16"] = mixed
    result["across"] = acr
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, default=float))
    print(f"pairs={names} endpoint_K={endpoint_k} verdict={verdict} baseline={baseline_choice}")
    for k, st in acr["primary"].items():
        ci = st["ci95"]
        print(f"  {k:16s} values={['%.2f' % v for v in st['values']]} mean={st['mean']:.2f} "
              f"ci={'—' if ci is None else '[%.2f, %.2f]' % tuple(ci)} -> {st['wording']} | discovery {st['discovery_value_k16']:.2f} ({st['discovery_vs_new_range']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
