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


# ----------------------------------------------------------------------------- RESULTS tables

CELL_LABEL = {
    "d_C_0": "d, C, bin 0", "q_C_0": "q, C, bin 0", "d_C_(0,.25]": "d, C, bin (0,.25]", "q_C_(0,.25]": "q, C, bin (0,.25]",
    "d_R_0": "d, R, bin 0", "q_R_0": "q, R, bin 0", "d_R_(0,.25]": "d, R, bin (0,.25]", "q_R_(0,.25]": "q, R, bin (0,.25]",
}


def results_tables(A: dict) -> str:
    acr = A["across"]
    names = acr["pairs"]
    K = A["endpoint_k_primary"]
    L = []
    L.append(f"Main sample: {len(names)} box-trained pair(s) ({', '.join(names) or 'none'}). Endpoint K for the primary "
             f"cells: {K} responses per question (rule: 64 only if every pair has all 64). Bridge verdict: "
             f"{A['bridge_verdict']}; baselines from the {'A40' if A['baseline_bank'] == 'a40' else 'regenerated box'} π0 banks; "
             "bins: the paper's frozen A40 cross-fit bins.\n")
    L.append("### Primary cells (step 3736; MaxRL − GRPO, pp)\n")
    L.append("Two-sided 95 % t-intervals across pairs (df = n − 1; none for n < 3); no multiplicity adjustment across "
             "the eight cells. In brackets after each pair's value: its within-pair conditional sampling interval "
             "(responses resampled within question), which is not seed uncertainty.\n")
    head = "| Cell | " + " | ".join(names) + " | Mean | 95 % CI | One-sided 95 % upper | Wording (δ = 3 pp) | Discovery (A40 seed 42, K=16) |"
    L.append(head)
    L.append("|---|" + "---|" * len(names) + "---:|---|---:|---|---|")
    for key, st in acr["primary"].items():
        per = []
        for n, v in zip(names, st["values"]):
            w = st["per_pair_within_interval_pp"][n]
            per.append(f"{v:+.2f} [{w[0]:+.2f}, {w[1]:+.2f}]")
        up = "—" if st["upper95_one_sided"] is None else f"{st['upper95_one_sided']:+.2f}"
        L.append(f"| {CELL_LABEL[key]} | " + " | ".join(per) + f" | {st['mean']:+.2f} | {ci(st['ci95'])} | {up} | "
                 f"{st['wording']} | {st['discovery_value_k16']:+.2f} ({st['discovery_vs_new_range']} the new pairs' range) |")
    L.append("")
    if K == 64:
        L.append("K = 16 version of the primary cells (protocol evaluation, for continuity with the paper):\n")
        L.append("| Cell | " + " | ".join(names) + " | Mean | 95 % CI |")
        L.append("|---|" + "---:|" * len(names) + "---:|---|")
        for key, st in acr["primary"].items():
            k16 = st["k16_across"]
            L.append(f"| {CELL_LABEL[key]} | " + " | ".join(f"{v:+.2f}" for v in st["k16_values"]) + f" | {k16['mean']:+.2f} | {ci(k16['ci95'])} |")
        L.append("")
    # mass
    L.append("### Advantage mass (MaxRL/GRPO), per pair\n")
    L.append("| Pair | Step | " + " | ".join(f"{b} ratio; share ratio" for b in BINS) + " |")
    L.append("|---|---|" + "---|" * len(BINS))
    for n in names + ["discovery"]:
        res = A["pairs"][n] if n != "discovery" else A["discovery"]
        for step in ("934", "1681", "2428", "3736"):
            blk = res["mass"][step]
            L.append(f"| {n if n != 'discovery' else 'seed 42 (A40)'} | {step} | " + " | ".join(
                f"{blk[b]['mass_ratio']:.2f}; {blk[b]['share_ratio']:.2f}" for b in BINS) + " |")
    L.append(f"\nReweighting realized (ratio > 1 in bins 0 and (0,.25] at step 3736 in every new pair): "
             f"**{acr['reweighting_realized_at_3736']}**; also at all four EVAL_STEPS: {acr['reweighting_above1_all_eval_steps']}.\n")
    # pre-exposure
    L.append("### Pre-exposure gain, GRPO arm, bin (0,.25] (frozen adjustment model), pp\n")
    L.append("| Pair | Cutoff | ΔC exposed | ΔC not yet exposed | U − E | n exposed / not yet (A; B) |")
    L.append("|---|---|---:|---:|---:|---|")
    for n in names + ["discovery"]:
        res = A["pairs"][n] if n != "discovery" else A["discovery"]
        bank = A["baseline_bank"] if n != "discovery" else "a40"
        for pct in ("25", "45", "65"):
            c = res["pre_exposure"][bank][pct]["(0,.25]"] if pct in res["pre_exposure"][bank] else res["pre_exposure"][bank][int(pct)]["(0,.25]"]
            if c.get("status") != "ok":
                L.append(f"| {n} | {STEP_OF[pct]} | — | — | — | not estimable |")
                continue
            L.append(f"| {n if n != 'discovery' else 'seed 42 (A40)'} | {STEP_OF[pct]} | {PP*c['exposed']:+.2f} | {PP*c['unexposed']:+.2f} | "
                     f"{PP*c['gap_u_minus_e']:+.2f} | {c['A']['n_exposed']}/{c['A']['n_unexposed']}; {c['B']['n_exposed']}/{c['B']['n_unexposed']} |")
    pe = acr["pre_exposure_(0,.25]"]
    L.append(f"\nPre-exposure gain replicated (ΔC_U > 0 at all three cutoffs in every new seed): **{acr['pre_exposure_replicated']}**.\n")
    L.append("Random-schedule exposure contrast (supplementary; adjusted U − E in (0,.25]; across-seed mean and 95 % t-interval; "
             "estimand: the average effect, under random scheduling, of a question having entered training by the cutoff "
             "instead of a random other question taking its place — not pure self-influence, not a share of the total gain):\n")
    L.append("| Cutoff | Per seed | Mean | 95 % CI |")
    L.append("|---|---|---:|---|")
    for pct in ("25", "45", "65"):
        st = pe[pct]["U_minus_E_across"]
        L.append(f"| {STEP_OF[pct]} | " + ", ".join(f"{v:+.2f}" for v in pe[pct]["U_minus_E"]) + f" | {st['mean']:+.2f} | {ci(st['ci95'])} |")
    L.append("")
    return "\n".join(L)


def secondary_tables(A: dict) -> str:
    acr = A["across"]
    names = acr["pairs"]
    L = ["### Secondary: d_b and q_b at all EVAL_STEPS (pp; across-pair mean and 95 % t-interval; no wording rule applies)\n"]
    for m in ("C", "R", "T"):
        L.append(f"**{m}**\n")
        L.append("| Step | Stat | " + " | ".join(BINS) + " |")
        L.append("|---|---|" + "---|" * len(BINS))
        for pct in ("25", "45", "65", "100"):
            for s_ in ("d", "q"):
                cells = []
                for b in BINS:
                    st = acr["secondary"][f"{s_}_{m}_{b}_{pct}"]
                    vals = ", ".join(f"{v:+.2f}" for v in st["values"])
                    cells.append(f"{st['mean']:+.2f} {ci(st['ci95'])} ({vals})")
                L.append(f"| {STEP_OF[pct]} | {s_} | " + " | ".join(cells) + " |")
        L.append("")
    return "\n".join(L)


def panel_table(A: dict) -> str:
    names = A["across"]["pairs"]
    L = ["### Whole-panel rates (%), per pair and arm (K = 16 protocol evaluation)\n",
         "| Pair | Arm | Step | R | T | C |", "|---|---|---|---:|---:|---:|"]
    for n in names + ["discovery"]:
        res = A["pairs"][n] if n != "discovery" else A["discovery"]
        for arm in ("grpo", "maxrl"):
            for pct in ("25", "45", "65", "100"):
                p = res["panel"][arm][pct]
                L.append(f"| {n if n != 'discovery' else 'seed 42 (A40)'} | {'GRPO' if arm == 'grpo' else 'MaxRL'} | {STEP_OF[pct]} | "
                         f"{PP*p['R']:.2f} | {PP*p['T']:.2f} | {PP*p['C']:.2f} |")
    L.append("")
    return "\n".join(L)


def clipping_table(A: dict) -> str:
    names = A["across"]["pairs"]
    L = ["### Clipping diagnostics (log proxy; clipping rescales the whole batch gradient, relative weights within a batch are untouched)\n",
         "| Pair | Arm | Steps logged | Share clipped | Mean coef. | Median | 5th pct |", "|---|---|---:|---:|---:|---:|---:|"]
    for n in names:
        for arm, c in A["pairs"][n].get("clipping", {}).items():
            if c.get("n_steps_logged"):
                L.append(f"| {n} | {arm} | {c['n_steps_logged']} | {PP*c['share_clipped']:.1f} % | {c['mean']:.3f} | "
                         f"{c['quantiles']['50']:.3f} | {c['quantiles']['5']:.3f} |")
    L.append("")
    cw = [n for n in names if "clip_weighted_mass" in A["pairs"][n] and "error" not in A["pairs"][n]["clip_weighted_mass"]]
    if cw:
        L.append("Clip-weighted |A| mass ratio MaxRL/GRPO (each group's |A| × the clip coefficient of the step that consumed it; "
                 "log proxy, not a measure of parameter contribution):\n")
        L.append("| Pair | Step | " + " | ".join(BINS) + " |")
        L.append("|---|---|" + "---:|" * len(BINS))
        for n in cw:
            for step, blk in A["pairs"][n]["clip_weighted_mass"].items():
                L.append(f"| {n} | {step} | " + " | ".join(f"{blk[b]['ratio']:.2f}" for b in BINS) + " |")
        L.append("")
    return "\n".join(L)


def discovery_sensitivity(A: dict) -> str:
    s_ = A.get("discovery_box_reevaluated") or {}
    if not s_:
        return "Discovery-pair checkpoints re-evaluated on the box: not available.\n"
    L = ["### Discovery pair: A40 evaluation vs its checkpoints re-evaluated on the box (sensitivity row), C, pp\n",
         "| Step | Stat | Source | " + " | ".join(BINS) + " |", "|---|---|---|" + "---:|" * len(BINS)]
    for pct, blk in s_.items():
        for st in ("d", "q"):
            a40 = A["discovery"]["dq_k16"][pct]["C"][st]
            L.append(f"| {STEP_OF[pct]} | {st} | A40 | " + " | ".join(f"{PP*a40[b]:+.2f}" for b in BINS) + " |")
            L.append(f"| {STEP_OF[pct]} | {st} | box | " + " | ".join(f"{PP*blk['C'][st][b]:+.2f}" for b in BINS) + " |")
    L.append("")
    return "\n".join(L)


def wording_sentences(A: dict) -> str:
    acr = A["across"]
    L = ["### Pre-registered sentences for the primary cells\n"]
    for key, st in acr["primary"].items():
        if st["ci95"] is None:
            L.append(f"- {CELL_LABEL[key]}: values {', '.join(f'{v:+.2f}' for v in st['values'])} pp across "
                     f"{st['n']} pair(s); no interval with fewer than three pairs — **{st['wording']}**.")
        else:
            L.append(f"- {CELL_LABEL[key]}: mean {st['mean']:+.2f} pp, 95 % CI {ci(st['ci95'])}, one-sided 95 % upper bound "
                     f"{st['upper95_one_sided']:+.2f} pp (n = {st['n']}) — **{st['wording']}**.")
    L.append("\nThe eight intervals carry no multiplicity adjustment.\n")
    return "\n".join(L)


def mixed_table(A: dict) -> str:
    mx = A["across"].get("mixed_supplement_k16")
    if not mx:
        return "No mixed estimate (bridge not PASS).\n"
    L = ["### Supplementary mixed estimate (bridge PASS only): A40 discovery pair pooled with the box pairs, K = 16\n",
         "Labelled as mixing the discovery run with replications across hardware. Not the pre-registered main inference; "
         "no wording rule is applied; no multiplicity adjustment.\n",
         "| Cell | Values (box pairs…, seed 42 A40) | n | Mean | 95 % CI |", "|---|---|---:|---:|---|"]
    for key, st in mx.items():
        L.append(f"| {CELL_LABEL[key]} | " + ", ".join(f"{v:+.2f}" for v in st["values"]) + f" | {st['n']} | {st['mean']:+.2f} | {ci(st['ci95'])} |")
    L.append("")
    return "\n".join(L)
