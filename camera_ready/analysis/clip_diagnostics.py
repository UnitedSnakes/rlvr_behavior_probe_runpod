"""POST HOC clipping diagnostics for the three box pairs (seeds 43-45). Not pre-registered.

Specified 2026-11-15, after RESULTS.md / analysis.json were known. Uses only the existing
group ledgers and the per-step pre-clip `grad_norm` logs; no retraining; writes two new files
and changes no existing result file.

    ../rlvr_behavior_probe_runpod/.venv/bin/python -m camera_ready.analysis.clip_diagnostics

Order of work: (0) reproduce the raw and clip-weighted mass in analysis.json, both through the
existing pipeline code and through this module's own implementation; abort without writing any
output if either disagrees beyond float rounding. Then (1) bin shares and mass-weighted mean
clip coefficients, (2) split by number of live groups per step (and clipped / not clipped),
(3) clip coefficient and clip share by the step's smallest positive K and live-group count.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

from camera_ready.analysis import core, pair as P

HERE = Path(__file__).resolve().parent
CR = HERE.parent
DATA = CR / "data"
RESULTS = CR / "results"
SEEDS = (43, 44, 45)
ARMS = P.ARMS  # ("grpo", "maxrl")
N_STEPS = 3736
EVAL_T = [core.PAPER_SCHEDULE[p] for p in core.EVAL_PCTS]  # 934, 1681, 2428, 3736
ENDPOINT = 3736
MAX_NORM = 1.0
G = core.GROUP_SIZE
PANEL_SET = frozenset(core.PANEL)  # panel index == dataset_index, as in core.mass_by_bin
REL_TOL = 1e-9
FEW, VERY_FEW = 30, 10
CELLS = {
    "live1": lambda a, s: a["n_live"][s] == 1,
    "live2": lambda a, s: a["n_live"][s] == 2,
    "live1_clipped": lambda a, s: a["n_live"][s] == 1 and a["clipped"][s],
    "live1_unclipped": lambda a, s: a["n_live"][s] == 1 and not a["clipped"][s],
    "live2_clipped": lambda a, s: a["n_live"][s] == 2 and a["clipped"][s],
    "live2_unclipped": lambda a, s: a["n_live"][s] == 2 and not a["clipped"][s],
}


def flag(n: int) -> str | None:
    return f"n<{VERY_FEW}" if n < VERY_FEW else (f"n<{FEW}" if n < FEW else None)


def ratio(num: float, den: float) -> float | None:
    return num / den if den > 0 else None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).resolve().open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- loading

def load_arm(root: Path, arm: str) -> dict:
    ledger = core.load_ledger(root / arm / "ledger")
    log = core.read_jsonl(root / arm / "step_log.jsonl")
    gn: dict[int, float] = {}
    for rec in log:
        if "grad_norm" in rec and rec.get("step") is not None:
            t = int(rec["step"])
            if t in gn:
                raise ValueError(f"{root}/{arm}: duplicate grad_norm for optimizer step {t}")
            gn[t] = float(rec["grad_norm"])
    if sorted(gn) != list(range(1, N_STEPS + 1)):
        raise ValueError(f"{root}/{arm}: grad_norm not logged at exactly optimizer steps 1..{N_STEPS}")
    steps = sorted(ledger.by_step)
    if steps != list(range(N_STEPS)):
        raise ValueError(f"{root}/{arm}: generation steps are not 0..{N_STEPS - 1}")
    # mapping check: the group generated at generation step s is consumed by optimizer step s+1
    rew_log = {int(r["step"]): float(r["reward"]) for r in log if "reward" in r and "grad_norm" in r}
    map_diff = max(
        abs(float(np.mean(np.concatenate([ledger.groups[k]["rewards"] for k in ledger.by_step[s]]))) - rew_log[s + 1])
        for s in steps
    )
    live = {k: 0 < g["K"] < G for k, g in ledger.groups.items()}
    live_mismatch = sum(live[k] != g["live"] for k, g in ledger.groups.items())
    nonlive_mass = sum(g["abs_adv"] for k, g in ledger.groups.items() if not live[k])
    n_live = {s: sum(live[k] for k in keys) for s, keys in ledger.by_step.items()}
    min_pos_k = {s: min((ledger.groups[k]["K"] for k in keys if ledger.groups[k]["K"] > 0), default=None)
                 for s, keys in ledger.by_step.items()}
    min_live_k = {s: min((ledger.groups[k]["K"] for k in keys if live[k]), default=None)
                  for s, keys in ledger.by_step.items()}
    minK_mismatch = sum(1 for s in steps if n_live[s] >= 1 and min_pos_k[s] != min_live_k[s])
    group_of: dict[int, tuple[int, int]] = {}
    for key in ledger.groups:
        if key[1] in group_of:
            raise ValueError(f"{root}/{arm}: question {key[1]} sampled twice")
        group_of[key[1]] = key
    return {
        "ledger": ledger,
        "gn": gn,
        "steps": steps,
        # user definition; grad_norm == 0 occurs only at steps with no live group -> c = 1 (limit of min(1, 1/g))
        "c": {s: 1.0 if gn[s + 1] == 0.0 else min(1.0, MAX_NORM / gn[s + 1]) for s in steps},
        "c_eps": {s: min(1.0, MAX_NORM / (gn[s + 1] + 1e-6)) for s in steps},   # analysis.json definition
        "clipped": {s: gn[s + 1] > MAX_NORM for s in steps},
        "n_live": n_live,
        "min_pos_k": min_pos_k,
        "group_of": group_of,
        "checks": {
            "step_mapping_max_abs_reward_diff": map_diff,
            "live_definition_mismatches_vs_analysis_json": live_mismatch,
            "mass_in_non_live_groups": nonlive_mass,
            "min_positive_K_differs_from_min_live_K": minK_mismatch,
            "grad_norm_exactly_1": sum(1 for v in gn.values() if v == MAX_NORM),
            "grad_norm_exactly_0": sum(1 for v in gn.values() if v == 0.0),
            "zero_norm_steps_with_a_live_group": sum(1 for s in steps if gn[s + 1] == 0.0 and n_live[s] > 0),
            "steps_without_live_group_and_nonzero_norm": sum(1 for s in steps if n_live[s] == 0 and gn[s + 1] != 0.0),
        },
        "log": log,
    }


# ---------------------------------------------------------------- bin-level mass

def bin_cells(arm: dict, bins, T: int, weight: str = "raw", select=None) -> dict:
    """Per bin: per-direction sums of |A| x weight over panel groups generated before T that pass
    `select`, and the symmetric per-question mean S_b (the analysis.json mass convention)."""
    W = {"raw": None, "clip": arm["c"], "clip_eps": arm["c_eps"]}[weight]
    groups = arm["ledger"].groups
    out = {}
    for b in core.FIVE_BINS:
        dirs = {}
        union: set[int] = set()
        for d in core.DIRECTIONS:
            m = core.members(bins, d, b)
            if not m:
                raise ValueError(f"empty bin {b} in direction {d}")
            tot, ng, st = 0.0, 0, set()
            for i in m:
                key = arm["group_of"].get(i)
                if key is None or key[0] >= T:
                    continue
                s = key[0]
                a = groups[key]["abs_adv"]
                if a == 0.0 or (select is not None and not select(arm, s)):
                    continue
                tot += a * (1.0 if W is None else W[s])
                ng += 1
                st.add(s)
                union.add(i)
            dirs[d] = {"sum": tot, "n_members": len(m), "n_groups": ng, "n_steps": len(st)}
        out[b] = {"S": 0.5 * sum(dirs[d]["sum"] / dirs[d]["n_members"] for d in core.DIRECTIONS), "dirs": dirs,
                  "n_union": len(union)}
    return out


def totals(arm: dict, T: int, weight: str) -> dict:
    W = {"raw": None, "clip": arm["c"], "clip_eps": arm["c_eps"]}[weight]
    alltrain = panel = 0.0
    for (s, i), g in arm["ledger"].groups.items():
        if s >= T:
            continue
        v = g["abs_adv"] * (1.0 if W is None else W[s])
        alltrain += v
        if i in PANEL_SET:
            panel += v
    return {"all_training_groups": alltrain, "panel_groups": panel}


# ---------------------------------------------------------------- reproduction check

def leaf_diffs(ref, new, path=""):
    if isinstance(ref, dict):
        for k in ref:
            yield from leaf_diffs(ref[k], new[k], f"{path}/{k}")
    else:
        r, n = float(ref), float(new)
        yield path, abs(n - r) / max(abs(r), 1e-300)


def reproduce(name: str, root: Path, arms: dict, bins, A: dict) -> dict:
    ref = A["pairs"][name]
    pr = P.Pair(name=name, seed=int(name[4:]), root=root, pcts=core.EVAL_PCTS)
    pr.ledgers = {arm: arms[arm]["ledger"] for arm in ARMS}
    pr.step_logs = {arm: arms[arm]["log"] for arm in ARMS}
    raw_pipe = {str(k): v for k, v in P.mass_table(pr, bins, EVAL_T).items()}
    cw_pipe = {str(k): v for k, v in P.clip_weighted_mass(pr, bins, EVAL_T).items()}
    raw_own, cw_own = {}, {}
    for T in EVAL_T:
        cr = {arm: bin_cells(arms[arm], bins, T, "raw") for arm in ARMS}
        ce = {arm: bin_cells(arms[arm], bins, T, "clip_eps") for arm in ARMS}
        tot = {arm: totals(arms[arm], T, "raw")["all_training_groups"] for arm in ARMS}
        raw_own[str(T)] = {
            b: {"S_G": cr["grpo"][b]["S"], "S_M": cr["maxrl"][b]["S"],
                "mass_ratio": cr["maxrl"][b]["S"] / cr["grpo"][b]["S"],
                "share_ratio": (cr["maxrl"][b]["S"] / tot["maxrl"]) / (cr["grpo"][b]["S"] / tot["grpo"])}
            for b in core.FIVE_BINS
        }
        raw_own[str(T)]["_total"] = tot
        cw_own[str(T)] = {b: {"S_G": ce["grpo"][b]["S"], "S_M": ce["maxrl"][b]["S"],
                              "ratio": ce["maxrl"][b]["S"] / ce["grpo"][b]["S"]} for b in core.FIVE_BINS}
    clip_pipe = {arm: P.clipping_summary(P.clip_series(arms[arm]["log"])) for arm in ARMS}
    live_overall = {
        arm: {str(c): sum(1 for s in arms[arm]["steps"] if arms[arm]["n_live"][s] == c) / N_STEPS for c in (0, 1, 2)}
        for arm in ARMS
    }
    comparisons = {
        "raw_mass__pipeline_code": (ref["mass"], raw_pipe),
        "raw_mass__own_code": (ref["mass"], raw_own),
        "clip_weighted_mass__pipeline_code": (ref["clip_weighted_mass"], cw_pipe),
        "clip_weighted_mass__own_code": (ref["clip_weighted_mass"], cw_own),
        "clipping_summary__pipeline_code": ({a: {k: ref["clipping"][a][k] for k in ("n_steps_logged", "share_clipped", "mean")} for a in ARMS},
                                            {a: {k: clip_pipe[a][k] for k in ("n_steps_logged", "share_clipped", "mean")} for a in ARMS}),
        "live_step_shares__own_code": ({a: ref["batch_structure"][a]["overall"] for a in ARMS}, live_overall),
    }
    out = {}
    for label, (r, n) in comparisons.items():
        d = list(leaf_diffs(r, n))
        worst = max(d, key=lambda x: x[1])
        out[label] = {"n_values": len(d), "max_rel_diff": worst[1], "worst_field": worst[0],
                      "n_above_tol": sum(1 for _, v in d if v > REL_TOL)}
    out["status"] = "PASS" if all(v["n_above_tol"] == 0 for v in out.values()) else "FAIL"
    return out


# ---------------------------------------------------------------- analyses 1-3

def shares_block(arms: dict, bins, T: int) -> dict:
    cells = {(arm, w): bin_cells(arms[arm], bins, T, w) for arm in ARMS for w in ("raw", "clip")}
    tots = {(arm, w): totals(arms[arm], T, w) for arm in ARMS for w in ("raw", "clip")}
    out = {"totals": {f"{arm}_{w}": tots[(arm, w)] for arm in ARMS for w in ("raw", "clip")}, "bins": {}}
    for b in core.FIVE_BINS:
        blk = {}
        for w in ("raw", "clip"):
            S = {arm: cells[(arm, w)][b]["S"] for arm in ARMS}
            v1 = {arm: S[arm] / tots[(arm, w)]["all_training_groups"] for arm in ARMS}
            v2 = {arm: 0.5 * sum(cells[(arm, w)][b]["dirs"][d]["sum"] / tots[(arm, w)]["panel_groups"]
                                 for d in core.DIRECTIONS) for arm in ARMS}
            blk[w] = {
                "S_G": S["grpo"], "S_M": S["maxrl"], "mass_ratio_M_over_G": ratio(S["maxrl"], S["grpo"]),
                "share_V1_G": v1["grpo"], "share_V1_M": v1["maxrl"],
                "share_ratio_V1_M_over_G": ratio(v1["maxrl"], v1["grpo"]),
                "share_V2_G": v2["grpo"], "share_V2_M": v2["maxrl"],
                "share_ratio_V2_M_over_G": ratio(v2["maxrl"], v2["grpo"]),
            }
        cbar = {}
        for arm in ARMS:
            raw, clip = cells[(arm, "raw")][b], cells[(arm, "clip")][b]
            per_dir = [ratio(clip["dirs"][d]["sum"], raw["dirs"][d]["sum"]) for d in core.DIRECTIONS]
            cbar[arm] = {
                "pooled": ratio(clip["S"], raw["S"]),
                "per_direction_mean": None if None in per_dir else 0.5 * sum(per_dir),
            }
        n = {arm: {d: cells[(arm, "raw")][b]["dirs"][d]["n_groups"] for d in core.DIRECTIONS} for arm in ARMS}
        nu = {arm: cells[(arm, "raw")][b]["n_union"] for arm in ARMS}
        blk.update({"cbar": cbar, "n_groups_per_direction": n,
                    "n_steps_per_direction": {arm: {d: cells[(arm, "raw")][b]["dirs"][d]["n_steps"] for d in core.DIRECTIONS} for arm in ARMS},
                    "n_groups_union": nu, "n_min": min(nu.values()), "flag": flag(min(nu.values()))})
        # identity check: clip mass ratio = raw mass ratio x cbar_M / cbar_G (pooled cbar)
        lhs = blk["clip"]["mass_ratio_M_over_G"]
        rhs = blk["raw"]["mass_ratio_M_over_G"] * cbar["maxrl"]["pooled"] / cbar["grpo"]["pooled"]
        blk["identity_rel_diff"] = abs(lhs - rhs) / abs(lhs)
        out["bins"][b] = blk
    # V2 remainder (bin "1" and anything outside the five bins), per arm and weight
    out["share_V2_outside_five_bins"] = {
        f"{arm}_{w}": 1.0 - sum(out["bins"][b][w][f"share_V2_{arm[0].upper()}"] for b in core.FIVE_BINS)
        for arm in ARMS for w in ("raw", "clip")
    }
    return out


def step_type_block(arms: dict, bins, T: int) -> dict:
    base = {(arm, w): bin_cells(arms[arm], bins, T, w) for arm in ARMS for w in ("raw", "clip")}
    out = {}
    for cname, sel in CELLS.items():
        cc = {(arm, w): bin_cells(arms[arm], bins, T, w, sel) for arm in ARMS for w in ("raw", "clip")}
        out[cname] = {}
        for b in core.FIVE_BINS:
            S = {(arm, w): cc[(arm, w)][b]["S"] for arm in ARMS for w in ("raw", "clip")}
            frac = {}
            for arm in ARMS:
                bd, cd = base[(arm, "raw")][b]["dirs"], cc[(arm, "raw")][b]["dirs"]
                per_dir = [ratio(cd[d]["sum"], bd[d]["sum"]) for d in core.DIRECTIONS]
                frac[arm] = {"pooled": ratio(S[(arm, "raw")], base[(arm, "raw")][b]["S"]),
                             "per_direction_mean": None if None in per_dir else 0.5 * sum(per_dir)}
            n = {arm: {d: cc[(arm, "raw")][b]["dirs"][d]["n_groups"] for d in core.DIRECTIONS} for arm in ARMS}
            nu = {arm: cc[(arm, "raw")][b]["n_union"] for arm in ARMS}
            nmin = min(nu.values())
            out[cname][b] = {
                "raw_S_G": S[("grpo", "raw")], "raw_S_M": S[("maxrl", "raw")],
                "clip_S_G": S[("grpo", "clip")], "clip_S_M": S[("maxrl", "clip")],
                "raw_ratio_M_over_G": ratio(S[("maxrl", "raw")], S[("grpo", "raw")]),
                "clip_ratio_M_over_G": ratio(S[("maxrl", "clip")], S[("grpo", "clip")]),
                "frac_of_bin_raw_mass": frac,
                "n_groups_per_direction": n,
                "n_steps_per_direction": {arm: {d: cc[(arm, "raw")][b]["dirs"][d]["n_steps"] for d in core.DIRECTIONS} for arm in ARMS},
                "n_groups_union": nu, "n_min": nmin, "flag": flag(nmin),
            }
    # additivity check: live1 + live2 raw S = bin raw S
    worst = 0.0
    for arm in ARMS:
        for b in core.FIVE_BINS:
            tot = base[(arm, "raw")][b]["S"]
            parts = out["live1"][b][f"raw_S_{arm[0].upper()}"] + out["live2"][b][f"raw_S_{arm[0].upper()}"]
            worst = max(worst, abs(parts - tot) / tot)
    out["_additivity_max_rel_diff"] = worst
    return out


def lowk_block(arm_list: list[dict]) -> dict:
    out = {}
    for nl in (0, 1, 2):
        rows = {}
        keys = [None] if nl == 0 else list(range(1, G)) + ["all"]
        for k in keys:
            cs, cl = [], []
            for a in arm_list:
                for s in a["steps"]:
                    if a["n_live"][s] != nl:
                        continue
                    if k not in (None, "all") and a["min_pos_k"][s] != k:
                        continue
                    cs.append(a["c"][s])
                    cl.append(a["clipped"][s])
            rows[str(k if k is not None else "all")] = {
                "n_steps": len(cs),
                "mean_c": float(np.mean(cs)) if cs else None,
                "share_clipped": float(np.mean(cl)) if cl else None,
                "flag": flag(len(cs)),
            }
        out[f"live{nl}"] = rows
    return out


# ---------------------------------------------------------------- markdown

def f2(x, nd=2):
    return "–" if x is None else f"{x:.{nd}f}"


def trio(vals, nd=2):
    return " / ".join(f2(v, nd) for v in vals)


def mk(blocks) -> str:
    n = min(x["n_min"] for x in blocks)
    return " ‡" if n < VERY_FEW else (" †" if n < FEW else "")


def ns(blocks, arm) -> str:
    return " / ".join(str(x["n_groups_union"][arm]) for x in blocks)


def render_md(R: dict) -> str:
    names = [f"seed{s}" for s in SEEDS]
    E = str(ENDPOINT)
    L = []
    L.append("# Clipping diagnostics (POST HOC)\n")
    L.append("**POST HOC — not pre-registered.** Specified on 2026-11-15, after `RESULTS.md` and `analysis.json` were known. "
             "Descriptive numbers only; nothing here is a causal estimate, and no pre-registered sentence or result file is changed. "
             "Scope: the three box pairs (seeds 43–45), GRPO and MaxRL arms, existing group ledgers and per-step pre-clip `grad_norm` logs; no retraining. "
             f"All numbers: `clip_diagnostics.json`. Code: `camera_ready/analysis/clip_diagnostics.py` (run {R['generated_utc']}, HEAD `{R['git_head'][:7]}`).\n")
    L.append("## Definitions used\n")
    L.append("- Clip coefficient of optimizer step t: c_t = min(1, 1.0 / grad_norm_t); a step is *clipped* iff grad_norm_t > 1.0 (strict). "
             "A group generated at ledger step s is weighted by c_{s+1}. Check: the ledger's mean reward at step s equals the logged reward at optimizer step s+1 "
             "at every step of every run (max |diff| = "
             f"{max(R['checks'][n][a]['step_mapping_max_abs_reward_diff'] for n in names for a in ARMS):.1e}); offsets 0 and 2 do not match.")
    L.append("- Live group: 0 < K < 16. Identical to `analysis.json`'s `advantage != 0` in every group of every run; non-live groups carry zero |A|.")
    L.append("- Bins: the frozen A40 cross-fit bins of `analysis.json`. Bin mass S_b = average over the two directions of the within-bin mean, over panel questions, "
             "of cumulative |A| (groups generated before step T) — the `analysis.json` convention. Clip-weighted mass replaces |A| by |A|·c.")
    L.append("- Cell counts n = number of distinct panel groups with non-zero mass in the cell, per arm, over both directions "
             "(each panel question is sampled at one step, so groups = steps; per-direction counts in the JSON). "
             f"A cell is marked † if the smaller arm has n < {FEW} and ‡ if n < {VERY_FEW} (thresholds chosen here; see ambiguity 7).\n")

    rep = R["reproduction_check"]
    L.append(f"## Reproduction check: {'PASS' if all(rep[n]['status'] == 'PASS' for n in names) else 'FAIL'}\n")
    worst = max((v["max_rel_diff"], n, k) for n in names for k, v in rep[n].items() if isinstance(v, dict))
    L.append("Raw and clip-weighted mass, mass ratios, share ratios and totals in `analysis.json` (4 eval steps × 5 bins × 3 pairs) were recomputed twice — "
             "with the existing pipeline functions and with this module's own code (using the pipeline's `c = min(1, 1/(g + 1e-6))`) — together with the "
             "clipping summaries and live-step shares. "
             f"Largest relative difference over all values: {worst[0]:.1e} ({worst[1]}, {worst[2]}); tolerance {REL_TOL:.0e}. "
             f"Switching to this note's c (no 1e-6) changes any clip-weighted ratio by at most {R['eps_effect_max_rel_on_clip_ratio']:.1e} (relative).\n")

    # ---- 1
    L.append("## 1. Bin shares (step 3736) — values for seed 43 / 44 / 45\n")
    L.append("Share ratio V1 = (S_b^M / total^M) / (S_b^G / total^G), total over all training groups before T (the `analysis.json` `share_ratio` convention; "
             "for clip-weighted shares the totals are clip-weighted). c̄_b = mass-weighted mean clip coefficient, pooled form S_b^clip / S_b^raw.\n")
    L.append("| Bin | Raw share ratio M/G | Clip-wtd share ratio M/G | c̄_b GRPO | c̄_b MaxRL | n GRPO | n MaxRL |")
    L.append("|---|---|---|---|---|---|---|")
    for b in core.FIVE_BINS:
        blks = [R["pairs"][n]["shares"][E]["bins"][b] for n in names]
        L.append(f"| {b}{mk(blks)} | {trio([x['raw']['share_ratio_V1_M_over_G'] for x in blks])} | {trio([x['clip']['share_ratio_V1_M_over_G'] for x in blks])} | "
                 f"{trio([x['cbar']['grpo']['pooled'] for x in blks], 3)} | {trio([x['cbar']['maxrl']['pooled'] for x in blks], 3)} | "
                 f"{ns(blks, 'grpo')} | {ns(blks, 'maxrl')} |")
    L.append("\nClip-weighted share ratio V1 at each evaluated step (seed 43 / 44 / 45):\n")
    L.append("| Bin | " + " | ".join(f"T = {T}" for T in EVAL_T) + " |")
    L.append("|---|" + "---|" * len(EVAL_T))
    for b in core.FIVE_BINS:
        L.append(f"| {b} | " + " | ".join(trio([R["pairs"][n]["shares"][str(T)]["bins"][b]["clip"]["share_ratio_V1_M_over_G"] for n in names]) for T in EVAL_T) + " |")
    L.append("\nRaw share ratios, shares themselves, mass ratios, the V2 share variant and the per-direction c̄ variant at every T are in the JSON (`pairs.*.shares`).\n")

    # ---- 2
    L.append("## 2. Split by live groups per step (step 3736) — seed 43 / 44 / 45\n")
    L.append("Type is each arm's own step type (ambiguity 4). Fraction = cell raw mass / bin raw mass, pooled form, per arm. Ratios are M/G of the cell's symmetric mass.\n")
    L.append("| Bin | Cell | Frac of bin raw mass GRPO | Frac MaxRL | Raw ratio M/G | Clip-wtd ratio M/G | n GRPO | n MaxRL |")
    L.append("|---|---|---|---|---|---|---|---|")
    for b in core.FIVE_BINS:
        for cname in CELLS:
            xs = [R["pairs"][n]["step_type"][E][cname][b] for n in names]
            L.append(f"| {b} | {cname}{mk(xs)} | {trio([x['frac_of_bin_raw_mass']['grpo']['pooled'] for x in xs])} | "
                     f"{trio([x['frac_of_bin_raw_mass']['maxrl']['pooled'] for x in xs])} | {trio([x['raw_ratio_M_over_G'] for x in xs])} | "
                     f"{trio([x['clip_ratio_M_over_G'] for x in xs])} | {ns(xs, 'grpo')} | {ns(xs, 'maxrl')} |")
    L.append("\nA ratio of 0.00 means MaxRL has no mass in the cell; – means GRPO has none. "
             "The mark after the bin or cell name refers to the smallest n among the three seeds. Values at the other evaluated steps: JSON `pairs.*.step_type`.\n")

    # ---- 3
    P3 = R["pooled_over_seeds"]["low_K"]
    L.append("## 3. Clip coefficient by the step's smallest positive K (all 3,736 steps per run, three seeds pooled)\n")
    L.append("For steps with ≥ 1 live group the smallest positive K equals the smallest live K (checked in every run). Per-seed tables: JSON `pairs.*.low_K`.\n")
    for nl in (1, 2):
        L.append(f"**{nl} live group{'s' if nl == 2 else ''} in the step**\n")
        L.append("| min K | n GRPO | n MaxRL | mean c GRPO | mean c MaxRL | clipped GRPO | clipped MaxRL |")
        L.append("|---|---:|---:|---:|---:|---:|---:|")
        for k in [str(k) for k in range(1, G)] + ["all"]:
            g, m = P3["grpo"][f"live{nl}"][k], P3["maxrl"][f"live{nl}"][k]
            mark = lambda r: str(r["n_steps"]) + ("†" if r["flag"] else "")
            pct = lambda r: "–" if r["share_clipped"] is None else f"{100 * r['share_clipped']:.1f} %"
            L.append(f"| {k} | {mark(g)} | {mark(m)} | {f2(g['mean_c'], 3)} | {f2(m['mean_c'], 3)} | {pct(g)} | {pct(m)} |")
        L.append("")
    g0, m0 = P3["grpo"]["live0"]["all"], P3["maxrl"]["live0"]["all"]
    L.append(f"Steps with no live group (no mass): GRPO {g0['n_steps']} (mean c {f2(g0['mean_c'], 3)}), MaxRL {m0['n_steps']} (mean c {f2(m0['mean_c'], 3)}).\n")

    # ---- flags
    L.append("## Cells with few groups / steps (all evaluated steps; every cell's counts are in the JSON)\n")
    def tally(blocks):
        blocks = list(blocks)
        return (len(blocks), sum(1 for x in blocks if x["flag"]), sum(1 for x in blocks if x["flag"] == f"n<{VERY_FEW}"))
    t1 = tally(R["pairs"][n]["shares"][str(T)]["bins"][b] for n in names for T in EVAL_T for b in core.FIVE_BINS)
    L.append(f"- Section 1 (bin × step × seed): {t1[1]} of {t1[0]} cells n < {FEW}, {t1[2]} n < {VERY_FEW}; by bin at T = 3736: "
             + ", ".join(f"{b} n = {min(R['pairs'][n]['shares'][E]['bins'][b]['n_min'] for n in names)}–{max(R['pairs'][n]['shares'][E]['bins'][b]['n_min'] for n in names)}"
                         for b in core.FIVE_BINS) + " (smaller arm).")
    for cname in CELLS:
        t = tally(R["pairs"][n]["step_type"][str(T)][cname][b] for n in names for T in EVAL_T for b in core.FIVE_BINS)
        L.append(f"- Section 2, {cname}: {t[1]} of {t[0]} cells n < {FEW}, {t[2]} n < {VERY_FEW}.")
    k3 = [(a, nl, k, r["n_steps"]) for a in ARMS for nl in (1, 2) for k, r in P3[a][f"live{nl}"].items() if r["flag"]]
    k3s = [(n, a, nl, k, r["n_steps"]) for n in names for a in ARMS for nl in (1, 2)
           for k, r in R["pairs"][n]["low_K"][a][f"live{nl}"].items() if r["flag"]]
    L.append(f"- Section 3: pooled, {len(k3)} of {2 * 2 * 16} arm × live × K cells n < {FEW}; per seed, {len(k3s)} of {3 * 2 * 2 * 16} "
             f"(smallest per-seed n = {min((x[4] for x in k3s), default='–')}).\n")

    # ---- ambiguities
    L.append("## Definitional ambiguities (not resolved here)\n")
    for i, a in enumerate(R["ambiguities"], 1):
        L.append(f"{i}. {a}")
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--analysis-json", type=Path, default=RESULTS / "analysis.json")
    ap.add_argument("--out-json", type=Path, default=RESULTS / "clip_diagnostics.json")
    ap.add_argument("--out-md", type=Path, default=RESULTS / "clip_diagnostics.md")
    args = ap.parse_args(argv)
    for p in (args.out_json, args.out_md):
        if p.exists():
            raise SystemExit(f"refusing to overwrite existing {p}")

    A = json.loads(args.analysis_json.read_text())
    bank_dir = args.data / "banks" / "a40_original"
    bins = core.frozen_bins(core.load_p0_bank(bank_dir))
    R: dict = {
        "label": "POST HOC - not pre-registered; descriptive only; specified 2026-11-15 after all camera-ready results were known",
        "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=CR).stdout.strip(),
        "code": "camera_ready/analysis/clip_diagnostics.py",
        "code_sha256": sha256(Path(__file__)),
        "inputs": {"analysis_json_sha256": sha256(args.analysis_json),
                   "bank": {p.name: sha256(p) for p in sorted(bank_dir.glob("rollouts_*.jsonl"))}},
        "definitions": {
            "c_step": "c_t = min(1, 1.0 / grad_norm_t), grad_norm_t = logged pre-clip total norm at optimizer step t",
            "clipped": "grad_norm_t > 1.0 (strict)",
            "step_mapping": "group generated at ledger generation_global_step s is weighted by c_{s+1}",
            "live_group": "0 < K < 16",
            "bin_mass": "S_b(T) = mean over directions A,B of the within-bin mean over panel questions of cumulative |A| (generation step < T); "
                        "clip-weighted: |A| * c. Bins: frozen A40 cross-fit bins (analysis.json convention).",
            "share_V1": "S_b / total |A| (or |A|c) over all training groups before T  [analysis.json share_ratio convention]",
            "share_V2": "mean over directions of (sum of |A| over the bin's panel groups) / (sum over all panel groups); sums to 1 over the six bins",
            "cbar_pooled": "S_b^clip / S_b^raw",
            "cbar_per_direction_mean": "mean over directions of (sum |A|c / sum |A|) within the bin",
            "frac_of_bin_raw_mass_pooled": "S_b,cell^raw / S_b^raw",
            "frac_of_bin_raw_mass_per_direction_mean": "mean over directions of (cell sum / bin sum)",
            "step_type": "each arm's own number of live groups in the step that generated the group",
            "low_K_stratum": "smallest positive K among the step's two groups; strata 1..15 within live count 1 or 2; all 3,736 steps of each run",
            "eval_T": EVAL_T,
            "flags": f"n<{VERY_FEW}, n<{FEW} (n = contributing groups; for low_K, steps)",
        },
        "checks": {}, "reproduction_check": {}, "pairs": {},
    }
    loaded = {}
    eps_effect = 0.0
    for seed in SEEDS:
        name, root = f"seed{seed}", args.data / f"seed{seed}"
        arms = {arm: load_arm(root, arm) for arm in ARMS}
        R["inputs"][name] = {arm: {"ledger": {p.name: sha256(p) for p in sorted((root / arm / "ledger").glob("*.jsonl"))},
                                   "step_log": sha256(root / arm / "step_log.jsonl")} for arm in ARMS}
        R["checks"][name] = {arm: arms[arm]["checks"] for arm in ARMS}
        rep = reproduce(name, root, arms, bins, A)
        R["reproduction_check"][name] = rep
        print(name, "reproduction", rep["status"], {k: v["max_rel_diff"] for k, v in rep.items() if isinstance(v, dict)})
        if rep["status"] != "PASS":
            print(json.dumps(rep, indent=1))
            raise SystemExit("reproduction check FAILED; nothing written")
        bad = {a: c for a, c in R["checks"][name].items()
               if c["step_mapping_max_abs_reward_diff"] > 1e-9 or c["live_definition_mismatches_vs_analysis_json"]
               or c["mass_in_non_live_groups"] != 0.0 or c["min_positive_K_differs_from_min_live_K"]
               or c["zero_norm_steps_with_a_live_group"]}
        if bad:
            raise SystemExit(f"structural check failed for {name}: {bad}; nothing written")
        for T in EVAL_T:
            ce ={arm: bin_cells(arms[arm], bins, T, "clip_eps") for arm in ARMS}
            cu = {arm: bin_cells(arms[arm], bins, T, "clip") for arm in ARMS}
            for b in core.FIVE_BINS:
                re_ = ce["maxrl"][b]["S"] / ce["grpo"][b]["S"]
                ru = cu["maxrl"][b]["S"] / cu["grpo"][b]["S"]
                eps_effect = max(eps_effect, abs(ru - re_) / re_)
        loaded[name] = arms
        R["pairs"][name] = {
            "shares": {str(T): shares_block(arms, bins, T) for T in EVAL_T},
            "step_type": {str(T): step_type_block(arms, bins, T) for T in EVAL_T},
            "low_K": {arm: lowk_block([arms[arm]]) for arm in ARMS},
            "step_level": {arm: {"n_steps": N_STEPS,
                                 "share_clipped_strict": float(np.mean(list(arms[arm]["clipped"].values()))),
                                 "mean_c": float(np.mean(list(arms[arm]["c"].values())))} for arm in ARMS},
        }
    R["eps_effect_max_rel_on_clip_ratio"] = eps_effect
    R["pooled_over_seeds"] = {"low_K": {arm: lowk_block([loaded[n][arm] for n in loaded]) for arm in ARMS},
                              "note": "steps pooled across the three seeds (each run contributes its 3,736 steps)"}
    worst_identity = max(blk["identity_rel_diff"] for n in R["pairs"] for T in EVAL_T for blk in R["pairs"][n]["shares"][str(T)]["bins"].values())
    worst_add = max(R["pairs"][n]["step_type"][str(T)]["_additivity_max_rel_diff"] for n in R["pairs"] for T in EVAL_T)
    R["internal_consistency"] = {"clip_ratio_equals_raw_ratio_times_cbar_ratio_max_rel_diff": worst_identity,
                                 "live1_plus_live2_equals_bin_mass_max_rel_diff": worst_add}
    # magnitude of the alternative definitions (so the choice can be judged without opening every cell)
    def maxdiff(get_a, get_b):
        vals = []
        for n in R["pairs"]:
            for T in EVAL_T:
                for b in core.FIVE_BINS:
                    x, y = get_a(n, str(T), b), get_b(n, str(T), b)
                    if x is not None and y is not None:
                        vals.append(abs(x - y))
        return max(vals)
    S_ = lambda n, T, b: R["pairs"][n]["shares"][T]["bins"][b]
    ST = lambda n, T, c, b: R["pairs"][n]["step_type"][T][c][b]
    R["alternative_definition_effects"] = {
        "share_ratio_V2_minus_V1_raw_max_abs": maxdiff(lambda n, T, b: S_(n, T, b)["raw"]["share_ratio_V2_M_over_G"], lambda n, T, b: S_(n, T, b)["raw"]["share_ratio_V1_M_over_G"]),
        "share_ratio_V2_minus_V1_clip_max_abs": maxdiff(lambda n, T, b: S_(n, T, b)["clip"]["share_ratio_V2_M_over_G"], lambda n, T, b: S_(n, T, b)["clip"]["share_ratio_V1_M_over_G"]),
        "cbar_per_direction_minus_pooled_max_abs": max(
            maxdiff(lambda n, T, b, a=a: S_(n, T, b)["cbar"][a]["per_direction_mean"], lambda n, T, b, a=a: S_(n, T, b)["cbar"][a]["pooled"]) for a in ARMS),
        "frac_per_direction_minus_pooled_max_abs": max(
            maxdiff(lambda n, T, b, a=a, c=c: ST(n, T, c, b)["frac_of_bin_raw_mass"][a]["per_direction_mean"],
                    lambda n, T, b, a=a, c=c: ST(n, T, c, b)["frac_of_bin_raw_mass"][a]["pooled"]) for a in ARMS for c in CELLS),
    }
    alt = R["alternative_definition_effects"]
    R["ambiguities"] = [
        "Share denominator. V1 (shown in the note) divides the symmetric per-question bin mass by the arm's total over *all* training groups before T, "
        "as `analysis.json`'s `share_ratio` does; V2 uses the fraction of the *panel's* mass, averaged over directions (sums to 1 over the six bins). "
        f"Largest |V2 − V1| in any share ratio: raw {alt['share_ratio_V2_minus_V1_raw_max_abs']:.3f}, clip-weighted {alt['share_ratio_V2_minus_V1_clip_max_abs']:.3f}. Both in the JSON.",
        "Cross-fit averaging of derived ratios (c̄_b, fraction of bin mass). Pooled form = ratio of the two symmetric masses (shown; gives clip ratio = raw ratio × c̄^M/c̄^G exactly); "
        "alternative = mean over directions of the per-direction ratios. Largest difference: c̄ "
        f"{alt['cbar_per_direction_minus_pooled_max_abs']:.3f}, fraction {alt['frac_per_direction_minus_pooled_max_abs']:.3f}. Both in the JSON.",
        "Cumulative vs interval mass. All T values are cumulative from step 0 (as in `analysis.json`); mass accrued only between evaluated steps is not computed.",
        "Step type is arm-specific: the same question can sit in a 1-live step in one arm and a 2-live step in the other, so the M/G ratio within a type compares "
        "partly different question sets. A joint classification (by both arms' types, or by GRPO's type only) is not computed.",
        "Clipped / not-clipped sub-split in section 2 was added so that the stated prediction (1-live and clipped steps) can be read; the request listed only the 1-vs-2 split.",
        "Section 3 uses all 3,736 steps of each run (whole training set, no bins); restricting to steps that consumed a panel question would give far fewer steps. "
        "Pooling over seeds is done by pooling steps; per-seed tables are in the JSON.",
        f"Small-cell thresholds ({VERY_FEW} and {FEW}) are this note's choice; every cell reports its counts.",
        "The pipeline's `c = min(1, 1/(g + 1e-6))` (the form torch's clip_grad_norm_ applies) differs from this note's c only at the 1e-6 level; "
        "the strict `> 1.0` rule excludes the steps whose logged norm is exactly 1.0 (count per run in `checks`). Logged norms are coarsely quantized (values such as 1.3515625).",
        "grad_norm = 0 makes 1.0/grad_norm undefined. It occurs at exactly the steps with no live group (33–52 per run; checked both ways), which carry no mass; "
        "c is set to 1 there (the limit of min(1, 1/g), and the factor torch applies). It affects only the 'no live group' row of section 3, whose mean c is 1 by this rule.",
    ]

    args.out_json.write_text(json.dumps(R, indent=1, allow_nan=False) + "\n")
    args.out_md.write_text(render_md(R) + "\n")
    print("wrote", args.out_json, args.out_md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
