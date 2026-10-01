"""Per-pair estimands (PREREG_RUNS.md §7, §F) for one GRPO/MaxRL pair.

A pair directory holds `{grpo,maxrl}/eval/pi_XXX/snapshot_raw.jsonl` (K=16
protocol), optional `{grpo,maxrl}/eval_extra/pi_100_b{1,2,3}/snapshot_raw.jsonl`,
`{grpo,maxrl}/ledger/*.jsonl`, and optionally `{grpo,maxrl}/step_log.jsonl`
(camera_ready_step_log.jsonl from training).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from camera_ready.analysis import core

ARMS = ("grpo", "maxrl")


@dataclass
class Pair:
    name: str
    seed: int
    root: Path
    pcts: tuple[int, ...]
    snaps: dict[str, dict[int, core.Snapshot]] = field(default_factory=dict)
    endpoint64: dict[str, core.Snapshot] = field(default_factory=dict)
    ledgers: dict[str, core.Ledger] = field(default_factory=dict)
    step_logs: dict[str, list[dict]] = field(default_factory=dict)


def seed_rule(seed: int, extra_batch: int = 0):
    return lambda i: seed * 100_000 + i + 75_000 + extra_batch * 1_000_000


def load_pair(name: str, seed: int, root: Path, pcts=core.EVAL_PCTS, load_ledgers: bool = True) -> Pair:
    pair = Pair(name=name, seed=seed, root=Path(root), pcts=tuple(pcts))
    for arm in ARMS:
        pair.snaps[arm] = {
            pct: core.load_snapshot(
                pair.root / arm / "eval" / f"pi_{pct:03d}" / "snapshot_raw.jsonl", expected_seed=seed_rule(seed)
            )
            for pct in pcts
        }
        extra_dirs = [pair.root / arm / "eval_extra" / f"pi_100_b{b}" / "snapshot_raw.jsonl" for b in (1, 2, 3)]
        if 100 in pcts and all(p.is_file() for p in extra_dirs):
            parts = [pair.snaps[arm][100]] + [
                core.load_snapshot(p, expected_seed=seed_rule(seed, b)) for b, p in zip((1, 2, 3), extra_dirs)
            ]
            pair.endpoint64[arm] = core.merge_snapshots(parts)
        if load_ledgers:
            pair.ledgers[arm] = core.load_ledger(pair.root / arm / "ledger")
        log = pair.root / arm / "step_log.jsonl"
        if log.is_file():
            pair.step_logs[arm] = core.read_jsonl(log)
    return pair


# ---------------------------------------------------------------- d, q, Delta X

def contrasts(snap_m: core.Snapshot, snap_g: core.Snapshot, bins) -> dict:
    return {m: core.contrast_by_bin(snap_m, snap_g, bins, m) for m in core.METRICS}


def deltas(snap: core.Snapshot, bins, baseline_bank: core.P0Bank) -> dict:
    return {m: core.delta_by_bin(snap, bins, baseline_bank, m) for m in core.METRICS}


# ---------------------------------------------------------------- F1 bootstrap

def _resp_matrix(snap: core.Snapshot) -> np.ndarray:
    """[Q, K, 3] array of R, T, C indicators in panel order (constant K required)."""
    ks = {snap.responses[i].k for i in core.PANEL}
    if len(ks) != 1:
        raise ValueError("bootstrap needs constant K")
    return np.stack(
        [np.stack([snap.responses[i].R, snap.responses[i].T, snap.responses[i].C], axis=1) for i in core.PANEL]
    )


def _bin_weight_matrix(bins, labels=core.FIVE_BINS) -> np.ndarray:
    """[Q, nbins]: symmetric-average weights so that x @ W gives the symmetric bin means."""
    W = np.zeros((len(core.PANEL), len(labels)))
    for j, b in enumerate(labels):
        for d in core.DIRECTIONS:
            m = core.members(bins, d, b)
            if not m:
                raise ValueError(f"empty bin {b} in direction {d}")
            W[m, j] += 0.5 / len(m)
    return W


def bootstrap_dq(snap_m, snap_g, bins, draws=3000, rng_seed=20261002, chunk=250) -> dict:
    """F1: resample each question's responses within arm; returns percentile intervals of d and q."""
    rng = np.random.default_rng(rng_seed)
    W = _bin_weight_matrix(bins)
    XM, XG = _resp_matrix(snap_m), _resp_matrix(snap_g)
    Q, K, _ = XM.shape
    out_d = np.empty((draws, 3, W.shape[1]))
    out_q = np.empty((draws, 3, W.shape[1]))
    done = 0
    while done < draws:
        n = min(chunk, draws - done)
        im = rng.integers(0, K, size=(n, Q, K))
        ig = rng.integers(0, K, size=(n, Q, K))
        rm = np.take_along_axis(XM[None], im[..., None].repeat(3, -1), axis=2).mean(axis=2)  # [n, Q, 3]
        rg = np.take_along_axis(XG[None], ig[..., None].repeat(3, -1), axis=2).mean(axis=2)
        diff = rm - rg
        d = np.einsum("nqm,qb->nmb", diff, W)
        gap = diff.mean(axis=1)  # [n, 3]
        out_d[done : done + n] = d
        out_q[done : done + n] = d - gap[:, :, None]
        done += n
    res = {}
    for k, m in enumerate(core.METRICS):
        res[m] = {}
        for j, b in enumerate(core.FIVE_BINS):
            res[m][b] = {
                "d_ci": [float(np.percentile(out_d[:, k, j], 2.5)), float(np.percentile(out_d[:, k, j], 97.5))],
                "q_ci": [float(np.percentile(out_q[:, k, j], 2.5)), float(np.percentile(out_q[:, k, j], 97.5))],
            }
    return res


# ---------------------------------------------------------------- mass (F3)

def mass_table(pair: Pair, bins, steps) -> dict:
    out = {}
    for step in steps:
        S = {arm: core.mass_by_bin(pair.ledgers[arm], bins, step) for arm in ARMS}
        tot = {arm: pair.ledgers[arm].total_mass_before(step) for arm in ARMS}
        out[step] = {
            b: {
                "S_G": S["grpo"][b],
                "S_M": S["maxrl"][b],
                "mass_ratio": S["maxrl"][b] / S["grpo"][b] if S["grpo"][b] > 0 else math.nan,
                "share_ratio": (S["maxrl"][b] / tot["maxrl"]) / (S["grpo"][b] / tot["grpo"])
                if S["grpo"][b] > 0
                else math.nan,
            }
            for b in core.FIVE_BINS
        }
        out[step]["_total"] = tot
    return out


# ---------------------------------------------------------------- F4 batch structure

def batch_structure(ledger: core.Ledger, bins) -> dict:
    live_count = {s: sum(ledger.groups[k]["live"] for k in keys) for s, keys in ledger.by_step.items()}
    n = len(live_count)
    overall = {str(c): sum(1 for v in live_count.values() if v == c) / n for c in (0, 1, 2)}
    step_of = {}
    for (s, i) in ledger.groups:
        step_of.setdefault(i, s)
    by_bin = {}
    for b in core.FIVE_BINS:
        shares = []
        for d in core.DIRECTIONS:
            steps = sorted({step_of[i] for i in core.members(bins, d, b) if i in step_of})
            if not steps:
                shares = None
                break
            shares.append({str(c): sum(1 for s in steps if live_count[s] == c) / len(steps) for c in (0, 1, 2)})
        by_bin[b] = None if shares is None else {c: 0.5 * (shares[0][c] + shares[1][c]) for c in ("0", "1", "2")}
    return {"overall": overall, "panel_steps_by_bin": by_bin, "n_steps": n}


# ---------------------------------------------------------------- F5 bin contents

def bin_contents(bins, baseline_bank: core.P0Bank) -> dict:
    out = {}
    for b in core.FIVE_BINS:
        vals = {"C": [], "T": [], "C_ge_half": []}
        for d in core.DIRECTIONS:
            m = core.members(bins, d, b)
            base = core.OTHER[d]
            c = [baseline_bank.rate(base, i, "C") for i in m]
            t = [baseline_bank.rate(base, i, "T") for i in m]
            vals["C"].append(float(np.mean(c)))
            vals["T"].append(float(np.mean(t)))
            vals["C_ge_half"].append(float(np.mean([x >= 0.5 for x in c])))
        out[b] = {k: 0.5 * (v[0] + v[1]) for k, v in vals.items()}
    return out


# ---------------------------------------------------------------- pre-exposure

def pre_exposure(pair: Pair, bins, baseline_bank, prompt_tokens) -> dict:
    steps = pair.ledgers["grpo"].exposure_steps()
    out = {}
    for pct in core.EXPOSURE_PCTS:
        rows = core.exposure_rows(
            pair.snaps["grpo"][pct], bins, baseline_bank, steps, prompt_tokens, core.PAPER_SCHEDULE[pct]
        )
        out[pct] = core.adjusted_exposure(rows, "C")
    out["never_sampled_panel_questions"] = sorted(i for i, s in steps.items() if s == core.NEVER)
    return out


# ---------------------------------------------------------------- clipping

def clip_series(step_log: list[dict], max_norm: float = 1.0) -> dict[int, float]:
    """optimizer step -> clip coefficient from logged pre-clip grad_norm."""
    out = {}
    for rec in step_log:
        if "grad_norm" in rec and rec.get("step") is not None:
            n = float(rec["grad_norm"])
            out[int(rec["step"])] = min(1.0, max_norm / (n + 1e-6))
    return out


def clipping_summary(coefs: dict[int, float]) -> dict:
    c = np.asarray(list(coefs.values()))
    if c.size == 0:
        return {"n_steps_logged": 0}
    return {
        "n_steps_logged": int(c.size),
        "share_clipped": float(np.mean(c < 1.0)),
        "mean": float(c.mean()),
        "quantiles": {str(q): float(np.percentile(c, q)) for q in (0, 5, 25, 50, 75, 95, 100)},
    }


def clip_weighted_mass(pair: Pair, bins, steps) -> dict | None:
    if not all(arm in pair.step_logs for arm in ARMS):
        return None
    coefs = {arm: clip_series(pair.step_logs[arm]) for arm in ARMS}
    for arm in ARMS:
        missing = [g for g in range(1, 3737) if g not in coefs[arm]]
        if missing:
            return {"error": f"{arm} step log missing {len(missing)} steps"}
    out = {}
    for step in steps:
        S = {
            arm: core.mass_by_bin(pair.ledgers[arm], bins, step, weight=lambda s, i, c=coefs[arm]: c[s + 1])
            for arm in ARMS
        }
        out[step] = {b: {"S_G": S["grpo"][b], "S_M": S["maxrl"][b], "ratio": S["maxrl"][b] / S["grpo"][b]} for b in core.FIVE_BINS}
    return out


def parse_hf_trainer_stdout(path: Path, logging_steps: int = 10) -> list[dict]:
    """Seed-42 MaxRL log: Trainer prints one dict per `logging_steps` optimizer steps."""
    import ast
    import re

    text = Path(path).read_text(encoding="utf-8", errors="replace")
    recs = []
    for m in re.finditer(r"\{'loss': [^{}]*\}", text):
        d = ast.literal_eval(m.group(0))
        recs.append({k: float(v) for k, v in d.items() if k in ("grad_norm", "learning_rate", "epoch", "step_time")})
    for k, rec in enumerate(recs):
        rec["step"] = (k + 1) * logging_steps
        if abs(rec["epoch"] * 3736 - rec["step"]) > 1.5:
            raise ValueError(f"log record {k} epoch {rec['epoch']} does not match step {rec['step']}")
    return recs
