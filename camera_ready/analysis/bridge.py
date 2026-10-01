"""Hardware-bridge statistics and verdict exactly as pre-registered in PREREG_BRIDGE.md.

    python -m camera_ready.analysis.bridge --box-root <dir> --a40-pair-dir <dir> --a40-bank <dir> --out <json>

Box layout: <box-root>/pi0_bank/rollouts_shard{0,1}of2.jsonl,
<box-root>/{grpo,maxrl}_seed42/pi_{025,045,065,100}/snapshot_raw.jsonl.
Missing checkpoints are simply absent from the set S (verdict NOT PASSED if fewer than all nine).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from camera_ready.analysis import core

STEP_PCTS = (25, 45, 65, 100)
B = 5000
RNG_SEED = 20261001


def _rates_bank(bank: core.P0Bank, metric: str) -> np.ndarray:
    return np.asarray([(bank.rate("A", i, metric) + bank.rate("B", i, metric)) / 2 for i in core.PANEL])


def _rates_half(bank: core.P0Bank, half: str, metric: str) -> np.ndarray:
    return np.asarray([bank.rate(half, i, metric) for i in core.PANEL])


def _rates_snap(snap: core.Snapshot, metric: str) -> np.ndarray:
    return np.asarray([snap.rate(i, metric) for i in core.PANEL])


def _len_cap(obj) -> tuple[float, float]:
    if isinstance(obj, core.P0Bank):
        resp = [obj.halves[h][i] for h in core.DIRECTIONS for i in core.PANEL]
    else:
        resp = [obj.responses[i] for i in core.PANEL]
    lengths = np.concatenate([r.length for r in resp])
    term = np.concatenate([r.T for r in resp])
    return float(lengths.mean()), float(1.0 - term.mean())


def compute(box_root: Path, a40_pair_dir: Path, a40_bank_dir: Path) -> dict:
    a40_bank = core.load_p0_bank(a40_bank_dir)
    bins = core.frozen_bins(a40_bank)
    checkpoints: dict[str, dict] = {}  # name -> {"box": obj, "a40": obj, "arm", "pct"}

    box_bank_dir = box_root / "pi0_bank"
    if box_bank_dir.is_dir() and list(box_bank_dir.glob("rollouts_shard*of*.jsonl")):
        box_bank = core.load_p0_bank(box_bank_dir)
        checkpoints["pi0"] = {"box": box_bank, "a40": a40_bank, "arm": "pi0", "pct": 0}
    seed_fn = lambda i: 42 * 100_000 + i + 75_000  # noqa: E731
    for arm in ("grpo", "maxrl"):
        for pct in STEP_PCTS:
            path = box_root / f"{arm}_seed42" / f"pi_{pct:03d}" / "snapshot_raw.jsonl"
            if not path.is_file():
                continue
            checkpoints[f"{arm}_{pct}"] = {
                "box": core.load_snapshot(path, expected_seed=seed_fn),
                "a40": core.load_snapshot(
                    a40_pair_dir / arm / "eval" / f"pi_{pct:03d}" / "snapshot_raw.jsonl", expected_seed=seed_fn
                ),
                "arm": arm,
                "pct": pct,
            }

    names = list(checkpoints)
    n = len(core.PANEL)
    # per-question differences, shape [checkpoint, metric, question]
    diffs = np.zeros((len(names), 3, n))
    for c, name in enumerate(names):
        ck = checkpoints[name]
        for k, m in enumerate(core.METRICS):
            if name == "pi0":
                diffs[c, k] = _rates_bank(ck["box"], m) - _rates_bank(ck["a40"], m)
            else:
                diffs[c, k] = _rates_snap(ck["box"], m) - _rates_snap(ck["a40"], m)

    cells = {}
    for c, name in enumerate(names):
        cells[name] = {}
        for k, m in enumerate(core.METRICS):
            d = diffs[c, k]
            delta = float(d.mean())
            se = float(d.std(ddof=1) / np.sqrt(n))
            z = 0.0 if se == 0 and delta == 0 else (float("inf") if se == 0 else delta / se)
            cells[name][m] = {"delta": delta, "se": se, "z": z}

    pairs = [
        (names.index(f"maxrl_{p}"), names.index(f"grpo_{p}"))
        for p in STEP_PCTS
        if f"maxrl_{p}" in names and f"grpo_{p}" in names
    ]

    def stats(idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        per_ck = diffs[:, :, idx].mean(axis=2)  # [ck, metric]
        pooled = per_ck.mean(axis=0)
        inter = (
            np.mean([per_ck[a] - per_ck[g] for a, g in pairs], axis=0) if pairs else np.full(3, np.nan)
        )
        return pooled, inter

    pooled, inter = stats(np.arange(n))
    rng = np.random.default_rng(RNG_SEED)
    boot_p = np.empty((B, 3))
    boot_i = np.empty((B, 3))
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        boot_p[b], boot_i[b] = stats(idx)
    ci_p = np.percentile(boot_p, [2.5, 97.5], axis=0)
    ci_i = np.percentile(boot_i, [2.5, 97.5], axis=0) if pairs else np.full((2, 3), np.nan)

    pp = 100.0
    gate_a = all(-1.5 < pp * ci_p[0, k] and pp * ci_p[1, k] < 1.5 for k in range(3))
    zs = {name: max(abs(cells[name][m]["z"]) for m in core.METRICS) for name in names}
    gate_b = all(z <= 3.5 for z in zs.values()) and sum(1 for z in zs.values() if z > 2.5) <= 1
    gate_c = bool(pairs) and all(
        abs(pp * inter[k]) <= 1.5 and -3.0 < pp * ci_i[0, k] and pp * ci_i[1, k] < 3.0
        for k in (core.METRICS.index("C"), core.METRICS.index("R"))
    )
    rl = [nm for nm in names if nm != "pi0"]
    gate_d = len(rl) >= 4 and {checkpoints[nm]["arm"] for nm in rl} == {"grpo", "maxrl"}
    gross = any(abs(pp * pooled[k]) > 5 for k in range(3)) or (
        "pi0" in cells and any(abs(pp * cells["pi0"][m]["delta"]) > 5 for m in core.METRICS)
    )
    complete = len(names) == 9
    if gross:
        verdict = "GROSS FAIL"
    elif complete and gate_a and gate_b and gate_c and gate_d:
        verdict = "PASS"
    else:
        verdict = "NOT PASSED" + ("" if complete else " (bridge unfinished)")

    # Non-gating descriptives
    descr = {}
    for name in names:
        ck = checkpoints[name]
        lb, cb = _len_cap(ck["box"])
        la, ca = _len_cap(ck["a40"])
        entry = {"mean_len_box": lb, "mean_len_a40": la, "cap_box": cb, "cap_a40": ca}
        per_bin = {}
        for m in core.METRICS:
            if name == "pi0":
                fn = lambda d, i, m=m: (ck["box"].rate("A", i, m) + ck["box"].rate("B", i, m)) / 2 - (
                    ck["a40"].rate("A", i, m) + ck["a40"].rate("B", i, m)
                ) / 2
            else:
                fn = lambda d, i, m=m: ck["box"].rate(i, m) - ck["a40"].rate(i, m)
            per_bin[m] = core.symmetric_bin_mean(bins, fn)
        entry["per_bin_delta"] = per_bin
        if name == "pi0":
            entry["per_bank_delta"] = {
                h: {m: float((_rates_half(ck["box"], h, m) - _rates_half(ck["a40"], h, m)).mean()) for m in core.METRICS}
                for h in core.DIRECTIONS
            }
        descr[name] = entry

    return {
        "checkpoints": names,
        "cells": cells,
        "pooled": {m: {"delta": float(pooled[k]), "ci": [float(ci_p[0, k]), float(ci_p[1, k])]} for k, m in enumerate(core.METRICS)},
        "interaction": {
            "steps": [checkpoints[names[a]]["pct"] for a, _ in pairs],
            **{m: {"delta": float(inter[k]), "ci": [float(ci_i[0, k]), float(ci_i[1, k])]} for k, m in enumerate(core.METRICS)},
        },
        "gates": {"a": gate_a, "b": gate_b, "c": gate_c, "d": gate_d, "gross": gross, "complete": complete},
        "max_abs_z_by_checkpoint": zs,
        "verdict": verdict,
        "descriptives": descr,
        "bootstrap": {"draws": B, "rng_seed": RNG_SEED},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--box-root", type=Path, required=True)
    parser.add_argument("--a40-pair-dir", type=Path, required=True)
    parser.add_argument("--a40-bank", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = compute(args.box_root, args.a40_pair_dir, args.a40_bank)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, default=float))
    print("checkpoints:", result["checkpoints"])
    for m, v in result["pooled"].items():
        print(f"pooled {m}: {100*v['delta']:+.3f} pp  CI [{100*v['ci'][0]:+.3f}, {100*v['ci'][1]:+.3f}]")
    for m in core.METRICS:
        v = result["interaction"][m]
        print(f"interaction {m}: {100*v['delta']:+.3f} pp  CI [{100*v['ci'][0]:+.3f}, {100*v['ci'][1]:+.3f}]")
    print("gates:", result["gates"])
    print("VERDICT:", result["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
