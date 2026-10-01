"""Across-pair statistics and wording rules (PREREG_RUNS.md §8).

Pure functions over per-pair numbers; no I/O. Units: percentage points for
contrasts, ratios for mass.
"""

from __future__ import annotations

import math

import numpy as np

DELTA_REF = 3.0  # pp

# two-sided 97.5 % and one-sided 95 % Student-t quantiles for df = 1..10
T975 = {1: 12.706204736, 2: 4.302652730, 3: 3.182446305, 4: 2.776445105, 5: 2.570581836,
        6: 2.446911851, 7: 2.364624252, 8: 2.306004135, 9: 2.262157163, 10: 2.228138852}
T95 = {1: 6.313751515, 2: 2.919985580, 3: 2.353363435, 4: 2.131846786, 5: 2.015048373,
       6: 1.943180281, 7: 1.894578605, 8: 1.859548038, 9: 1.833112933, 10: 1.812461123}


def across_pairs(values: list[float]) -> dict:
    """Mean, two-sided 95 % t-interval and one-sided 95 % upper bound (n >= 3 only)."""
    x = np.asarray(values, dtype=float)
    n = int(x.size)
    out = {"n": n, "values": [float(v) for v in x], "mean": float(x.mean()) if n else math.nan}
    if n >= 3:
        sd = float(x.std(ddof=1))
        se = sd / math.sqrt(n)
        df = n - 1
        out.update(
            {
                "sd": sd,
                "ci95": [out["mean"] - T975[df] * se, out["mean"] + T975[df] * se],
                "upper95_one_sided": out["mean"] + T95[df] * se,
            }
        )
    else:
        out.update({"sd": float(x.std(ddof=1)) if n >= 2 else math.nan, "ci95": None, "upper95_one_sided": None})
    return out


def wording(stat: dict, delta: float = DELTA_REF) -> str:
    """Pre-registered sentence for a primary cell from its two-sided interval."""
    ci = stat.get("ci95")
    if ci is None:
        return "unresolved at this number of runs"
    lo, hi = ci
    if lo > 0:
        return "additional gain under MaxRL"
    if hi < 0:
        return "lower under MaxRL"
    if -delta < lo and hi < delta:
        return f"no additional gain larger than {delta:g} pp detected"
    return "unresolved at this number of runs"


def inside_range(value: float, others: list[float]) -> str:
    if not others:
        return "no new pairs"
    lo, hi = min(others), max(others)
    return "inside" if lo <= value <= hi else ("below" if value < lo else "above")
