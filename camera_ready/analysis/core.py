"""Core estimands for the camera-ready replication analysis.

Re-implements, without importing them, the definitions used by the submitted
paper (analyses/snapshot_crossfit_trajectory.py, ledger_crossfit_signal_allocation.py,
exposure_split_adjusted.py, controlled_run/maxrl_canonical_acceptance.py), so the
same code can be applied to any number of GRPO/MaxRL pairs.

Conventions
-----------
- Direction "A" means bank A selects the bin and bank B supplies the baseline;
  direction "B" is the reverse. Symmetric statistics average the two direction
  means with equal weight; a bin that is empty in either direction is omitted.
- A snapshot saved after optimizer step S sees ledger groups with
  generation_global_step < S. A panel question is exposed at S iff its own
  group's step is < S. A question never sampled by the run has step = +inf.
- All rates are fractions; reporting code converts to percentage points.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

BIN_ORDER = ("0", "(0,.25]", "(.25,.5]", "(.5,.75]", "(.75,1)", "1")
FIVE_BINS = BIN_ORDER[:5]
DIRECTIONS = ("A", "B")
OTHER = {"A": "B", "B": "A"}
METRICS = ("R", "T", "C")
PANEL = tuple(range(256))
GROUP_SIZE = 16
PAPER_SCHEDULE = {
    5: 187, 10: 374, 15: 560, 20: 747, 25: 934, 30: 1121, 35: 1308, 40: 1494,
    45: 1681, 50: 1868, 55: 2055, 60: 2242, 65: 2428, 70: 2615, 75: 2802,
    80: 2989, 85: 3176, 90: 3362, 95: 3549, 100: 3736,
}
EVAL_PCTS = (25, 45, 65, 100)
EXPOSURE_PCTS = (25, 45, 65)
NEVER = math.inf


def assign_bin(p: float) -> str:
    value = float(p)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"rate outside [0,1]: {value}")
    if value == 0.0:
        return "0"
    if value <= 0.25:
        return "(0,.25]"
    if value <= 0.5:
        return "(.25,.5]"
    if value <= 0.75:
        return "(.5,.75]"
    if value < 1.0:
        return "(.75,1)"
    return "1"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"bad JSONL {path}:{lineno}: {exc}") from exc
    return rows


# --------------------------------------------------------------------------
# Response banks
# --------------------------------------------------------------------------

@dataclass
class Responses:
    """Binary R/T/C indicators and lengths for the responses of one question."""

    R: np.ndarray
    T: np.ndarray
    C: np.ndarray
    length: np.ndarray

    def rate(self, metric: str) -> float:
        return float(getattr(self, metric).mean())

    @property
    def k(self) -> int:
        return int(self.R.size)


def _responses(rows: list[dict], length_key: str) -> Responses:
    if not rows:
        raise ValueError("empty response list")
    R = np.asarray([int(r["canonical_reward"]) for r in rows], dtype=float)
    T = np.asarray([bool(r["terminated"]) for r in rows], dtype=float)
    C = np.asarray([bool(r["correct"]) for r in rows], dtype=float)
    L = np.asarray([float(r[length_key]) for r in rows], dtype=float)
    if not np.all((R == 0) | (R == 1)):
        raise ValueError("non-binary reward")
    if np.any(R > T * C + 1e-12) or np.any(R < T * C - 1e-12):
        raise ValueError("R != T*C for some response")
    return Responses(R=R, T=T, C=C, length=L)


@dataclass
class P0Bank:
    """Two independent 16-response pi0 halves per question."""

    halves: dict[str, dict[int, Responses]]
    source: str = ""

    def rate(self, half: str, idx: int, metric: str) -> float:
        return self.halves[half][idx].rate(metric)


def load_p0_bank(directory: Path, panel: Iterable[int] = PANEL) -> P0Bank:
    """Load a K=32 cross-fit bank written by diagnose_p0_signal_budget.py."""
    directory = Path(directory)
    records = []
    for path in sorted(directory.glob("rollouts_shard*of*.jsonl")):
        records.extend(read_jsonl(path))
    if not records:
        raise FileNotFoundError(f"no rollouts_shard*.jsonl under {directory}")
    by_index: dict[int, dict] = {}
    for record in records:
        idx = int(record["dataset_index"])
        if idx in by_index:
            raise ValueError(f"duplicate p0 record {idx}")
        by_index[idx] = record
    expected = set(int(i) for i in panel)
    if set(by_index) != expected:
        raise ValueError(
            f"p0 bank index mismatch: missing={sorted(expected - set(by_index))[:5]} "
            f"extra={sorted(set(by_index) - expected)[:5]}"
        )
    halves: dict[str, dict[int, Responses]] = {"A": {}, "B": {}}
    for idx, record in by_index.items():
        for half in DIRECTIONS:
            resp = _responses(list(record[f"rollouts_{half}"]), "n_tokens")
            if resp.k != GROUP_SIZE:
                raise ValueError(f"p0 half {half} of {idx} has {resp.k} responses")
            halves[half][idx] = resp
        # Internal consistency with the stored summaries.
        if not math.isclose(record["p0_A"], halves["A"][idx].rate("R"), abs_tol=1e-12):
            raise ValueError(f"p0_A mismatch at {idx}")
        if not math.isclose(record["p0_B"], halves["B"][idx].rate("R"), abs_tol=1e-12):
            raise ValueError(f"p0_B mismatch at {idx}")
    return P0Bank(halves=halves, source=str(directory))


@dataclass
class Snapshot:
    """Evaluation of one policy on the panel: per-question responses."""

    responses: dict[int, Responses]
    seeds: dict[int, int]
    source: str = ""

    def rate(self, idx: int, metric: str) -> float:
        return self.responses[idx].rate(metric)

    def panel_rate(self, metric: str, panel: Iterable[int] = PANEL) -> float:
        # Pooled over responses; equals the mean of per-question rates when K is constant.
        num = sum(float(getattr(self.responses[i], metric).sum()) for i in panel)
        den = sum(self.responses[i].k for i in panel)
        return num / den


def load_snapshot(
    path: Path,
    *,
    panel: Iterable[int] = PANEL,
    expected_k: int | None = GROUP_SIZE,
    expected_seed: Callable[[int], int] | None = None,
) -> Snapshot:
    rows = read_jsonl(Path(path))
    responses: dict[int, Responses] = {}
    seeds: dict[int, int] = {}
    for record in rows:
        idx = int(record["dataset_index"])
        if idx in responses:
            raise ValueError(f"duplicate question {idx} in {path}")
        resp = _responses(list(record["rollouts"]), "completion_length")
        n = int(record["n_rollouts"])
        if resp.k != n:
            raise ValueError(f"{path}: {idx} declares {n} rollouts, has {resp.k}")
        if expected_k is not None and n != expected_k:
            raise ValueError(f"{path}: {idx} has K={n}, expected {expected_k}")
        for metric, key in (("R", "n_reward"), ("T", "n_terminated"), ("C", "n_correct")):
            if int(record[key]) != int(getattr(resp, metric).sum()):
                raise ValueError(f"{path}: {idx} count mismatch for {metric}")
        seed = int(record["question_seed"])
        if expected_seed is not None and seed != expected_seed(idx):
            raise ValueError(f"{path}: {idx} seed {seed} != expected {expected_seed(idx)}")
        responses[idx] = resp
        seeds[idx] = seed
    expected = set(int(i) for i in panel)
    if set(responses) != expected:
        raise ValueError(
            f"{path}: index mismatch missing={sorted(expected - set(responses))[:5]} "
            f"extra={sorted(set(responses) - expected)[:5]}"
        )
    return Snapshot(responses=responses, seeds=seeds, source=str(path))


def merge_snapshots(parts: list[Snapshot]) -> Snapshot:
    """Concatenate several response batches of the same policy question by question."""
    if not parts:
        raise ValueError("nothing to merge")
    keys = set(parts[0].responses)
    for part in parts[1:]:
        if set(part.responses) != keys:
            raise ValueError("cannot merge snapshots with different panels")
    merged: dict[int, Responses] = {}
    for idx in keys:
        rs = [p.responses[idx] for p in parts]
        merged[idx] = Responses(
            R=np.concatenate([r.R for r in rs]),
            T=np.concatenate([r.T for r in rs]),
            C=np.concatenate([r.C for r in rs]),
            length=np.concatenate([r.length for r in rs]),
        )
    return Snapshot(responses=merged, seeds={}, source="+".join(p.source for p in parts))


# --------------------------------------------------------------------------
# Bins and cross-fit averaging
# --------------------------------------------------------------------------

def frozen_bins(bin_bank: P0Bank, panel: Iterable[int] = PANEL) -> dict[str, dict[int, str]]:
    """Bin membership per direction: direction d bins question i by bank half d."""
    return {d: {int(i): assign_bin(bin_bank.rate(d, int(i), "R")) for i in panel} for d in DIRECTIONS}


def bin_counts(bins: dict[str, dict[int, str]]) -> dict[str, dict[str, int]]:
    return {d: {b: sum(1 for v in bins[d].values() if v == b) for b in BIN_ORDER} for d in DIRECTIONS}


def members(bins: dict[str, dict[int, str]], direction: str, label: str) -> list[int]:
    return sorted(i for i, b in bins[direction].items() if b == label)


def symmetric_bin_mean(
    bins: dict[str, dict[int, str]],
    value: Callable[[str, int], float],
    labels: Iterable[str] = FIVE_BINS,
) -> dict[str, float]:
    """Average over directions of the within-bin mean of value(direction, idx)."""
    out = {}
    for label in labels:
        dir_means = []
        for d in DIRECTIONS:
            m = members(bins, d, label)
            if not m:
                break
            dir_means.append(float(np.mean([value(d, i) for i in m])))
        else:
            out[label] = 0.5 * (dir_means[0] + dir_means[1])
    return out


def delta_by_bin(
    snap: Snapshot,
    bins: dict[str, dict[int, str]],
    baseline_bank: P0Bank,
    metric: str,
    labels: Iterable[str] = FIVE_BINS,
) -> dict[str, float]:
    """Cross-fit Delta X_b: snapshot rate minus the opposite-half pi0 rate."""
    return symmetric_bin_mean(
        bins,
        lambda d, i: snap.rate(i, metric) - baseline_bank.rate(OTHER[d], i, metric),
        labels,
    )


def contrast_by_bin(
    snap_m: Snapshot,
    snap_g: Snapshot,
    bins: dict[str, dict[int, str]],
    metric: str,
    labels: Iterable[str] = FIVE_BINS,
) -> dict[str, dict[str, float]]:
    """d_b = X_M - X_G within bin (baselines cancel); q_b = d_b - panel gap."""
    d = symmetric_bin_mean(bins, lambda _d, i: snap_m.rate(i, metric) - snap_g.rate(i, metric), labels)
    panel_gap = snap_m.panel_rate(metric) - snap_g.panel_rate(metric)
    return {"d": d, "q": {b: v - panel_gap for b, v in d.items()}, "panel_gap": panel_gap}


def initial_panel_rates(bank: P0Bank, panel: Iterable[int] = PANEL) -> dict[str, float]:
    """Table 1 initial row: all 32 pi0 responses per question pooled."""
    panel = list(panel)
    return {
        m: float(np.mean([(bank.rate("A", i, m) + bank.rate("B", i, m)) / 2 for i in panel]))
        for m in METRICS
    }


# --------------------------------------------------------------------------
# Ledger: groups, exposure, advantage mass, structural audit
# --------------------------------------------------------------------------

@dataclass
class Ledger:
    rows: int
    rank_files: int
    groups: dict[tuple[int, int], dict] = field(default_factory=dict)
    by_step: dict[int, list[tuple[int, int]]] = field(default_factory=dict)
    token_ratio_count: int = 0
    token_ratio_sum: float = 0.0
    token_ratio_sq_sum: float = 0.0
    nonfinite_numeric_fields: int = 0
    step_rank_counts: dict[tuple[int, int], int] = field(default_factory=dict)

    def exposure_steps(self, panel: Iterable[int] = PANEL) -> dict[int, float]:
        seen: dict[int, list[int]] = defaultdict(list)
        for step, idx in self.groups:
            seen[idx].append(step)
        out = {}
        for i in panel:
            steps = seen.get(int(i), [])
            if len(steps) > 1:
                raise ValueError(f"question {i} sampled at more than one step: {steps}")
            out[int(i)] = float(steps[0]) if steps else NEVER
        return out

    def question_mass(self) -> dict[int, float]:
        mass: dict[int, float] = defaultdict(float)
        for (_step, idx), group in self.groups.items():
            mass[idx] += group["abs_adv"]
        return dict(mass)

    def total_mass_before(self, step: int) -> float:
        return float(sum(g["abs_adv"] for (s, _i), g in self.groups.items() if s < step))


def load_ledger(directory: Path) -> Ledger:
    files = sorted(Path(directory).glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no ledger files under {directory}")
    ledger = Ledger(rows=0, rank_files=len(files))
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    step_rank: dict[tuple[int, int], int] = defaultdict(int)
    for path in files:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                ledger.rows += 1
                for value in row.values():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        if not math.isfinite(float(value)):
                            ledger.nonfinite_numeric_fields += 1
                step = int(row["generation_global_step"])
                grouped[(step, int(row["dataset_index"]))].append(
                    {
                        "reward": float(row["canonical_reward"]),
                        "advantage": float(row["advantage"]),
                        "group_successes": int(row["group_successes"]),
                        "group_size": int(row["group_size"]),
                        "completion_length": int(row["completion_length"]),
                        "terminated": bool(row["terminated"]),
                        "correct": bool(row["correct"]),
                    }
                )
                step_rank[(step, int(row["rank"]))] += 1
                ledger.token_ratio_count += int(row["actual_is_ratio_count"])
                ledger.token_ratio_sum += float(row["actual_is_ratio_sum"])
                ledger.token_ratio_sq_sum += float(row["actual_is_ratio_sq_sum"])
    for key, rows in grouped.items():
        rewards = np.asarray([r["reward"] for r in rows])
        adv = np.asarray([r["advantage"] for r in rows])
        ledger.groups[key] = {
            "n": len(rows),
            "K": int(rewards.sum()),
            "declared_K": {r["group_successes"] for r in rows},
            "declared_G": {r["group_size"] for r in rows},
            "rewards": rewards,
            "advantages": adv,
            "abs_adv": float(np.abs(adv).sum()),
            "live": bool(np.any(adv != 0.0)),
            "mean_len": float(np.mean([r["completion_length"] for r in rows])),
            "cap_hits": int(sum(1 for r in rows if not r["terminated"])),
            "R": float(rewards.mean()),
            "T": float(np.mean([r["terminated"] for r in rows])),
            "C": float(np.mean([r["correct"] for r in rows])),
        }
        ledger.by_step.setdefault(key[0], []).append(key)
    ledger.step_rank_counts = dict(step_rank)
    return ledger


def expected_advantages(rewards: np.ndarray, objective: str) -> np.ndarray:
    """Float64 reconstruction of the implemented group advantages."""
    r = np.asarray(rewards, dtype=float)
    G = r.size
    K = r.sum()
    if objective == "maxrl":
        if K == 0:
            return np.zeros(G)
        p = K / G
        return (r - p) / p
    if objective == "grpo":
        mean = r.mean()
        sd = r.std(ddof=1)
        return (r - mean) / (sd + 1e-4)
    raise ValueError(objective)


def structural_audit(ledger: Ledger, objective: str, expected_steps: int = 3736) -> dict:
    """The seed-42 MaxRL structural checks, applied to either objective."""
    problems = []
    if ledger.rank_files != 2:
        problems.append(f"rank_files={ledger.rank_files}")
    if ledger.rows != expected_steps * 32:
        problems.append(f"rows={ledger.rows}")
    if len(ledger.groups) != expected_steps * 2:
        problems.append(f"groups={len(ledger.groups)}")
    if set(ledger.by_step) != set(range(expected_steps)):
        problems.append("generation steps are not exactly 0..N-1")
    if any(len(v) != 2 for v in ledger.by_step.values()):
        problems.append("a step does not hold exactly two groups")
    if any(c != 16 for c in ledger.step_rank_counts.values()):
        problems.append("a (step, rank) does not hold exactly 16 rows")
    if ledger.nonfinite_numeric_fields:
        problems.append(f"nonfinite={ledger.nonfinite_numeric_fields}")
    max_err = 0.0
    for key, g in ledger.groups.items():
        if g["n"] != GROUP_SIZE or g["declared_G"] != {GROUP_SIZE} or g["declared_K"] != {g["K"]}:
            problems.append(f"group {key} malformed")
            continue
        err = float(np.max(np.abs(g["advantages"] - expected_advantages(g["rewards"], objective))))
        max_err = max(max_err, err)
    if max_err > 1e-6:
        problems.append(f"max_advantage_error={max_err}")
    ess = (ledger.token_ratio_sum ** 2) / ledger.token_ratio_sq_sum / ledger.token_ratio_count
    return {
        "status": "PASS" if not problems else "FAIL",
        "problems": problems[:20],
        "objective": objective,
        "rows": ledger.rows,
        "groups": len(ledger.groups),
        "rank_files": ledger.rank_files,
        "steps": len(ledger.by_step),
        "max_advantage_error": max_err,
        "aggregate_token_is_ess_fraction": ess,
        "nonfinite_numeric_fields": ledger.nonfinite_numeric_fields,
    }


def mass_by_bin(
    ledger: Ledger,
    bins: dict[str, dict[int, str]],
    step: int,
    labels: Iterable[str] = FIVE_BINS,
    weight: Callable[[int, int], float] | None = None,
) -> dict[str, float]:
    """Symmetric S_t(b): cumulative |A| of the bin's own groups before `step`,
    divided by the number of panel questions in the bin (exposed or not).

    `weight(step, idx)` optionally rescales each group's mass (clip-coefficient proxy)."""
    group_of: dict[int, tuple[int, int]] = {}
    for key in ledger.groups:
        group_of.setdefault(key[1], key)

    def value(_d: str, i: int) -> float:
        key = group_of.get(i)
        if key is None or key[0] >= step:
            return 0.0
        w = 1.0 if weight is None else weight(key[0], i)
        return ledger.groups[key]["abs_adv"] * w

    return symmetric_bin_mean(bins, value, labels)


# --------------------------------------------------------------------------
# Covariate-adjusted own-exposure estimator (frozen model, Appendix C)
# --------------------------------------------------------------------------

COVARIATES = ("baseline_p0", "baseline_p0_completion_length", "prompt_token_count")


def exposure_rows(
    snap: Snapshot,
    bins: dict[str, dict[int, str]],
    baseline_bank: P0Bank,
    exposure_steps: dict[int, float],
    prompt_tokens: dict[int, float],
    snapshot_step: int,
    panel: Iterable[int] = PANEL,
) -> list[dict]:
    out = []
    for d in DIRECTIONS:
        base = OTHER[d]
        for i in panel:
            resp = baseline_bank.halves[base][i]
            row = {
                "direction": d,
                "bin": bins[d][i],
                "dataset_index": i,
                "exposed": exposure_steps[i] < snapshot_step,
                "baseline_p0": resp.rate("R"),
                "baseline_p0_completion_length": float(resp.length.mean()),
                "prompt_token_count": float(prompt_tokens[i]),
            }
            for m in METRICS:
                row[f"delta_{m}"] = snap.rate(i, m) - resp.rate(m)
            out.append(row)
    return out


def fit_exposure_cell(cell: list[dict], metric: str, weights: np.ndarray | None = None) -> dict | None:
    """OLS: Delta X = a + tau*E + beta'z (z standardized in the cell; constants dropped).
    Returns adjusted unexposed (a), exposed (a+tau) and gap U-E (-tau); None if ineligible."""
    w = np.ones(len(cell)) if weights is None else np.asarray(weights, dtype=float)
    keep = w > 0
    cell = [row for row, k in zip(cell, keep) if k]
    w = w[keep]
    exposed = np.asarray([1.0 if r["exposed"] else 0.0 for r in cell])
    n_e = float(np.sum(w * exposed))
    n_u = float(np.sum(w * (1 - exposed)))
    if weights is None and min(n_e, n_u) < 2:
        return None
    if weights is not None and (min(int((exposed > 0).sum()), int((exposed == 0).sum())) < 2):
        return None
    cols = [np.ones(len(cell)), exposed]
    for name in COVARIATES:
        v = np.asarray([float(r[name]) for r in cell])
        mean = float(np.sum(w * v) / np.sum(w))
        var = float(np.sum(w * (v - mean) ** 2) / (np.sum(w) - 1)) if np.sum(w) > 1 else 0.0
        sd = math.sqrt(var) if var > 0 else 0.0
        if sd <= 0.0:
            continue
        cols.append((v - mean) / sd)
    X = np.column_stack(cols)
    if np.linalg.matrix_rank(X * np.sqrt(w)[:, None]) != X.shape[1]:
        return None
    y = np.asarray([float(r[f"delta_{metric}"]) for r in cell])
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    return {
        "unexposed": float(coef[0]),
        "exposed": float(coef[0] + coef[1]),
        "gap_u_minus_e": float(-coef[1]),
        "n_exposed": int(round(n_e)),
        "n_unexposed": int(round(n_u)),
    }


def adjusted_exposure(
    rows: list[dict],
    metric: str = "C",
    labels: Iterable[str] = FIVE_BINS,
) -> dict[str, dict]:
    """Symmetric adjusted exposure estimates per bin (direction fits averaged)."""
    out = {}
    for label in labels:
        fits = {}
        for d in DIRECTIONS:
            cell = [r for r in rows if r["direction"] == d and r["bin"] == label]
            fits[d] = fit_exposure_cell(cell, metric) if cell else None
        if fits["A"] is None or fits["B"] is None:
            out[label] = {"status": "not_estimable", "A": fits["A"], "B": fits["B"]}
            continue
        sym = {k: 0.5 * (fits["A"][k] + fits["B"][k]) for k in ("unexposed", "exposed")}
        sym["gap_u_minus_e"] = sym["unexposed"] - sym["exposed"]
        sym["status"] = "ok"
        sym["A"] = fits["A"]
        sym["B"] = fits["B"]
        out[label] = sym
    return out
