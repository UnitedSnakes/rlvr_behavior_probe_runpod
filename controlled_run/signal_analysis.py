from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

import numpy as np


def _finite_rows(rows: Iterable[dict]) -> list[dict]:
    valid = []
    for row in rows:
        raw = row.get("raw_log_rho")
        length = row.get("completion_length")
        if raw is None or length is None:
            continue
        raw = float(raw)
        length = int(length)
        if not math.isfinite(raw) or length <= 0:
            continue
        valid.append(row)
    return valid


def summarize_sequence_mean_logprob_diff(rows: Iterable[dict], *, top_p: float) -> dict:
    """Summarize sequence-average log-probability differences.

    ``raw_log_rho / completion_length`` is a sequence average, not a dump of
    individual token-level differences. The top-p band is therefore only a
    coarse diagnostic for whether the sequence aggregate is compatible with
    the known processed-logprob truncation mechanism.
    """
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must lie in (0, 1]")

    valid = _finite_rows(rows)
    if not valid:
        raise ValueError("No finite signal-ledger rows with positive completion length")

    values = np.asarray(
        [float(row["raw_log_rho"]) / int(row["completion_length"]) for row in valid],
        dtype=float,
    )
    lower = math.log(top_p)
    in_band = (values >= lower) & (values <= 0.0)

    return {
        "n": int(values.size),
        "top_p_log_lower_bound": lower,
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "min": float(values.min()),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.5)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
        "fraction_in_top_p_band": float(in_band.mean()),
        "fraction_below_top_p_band": float((values < lower).mean()),
        "fraction_positive": float((values > 0.0).mean()),
    }


def _cluster_robust_covariance(x: np.ndarray, residuals: np.ndarray, clusters: list[tuple]) -> np.ndarray:
    n, p = x.shape
    grouped: dict[tuple, list[int]] = defaultdict(list)
    for index, cluster in enumerate(clusters):
        grouped[cluster].append(index)

    g = len(grouped)
    if g <= 1:
        raise ValueError("Cluster-robust covariance requires at least two clusters")
    if n <= p:
        raise ValueError("Cluster-robust covariance requires more observations than coefficients")

    bread = np.linalg.pinv(x.T @ x)
    meat = np.zeros((p, p), dtype=float)
    for indices in grouped.values():
        xg = x[indices, :]
        ug = residuals[indices]
        score = xg.T @ ug
        meat += np.outer(score, score)

    # CR1 small-sample correction, matching the common one-way clustered
    # sandwich estimator: G/(G-1) * (N-1)/(N-K).
    correction = (g / (g - 1)) * ((n - 1) / (n - p))
    return correction * bread @ meat @ bread


def fit_length_difficulty_regression(rows: Iterable[dict]) -> dict:
    """Fit raw log-rho on length and observed group success fraction.

    OLS is at rollout level, while uncertainty is clustered by the prompt
    group key ``(generation_global_step, dataset_index)`` because k/G is shared
    by all G rollouts from a prompt group.
    """
    valid = _finite_rows(rows)
    if not valid:
        raise ValueError("No finite signal-ledger rows available for regression")

    y = np.asarray([float(row["raw_log_rho"]) for row in valid], dtype=float)
    length = np.asarray([float(row["completion_length"]) for row in valid], dtype=float)
    success_fraction = np.asarray(
        [float(row["group_successes"]) / float(row["group_size"]) for row in valid],
        dtype=float,
    )
    x = np.column_stack([np.ones_like(length), length, success_fraction])

    beta, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ beta
    residuals = y - fitted
    centered = y - y.mean()
    ss_tot = float(centered @ centered)
    ss_res = float(residuals @ residuals)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")

    clusters = [
        (int(row["generation_global_step"]), int(row["dataset_index"]))
        for row in valid
    ]
    covariance = _cluster_robust_covariance(x, residuals, clusters)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))

    return {
        "n": len(valid),
        "n_clusters": len(set(clusters)),
        "intercept": float(beta[0]),
        "length_coef": float(beta[1]),
        "success_fraction_coef": float(beta[2]),
        "r_squared": r_squared,
        "cluster_se_intercept": float(standard_errors[0]),
        "cluster_se_length": float(standard_errors[1]),
        "cluster_se_success_fraction": float(standard_errors[2]),
    }
