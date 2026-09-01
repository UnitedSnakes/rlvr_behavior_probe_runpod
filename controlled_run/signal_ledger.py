from __future__ import annotations

import json
import math
from pathlib import Path

import torch


TRL_LOG_IS_LOWER_CLAMP = math.log(1e-8)
TRL_LOG_IS_UPPER_CLAMP = 20.0

LEDGER_FIELDS = (
    "generation_global_step",
    "rank",
    "dataset_index",
    "correct",
    "terminated",
    "canonical_reward",
    "completion_length",
    "group_successes",
    "group_size",
    "group_reward_std",
    "advantage",
    "raw_log_rho",
    "effective_log_rho",
    "importance_sampling_ratio",
    "upper_cap_masked",
)


def compute_raw_sequence_log_rho(
    old_per_token_logps: torch.Tensor,
    sampling_per_token_logps: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct pre-exp sequence log importance ratios exactly as TRL does."""
    if old_per_token_logps.shape != sampling_per_token_logps.shape:
        raise ValueError("old and sampling log-probability tensors must have matching shapes")
    if mask.shape != old_per_token_logps.shape:
        raise ValueError("importance-sampling mask must match per-token log-probability shape")

    per_token_diff = (old_per_token_logps - sampling_per_token_logps) * mask
    per_token_diff = torch.nan_to_num(per_token_diff, nan=0.0)
    return per_token_diff.sum(dim=-1, keepdim=True)


def compute_effective_log_rho(
    importance_sampling_ratio: torch.Tensor,
    *,
    lower_clamp: float = TRL_LOG_IS_LOWER_CLAMP,
    upper_clamp: float = TRL_LOG_IS_UPPER_CLAMP,
) -> torch.Tensor:
    """Return the log-ratio value that reaches TRL's loss after ratio masking/underflow."""
    return torch.clamp(torch.log(importance_sampling_ratio), lower_clamp, upper_clamp)


def infer_upper_cap_mask(raw_log_rho: torch.Tensor, *, clip_max: float | None) -> torch.Tensor:
    """Infer sequence-mask rejection from the pre-exp ratio, before zero-fill erases provenance."""
    if clip_max is None:
        return torch.zeros_like(raw_log_rho, dtype=torch.bool)
    if clip_max <= 0:
        raise ValueError("clip_max must be positive when supplied")
    return raw_log_rho > math.log(clip_max)


def append_ledger_rows(path: Path, rows: list[dict]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("a", encoding="utf-8") as handle:
        for row in rows:
            missing = [field for field in LEDGER_FIELDS if field not in row]
            extra = [field for field in row if field not in LEDGER_FIELDS]
            if missing or extra:
                raise ValueError(
                    f"Signal ledger row schema mismatch; missing={missing}, extra={extra}"
                )
            handle.write(json.dumps({field: row[field] for field in LEDGER_FIELDS}, sort_keys=True))
            handle.write("\n")
