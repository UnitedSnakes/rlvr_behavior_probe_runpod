from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch


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
    "token_delta_count",
    "token_delta_mean",
    "token_delta_std",
    "token_delta_min",
    "token_delta_max",
    "token_delta_positive_fraction",
    "token_ratio_sum",
    "token_ratio_sq_sum",
    "token_ratio_gt_clip_fraction",
    "token_abs_delta_gt_1_fraction",
    "trimmed_token_count_1pct",
    "trimmed_raw_log_rho_1pct",
    "actual_is_ratio_count",
    "actual_is_ratio_mean",
    "actual_is_ratio_std",
    "actual_is_ratio_min",
    "actual_is_ratio_max",
    "actual_is_ratio_sum",
    "actual_is_ratio_sq_sum",
    "actual_is_log_ratio_mean",
    "actual_is_ratio_at_upper_cap_fraction",
)


def _validate_per_token_inputs(
    old_per_token_logps: torch.Tensor,
    sampling_per_token_logps: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    if old_per_token_logps.shape != sampling_per_token_logps.shape:
        raise ValueError("old and sampling log-probability tensors must have matching shapes")
    if mask.shape != old_per_token_logps.shape:
        raise ValueError("importance-sampling mask must match per-token log-probability shape")
    if old_per_token_logps.ndim != 2:
        raise ValueError("per-token log-probability tensors must be rank-2 [batch, tokens]")


def compute_raw_sequence_log_rho(
    old_per_token_logps: torch.Tensor,
    sampling_per_token_logps: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Counterfactual sequence-product log-ratio before any vLLM IS clipping/masking."""
    _validate_per_token_inputs(old_per_token_logps, sampling_per_token_logps, mask)

    per_token_diff = (old_per_token_logps - sampling_per_token_logps) * mask
    per_token_diff = torch.nan_to_num(per_token_diff, nan=0.0)
    return per_token_diff.sum(dim=-1, keepdim=True)


def compute_token_logprob_summary(
    old_per_token_logps: torch.Tensor,
    sampling_per_token_logps: torch.Tensor,
    mask: torch.Tensor,
    *,
    clip_max: float | None,
    trim_fraction: float = 0.01,
) -> dict[str, torch.Tensor]:
    """Summarize active-token train-vs-vLLM log-probability discrepancies."""
    _validate_per_token_inputs(old_per_token_logps, sampling_per_token_logps, mask)
    if not 0.0 <= trim_fraction < 1.0:
        raise ValueError("trim_fraction must lie in [0, 1)")
    if clip_max is not None and clip_max <= 0:
        raise ValueError("clip_max must be positive when supplied")

    raw_delta = old_per_token_logps - sampling_per_token_logps
    active_mask = mask.to(dtype=torch.bool)
    clip_log = math.log(clip_max) if clip_max is not None else None

    values: dict[str, list[torch.Tensor]] = {
        "token_delta_count": [],
        "token_delta_mean": [],
        "token_delta_std": [],
        "token_delta_min": [],
        "token_delta_max": [],
        "token_delta_positive_fraction": [],
        "token_ratio_sum": [],
        "token_ratio_sq_sum": [],
        "token_ratio_gt_clip_fraction": [],
        "token_abs_delta_gt_1_fraction": [],
        "trimmed_token_count_1pct": [],
        "trimmed_raw_log_rho_1pct": [],
    }

    for row_index in range(raw_delta.shape[0]):
        active = raw_delta[row_index][active_mask[row_index]]
        if active.numel() == 0:
            raise ValueError("token-level IS summary requires at least one active token per rollout")
        active = torch.nan_to_num(active, nan=0.0).to(torch.float64)
        count = int(active.numel())

        values["token_delta_count"].append(
            torch.tensor(count, dtype=torch.int64, device=active.device)
        )
        values["token_delta_mean"].append(active.mean())
        values["token_delta_std"].append(active.std(correction=0))
        values["token_delta_min"].append(active.min())
        values["token_delta_max"].append(active.max())
        values["token_delta_positive_fraction"].append((active > 0).to(torch.float64).mean())

        ratios = torch.exp(active)
        values["token_ratio_sum"].append(ratios.sum())
        values["token_ratio_sq_sum"].append(torch.exp(2.0 * active).sum())
        if clip_log is None:
            values["token_ratio_gt_clip_fraction"].append(
                torch.zeros((), dtype=torch.float64, device=active.device)
            )
        else:
            values["token_ratio_gt_clip_fraction"].append(
                (active > clip_log).to(torch.float64).mean()
            )
        values["token_abs_delta_gt_1_fraction"].append(
            (active.abs() > 1.0).to(torch.float64).mean()
        )

        trim_count = math.floor(trim_fraction * count)
        if trim_count:
            remove = torch.topk(active.abs(), k=trim_count, largest=True).indices
            keep = torch.ones(count, dtype=torch.bool, device=active.device)
            keep[remove] = False
            trimmed = active[keep]
        else:
            trimmed = active
        values["trimmed_token_count_1pct"].append(
            torch.tensor(trimmed.numel(), dtype=torch.int64, device=active.device)
        )
        values["trimmed_raw_log_rho_1pct"].append(trimmed.sum())

    return {name: torch.stack(items) for name, items in values.items()}


def compute_actual_token_is_ratio_summary(
    importance_sampling_ratio: torch.Tensor,
    mask: torch.Tensor,
    *,
    clip_max: float | None,
) -> dict[str, torch.Tensor]:
    """Summarize the post-truncate token ratios actually multiplied into DAPO loss."""
    if importance_sampling_ratio.ndim != 2:
        raise ValueError("actual token IS ratios must be rank-2 [batch, tokens]")
    if importance_sampling_ratio.shape != mask.shape:
        raise ValueError("actual token IS ratios must match the active loss-mask shape")
    if clip_max is not None and clip_max <= 0:
        raise ValueError("clip_max must be positive when supplied")

    active_mask = mask.to(dtype=torch.bool)
    values: dict[str, list[torch.Tensor]] = {
        "actual_is_ratio_count": [],
        "actual_is_ratio_mean": [],
        "actual_is_ratio_std": [],
        "actual_is_ratio_min": [],
        "actual_is_ratio_max": [],
        "actual_is_ratio_sum": [],
        "actual_is_ratio_sq_sum": [],
        "actual_is_log_ratio_mean": [],
        "actual_is_ratio_at_upper_cap_fraction": [],
    }

    for row_index in range(importance_sampling_ratio.shape[0]):
        active = importance_sampling_ratio[row_index][active_mask[row_index]].to(torch.float64)
        if active.numel() == 0:
            raise ValueError("actual token IS summary requires at least one active token per rollout")
        if not torch.isfinite(active).all():
            raise ValueError("actual token IS ratios must be finite on active tokens")
        if (active <= 0).any():
            raise ValueError("actual token IS ratios must be positive on active tokens")

        values["actual_is_ratio_count"].append(
            torch.tensor(active.numel(), dtype=torch.int64, device=active.device)
        )
        values["actual_is_ratio_mean"].append(active.mean())
        values["actual_is_ratio_std"].append(active.std(correction=0))
        values["actual_is_ratio_min"].append(active.min())
        values["actual_is_ratio_max"].append(active.max())
        values["actual_is_ratio_sum"].append(active.sum())
        values["actual_is_ratio_sq_sum"].append((active * active).sum())
        values["actual_is_log_ratio_mean"].append(torch.log(active).mean())
        if clip_max is None:
            values["actual_is_ratio_at_upper_cap_fraction"].append(
                torch.zeros((), dtype=torch.float64, device=active.device)
            )
        else:
            values["actual_is_ratio_at_upper_cap_fraction"].append(
                (active >= clip_max).to(torch.float64).mean()
            )

    return {name: torch.stack(items) for name, items in values.items()}


def compute_effective_log_rho(importance_sampling_ratio: torch.Tensor) -> torch.Tensor:
    """Log of a scalar post-mask sequence ratio used by legacy sequence-mode pilots."""
    return torch.log(importance_sampling_ratio)


def infer_upper_cap_mask(raw_log_rho: torch.Tensor, *, clip_max: float | None) -> torch.Tensor:
    """Infer legacy sequence-mask rejection from the pre-exp sequence ratio."""
    if clip_max is None:
        return torch.zeros_like(raw_log_rho, dtype=torch.bool)
    if clip_max <= 0:
        raise ValueError("clip_max must be positive when supplied")
    return raw_log_rho > math.log(clip_max)


class RewardBatchRecorder:
    """One-generation handoff from the canonical reward to the trainer ledger."""

    def __init__(self):
        self._pending: list[dict] | None = None

    def capture(
        self,
        *,
        dataset_indices,
        correctness,
        terminated,
        rewards,
        completion_lengths,
    ) -> None:
        if self._pending is not None:
            raise RuntimeError("Signal ledger reward batch was not consumed before the next generation")

        values = [
            list(dataset_indices),
            list(correctness),
            list(terminated),
            list(rewards),
            list(completion_lengths),
        ]
        lengths = {len(value) for value in values}
        if len(lengths) != 1:
            raise ValueError("Signal ledger reward metadata lengths must match")

        self._pending = [
            {
                "dataset_index": int(dataset_index),
                "correct": bool(correct),
                "terminated": bool(did_terminate),
                "canonical_reward": float(reward),
                "completion_length": int(completion_length),
            }
            for dataset_index, correct, did_terminate, reward, completion_length in zip(*values, strict=True)
        ]

    def pop(self) -> list[dict]:
        if self._pending is None:
            raise RuntimeError("Signal ledger reward metadata is missing for this generation")
        pending = self._pending
        self._pending = None
        return pending


def utc_launch_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _optional_scalar(tensor: torch.Tensor | None, index: int) -> float | None:
    if tensor is None:
        return None
    value = float(tensor[index].item())
    return value if math.isfinite(value) else None


def _optional_int(tensor: torch.Tensor | None, index: int) -> int | None:
    if tensor is None:
        return None
    return int(tensor[index].item())


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
            handle.write(
                json.dumps(
                    {field: row[field] for field in LEDGER_FIELDS},
                    sort_keys=True,
                    allow_nan=False,
                )
            )
            handle.write("\n")


def make_signal_ledger_trainer(
    base_trainer_class,
    *,
    recorder: RewardBatchRecorder,
    ledger_dir: Path,
    importance_sampling_clip_max: float | None,
    importance_sampling_mode: str,
    launch_timestamp: str,
):
    """Wrap TRL's trainer without copying its generation or loss implementation."""
    if importance_sampling_mode not in {
        "sequence_mask",
        "sequence_truncate",
        "token_mask",
        "token_truncate",
    }:
        raise ValueError(f"Unsupported importance-sampling mode: {importance_sampling_mode}")

    class SignalLedgerGRPOTrainer(base_trainer_class):
        def _generate_and_score_completions(self, inputs):
            output = super()._generate_and_score_completions(inputs)
            metadata = recorder.pop()

            local_count = len(metadata)
            advantages = output["advantages"]
            if int(advantages.shape[0]) != local_count:
                raise RuntimeError(
                    "Signal ledger metadata count does not match local GRPO advantages: "
                    f"{local_count} != {int(advantages.shape[0])}"
                )

            device = advantages.device
            local_rewards = torch.tensor(
                [row["canonical_reward"] for row in metadata],
                dtype=torch.float32,
                device=device,
            )
            global_rewards = self.accelerator.gather(local_rewards)

            group_size = int(self.num_generations if self.model.training else self.num_generations_eval)
            if group_size <= 0 or global_rewards.numel() % group_size != 0:
                raise RuntimeError(
                    "Signal ledger cannot reconstruct reward groups: "
                    f"n={global_rewards.numel()}, G={group_size}"
                )

            grouped_rewards = global_rewards.view(-1, group_size)
            group_successes = grouped_rewards.sum(dim=1).repeat_interleave(group_size)
            if group_size > 1:
                group_reward_std = torch.std(
                    grouped_rewards,
                    dim=1,
                    correction=1,
                ).repeat_interleave(group_size)
            else:
                group_reward_std = torch.zeros_like(group_successes)

            start = self.accelerator.process_index * local_count
            stop = start + local_count
            local_group_successes = group_successes[start:stop]
            local_group_reward_std = group_reward_std[start:stop]
            if local_group_successes.numel() != local_count:
                raise RuntimeError("Signal ledger process slice does not match local generation batch")

            raw_log_rho = None
            effective_log_rho = None
            post_is_ratio = output.get("importance_sampling_ratio")
            upper_cap_masked = None
            token_summary = None
            actual_is_summary = None
            loss_mask = None

            old_per_token_logps = output.get("old_per_token_logps")
            sampling_per_token_logps = output.get("sampling_per_token_logps")
            if old_per_token_logps is not None and sampling_per_token_logps is not None:
                loss_mask = output["completion_mask"]
                if "tool_mask" in output:
                    loss_mask = loss_mask * output["tool_mask"]
                raw_log_rho = compute_raw_sequence_log_rho(
                    old_per_token_logps,
                    sampling_per_token_logps,
                    loss_mask,
                )
                token_summary = compute_token_logprob_summary(
                    old_per_token_logps,
                    sampling_per_token_logps,
                    loss_mask,
                    clip_max=importance_sampling_clip_max,
                    trim_fraction=0.01,
                )

            sequence_mode = importance_sampling_mode.startswith("sequence_")
            token_mode = importance_sampling_mode.startswith("token_")
            if post_is_ratio is not None:
                if sequence_mode:
                    if post_is_ratio.ndim != 2 or post_is_ratio.shape[1] != 1:
                        raise RuntimeError(
                            "Sequence-level vLLM IS mode must return ratios with shape [B,1]"
                        )
                    effective_log_rho = compute_effective_log_rho(post_is_ratio)
                    if importance_sampling_mode == "sequence_mask" and raw_log_rho is not None:
                        upper_cap_masked = infer_upper_cap_mask(
                            raw_log_rho,
                            clip_max=importance_sampling_clip_max,
                        )
                    else:
                        upper_cap_masked = torch.zeros_like(post_is_ratio, dtype=torch.bool)
                elif token_mode:
                    if loss_mask is None:
                        raise RuntimeError("Token-level vLLM IS requires a completion loss mask")
                    actual_is_summary = compute_actual_token_is_ratio_summary(
                        post_is_ratio,
                        loss_mask,
                        clip_max=importance_sampling_clip_max,
                    )
                    upper_cap_masked = torch.zeros(
                        (local_count, 1), dtype=torch.bool, device=advantages.device
                    )

            rank = int(self.accelerator.process_index)
            ledger_path = Path(ledger_dir) / f"signal_ledger_{launch_timestamp}_rank{rank}.jsonl"
            rows = []
            for index, reward_metadata in enumerate(metadata):
                rows.append(
                    {
                        "generation_global_step": int(self.state.global_step),
                        "rank": rank,
                        **reward_metadata,
                        "group_successes": int(local_group_successes[index].item()),
                        "group_size": group_size,
                        "group_reward_std": float(local_group_reward_std[index].item()),
                        "advantage": float(advantages[index].item()),
                        "raw_log_rho": _optional_scalar(raw_log_rho, index),
                        "effective_log_rho": _optional_scalar(effective_log_rho, index),
                        "importance_sampling_ratio": (
                            _optional_scalar(post_is_ratio, index) if sequence_mode else None
                        ),
                        "upper_cap_masked": (
                            bool(upper_cap_masked[index].item()) if upper_cap_masked is not None else None
                        ),
                        "token_delta_count": _optional_int(
                            None if token_summary is None else token_summary["token_delta_count"], index
                        ),
                        "token_delta_mean": _optional_scalar(
                            None if token_summary is None else token_summary["token_delta_mean"], index
                        ),
                        "token_delta_std": _optional_scalar(
                            None if token_summary is None else token_summary["token_delta_std"], index
                        ),
                        "token_delta_min": _optional_scalar(
                            None if token_summary is None else token_summary["token_delta_min"], index
                        ),
                        "token_delta_max": _optional_scalar(
                            None if token_summary is None else token_summary["token_delta_max"], index
                        ),
                        "token_delta_positive_fraction": _optional_scalar(
                            None if token_summary is None else token_summary["token_delta_positive_fraction"], index
                        ),
                        "token_ratio_sum": _optional_scalar(
                            None if token_summary is None else token_summary["token_ratio_sum"], index
                        ),
                        "token_ratio_sq_sum": _optional_scalar(
                            None if token_summary is None else token_summary["token_ratio_sq_sum"], index
                        ),
                        "token_ratio_gt_clip_fraction": _optional_scalar(
                            None if token_summary is None else token_summary["token_ratio_gt_clip_fraction"], index
                        ),
                        "token_abs_delta_gt_1_fraction": _optional_scalar(
                            None if token_summary is None else token_summary["token_abs_delta_gt_1_fraction"], index
                        ),
                        "trimmed_token_count_1pct": _optional_int(
                            None if token_summary is None else token_summary["trimmed_token_count_1pct"], index
                        ),
                        "trimmed_raw_log_rho_1pct": _optional_scalar(
                            None if token_summary is None else token_summary["trimmed_raw_log_rho_1pct"], index
                        ),
                        "actual_is_ratio_count": _optional_int(
                            None if actual_is_summary is None else actual_is_summary["actual_is_ratio_count"], index
                        ),
                        "actual_is_ratio_mean": _optional_scalar(
                            None if actual_is_summary is None else actual_is_summary["actual_is_ratio_mean"], index
                        ),
                        "actual_is_ratio_std": _optional_scalar(
                            None if actual_is_summary is None else actual_is_summary["actual_is_ratio_std"], index
                        ),
                        "actual_is_ratio_min": _optional_scalar(
                            None if actual_is_summary is None else actual_is_summary["actual_is_ratio_min"], index
                        ),
                        "actual_is_ratio_max": _optional_scalar(
                            None if actual_is_summary is None else actual_is_summary["actual_is_ratio_max"], index
                        ),
                        "actual_is_ratio_sum": _optional_scalar(
                            None if actual_is_summary is None else actual_is_summary["actual_is_ratio_sum"], index
                        ),
                        "actual_is_ratio_sq_sum": _optional_scalar(
                            None if actual_is_summary is None else actual_is_summary["actual_is_ratio_sq_sum"], index
                        ),
                        "actual_is_log_ratio_mean": _optional_scalar(
                            None if actual_is_summary is None else actual_is_summary["actual_is_log_ratio_mean"], index
                        ),
                        "actual_is_ratio_at_upper_cap_fraction": _optional_scalar(
                            None
                            if actual_is_summary is None
                            else actual_is_summary["actual_is_ratio_at_upper_cap_fraction"],
                            index,
                        ),
                    }
                )
            append_ledger_rows(ledger_path, rows)
            return output

    SignalLedgerGRPOTrainer.__name__ = "SignalLedgerGRPOTrainer"
    return SignalLedgerGRPOTrainer
