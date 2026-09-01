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
)


def compute_raw_sequence_log_rho(
    old_per_token_logps: torch.Tensor,
    sampling_per_token_logps: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct the sequence log-ratio before exp/masking exactly as TRL does."""
    if old_per_token_logps.shape != sampling_per_token_logps.shape:
        raise ValueError("old and sampling log-probability tensors must have matching shapes")
    if mask.shape != old_per_token_logps.shape:
        raise ValueError("importance-sampling mask must match per-token log-probability shape")

    per_token_diff = (old_per_token_logps - sampling_per_token_logps) * mask
    per_token_diff = torch.nan_to_num(per_token_diff, nan=0.0)
    return per_token_diff.sum(dim=-1, keepdim=True)


def compute_effective_log_rho(importance_sampling_ratio: torch.Tensor) -> torch.Tensor:
    """Log of the post-mask ratio that DAPO actually multiplies into the loss.

    For the canonical DAPO path TRL does not clamp this value again in log
    space. A zero post-mask ratio therefore maps to -inf. The raw log-ratio and
    upper-cap flag are logged separately so an upper-cap rejection can be
    distinguished from floating-point underflow.
    """
    return torch.log(importance_sampling_ratio)


def infer_upper_cap_mask(raw_log_rho: torch.Tensor, *, clip_max: float | None) -> torch.Tensor:
    """Infer sequence-mask rejection from the pre-exp ratio, before zero-fill erases provenance."""
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
    launch_timestamp: str,
):
    """Wrap TRL's trainer without copying its generation or loss implementation."""

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
            post_mask_ratio = output.get("importance_sampling_ratio")
            upper_cap_masked = None

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
                upper_cap_masked = infer_upper_cap_mask(
                    raw_log_rho,
                    clip_max=importance_sampling_clip_max,
                )

            if post_mask_ratio is not None:
                if post_mask_ratio.ndim != 2 or post_mask_ratio.shape[1] != 1:
                    raise RuntimeError(
                        "Canonical signal ledger expects sequence-level vLLM importance sampling ratios"
                    )
                effective_log_rho = compute_effective_log_rho(post_mask_ratio)

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
                        "importance_sampling_ratio": _optional_scalar(post_mask_ratio, index),
                        "upper_cap_masked": (
                            bool(upper_cap_masked[index].item()) if upper_cap_masked is not None else None
                        ),
                    }
                )
            append_ledger_rows(ledger_path, rows)
            return output

    SignalLedgerGRPOTrainer.__name__ = "SignalLedgerGRPOTrainer"
    return SignalLedgerGRPOTrainer
