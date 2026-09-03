from __future__ import annotations

import torch

from controlled_run.signal_ledger import RewardBatchRecorder


ADVANTAGE_ESTIMATOR = "practical_maxrl"
ALL_FAILURE_BEHAVIOR = "zero_group_gradient"
DENOMINATOR_EPSILON = 0.0


def _validate_group_size(group_size: int) -> int:
    if isinstance(group_size, bool) or not isinstance(group_size, int):
        raise ValueError("group_size must be an integer")
    if group_size <= 1:
        raise ValueError("group_size must be greater than 1 for practical MaxRL")
    return group_size


def _validate_binary_rewards(rewards: torch.Tensor, group_size: int) -> None:
    if not isinstance(rewards, torch.Tensor):
        raise ValueError("rewards must be a torch.Tensor")
    if rewards.ndim != 1:
        raise ValueError("rewards must be a one-dimensional tensor")
    if not rewards.is_floating_point():
        raise ValueError("rewards must use a floating-point dtype")
    if rewards.numel() == 0 or rewards.numel() % group_size != 0:
        raise ValueError(
            "reward count must be nonzero and divisible by group_size; "
            f"got {rewards.numel()} rewards for group_size={group_size}"
        )
    if not torch.isfinite(rewards).all().item():
        raise ValueError("rewards must all be finite")
    if not ((rewards == 0) | (rewards == 1)).all().item():
        raise ValueError("practical MaxRL requires binary rewards in {0, 1}")


def compute_practical_maxrl_advantages(
    rewards: torch.Tensor,
    *,
    group_size: int,
) -> torch.Tensor:
    """Compute the dropped-baseline practical MaxRL group advantage.

    For each binary-reward group of size N, let p_hat = K / N. The estimator is

        A_i = 0                            if K = 0
        A_i = (r_i - p_hat) / p_hat        if K > 0.

    With the all-failure baseline dropped, the N-rollout estimator corresponds
    to the order-(N-1) truncated MaxRL objective proved in Theorem 5 of
    Tajwar et al. (2026). No denominator epsilon is used: for an active finite
    group, p_hat is at least 1/N.

    The input is expected to contain complete prompt groups contiguously, as in
    TRL's globally aggregated GRPO reward tensor before its rank-local slice.
    """
    group_size = _validate_group_size(group_size)
    _validate_binary_rewards(rewards, group_size)

    grouped = rewards.reshape(-1, group_size)
    p_hat = grouped.mean(dim=1, keepdim=True)
    active = p_hat > 0

    advantages = torch.zeros_like(grouped)
    if active.any().item():
        active_rows = active.squeeze(1)
        active_rewards = grouped[active_rows]
        active_p_hat = p_hat[active_rows]
        advantages[active_rows] = (active_rewards - active_p_hat) / active_p_hat

    return advantages.reshape_as(rewards)


def practical_maxrl_metadata(*, group_size: int) -> dict[str, object]:
    """Return the frozen provenance fields for the practical estimator."""
    group_size = _validate_group_size(group_size)
    return {
        "advantage_estimator": ADVANTAGE_ESTIMATOR,
        "rollouts_per_prompt": group_size,
        "effective_maxrl_order": group_size - 1,
        "all_failure_behavior": ALL_FAILURE_BEHAVIOR,
        "maxrl_denominator_epsilon": DENOMINATOR_EPSILON,
    }


class MaxRLRewardBatchRecorder(RewardBatchRecorder):
    """Reward handoff that MaxRL can inspect before the ledger consumes it."""

    def peek(self) -> list[dict]:
        if self._pending is None:
            raise RuntimeError("MaxRL reward metadata is missing for this generation")
        return [dict(row) for row in self._pending]


def _replace_latest_trl_advantage_log(trainer, global_advantages: torch.Tensor) -> None:
    """Make TRL's completion-table advantage log match the actual MaxRL signal.

    TRL logs its global GRPO advantages inside the base scoring method before
    this wrapper can replace the returned local tensor. When that standard log
    buffer is present, replace exactly the latest global batch in place so the
    diagnostic table and the loss/ledger share the same advantage semantics.
    """
    logs = getattr(trainer, "_logs", None)
    if logs is None or "advantages" not in logs:
        return

    advantage_log = logs["advantages"]
    if not isinstance(advantage_log, list):
        raise RuntimeError("Practical MaxRL expects TRL _logs['advantages'] to be a list")

    batch_size = int(global_advantages.numel())
    if len(advantage_log) < batch_size:
        raise RuntimeError(
            "TRL advantage log is shorter than the just-scored global MaxRL batch: "
            f"{len(advantage_log)} < {batch_size}"
        )
    advantage_log[-batch_size:] = global_advantages.tolist()


def make_practical_maxrl_trainer(base_trainer_class, *, recorder: MaxRLRewardBatchRecorder):
    """Wrap a TRL-compatible trainer by replacing only its group advantages.

    This wrapper is intended to sit *inside* the existing signal-ledger wrapper:

        TRL GRPOTrainer -> practical MaxRL wrapper -> signal-ledger wrapper

    The reward recorder is peeked here and consumed later by the ledger. This
    leaves the canonical GRPO ledger wrapper unchanged while ensuring that the
    ledger records the exact advantage tensor subsequently consumed by the loss.
    """

    class PracticalMaxRLTrainer(base_trainer_class):
        def _generate_and_score_completions(self, inputs):
            output = super()._generate_and_score_completions(inputs)
            metadata = recorder.peek()

            base_advantages = output.get("advantages")
            if base_advantages is None:
                raise RuntimeError("Practical MaxRL requires trainer output['advantages']")
            if base_advantages.ndim != 1:
                raise RuntimeError("Practical MaxRL expects local advantages to be one-dimensional")

            local_count = len(metadata)
            if int(base_advantages.shape[0]) != local_count:
                raise RuntimeError(
                    "MaxRL reward metadata count does not match local trainer advantages: "
                    f"{local_count} != {int(base_advantages.shape[0])}"
                )

            local_rewards = torch.tensor(
                [row["canonical_reward"] for row in metadata],
                dtype=torch.float32,
                device=base_advantages.device,
            )
            global_rewards = self.accelerator.gather(local_rewards)

            group_size = int(self.num_generations if self.model.training else self.num_generations_eval)
            global_advantages = compute_practical_maxrl_advantages(
                global_rewards,
                group_size=group_size,
            )
            _replace_latest_trl_advantage_log(self, global_advantages)

            start = int(self.accelerator.process_index) * local_count
            stop = start + local_count
            local_advantages = global_advantages[start:stop]
            if int(local_advantages.numel()) != local_count:
                raise RuntimeError(
                    "Practical MaxRL process slice does not match local generation batch: "
                    f"{int(local_advantages.numel())} != {local_count}"
                )

            output["advantages"] = local_advantages.to(
                device=base_advantages.device,
                dtype=base_advantages.dtype,
            )
            return output

    PracticalMaxRLTrainer.__name__ = "PracticalMaxRLTrainer"
    return PracticalMaxRLTrainer
