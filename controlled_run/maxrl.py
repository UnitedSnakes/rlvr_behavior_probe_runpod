from __future__ import annotations

import torch


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
    the GRPO generated batch after cross-process gathering.
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
