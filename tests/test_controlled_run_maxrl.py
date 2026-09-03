from __future__ import annotations

import importlib

import pytest
import torch


def _maxrl():
    return importlib.import_module("controlled_run.maxrl")


def test_practical_maxrl_all_failure_group_is_zero():
    maxrl = _maxrl()
    rewards = torch.zeros(16, dtype=torch.float32)

    advantages = maxrl.compute_practical_maxrl_advantages(rewards, group_size=16)

    assert torch.equal(advantages, torch.zeros_like(rewards))


def test_practical_maxrl_all_success_group_is_zero():
    maxrl = _maxrl()
    rewards = torch.ones(16, dtype=torch.float32)

    advantages = maxrl.compute_practical_maxrl_advantages(rewards, group_size=16)

    assert torch.equal(advantages, torch.zeros_like(rewards))


def test_practical_maxrl_k1_has_success_15_and_failures_minus_one():
    maxrl = _maxrl()
    rewards = torch.tensor([1.0] + [0.0] * 15)

    advantages = maxrl.compute_practical_maxrl_advantages(rewards, group_size=16)

    assert advantages[0].item() == 15.0
    assert torch.equal(advantages[1:], torch.full((15,), -1.0))
    assert advantages.sum().item() == 0.0


def test_practical_maxrl_k4_has_success_3_and_failures_minus_one():
    maxrl = _maxrl()
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0] + [0.0] * 9)

    advantages = maxrl.compute_practical_maxrl_advantages(rewards, group_size=16)

    assert torch.equal(advantages[rewards == 1], torch.full((4,), 3.0))
    assert torch.equal(advantages[rewards == 0], torch.full((12,), -1.0))
    assert advantages.sum().item() == 0.0


def test_practical_maxrl_handles_multiple_groups_independently():
    maxrl = _maxrl()
    rewards = torch.tensor(
        [1.0] + [0.0] * 15
        + [1.0] * 4 + [0.0] * 12
        + [0.0] * 16
        + [1.0] * 16,
        dtype=torch.float64,
    )

    advantages = maxrl.compute_practical_maxrl_advantages(rewards, group_size=16)
    grouped = advantages.reshape(-1, 16)

    assert torch.allclose(grouped.sum(dim=1), torch.zeros(4, dtype=torch.float64), atol=0, rtol=0)
    assert grouped[0, 0].item() == 15.0
    assert grouped[1, 0].item() == 3.0
    assert torch.equal(grouped[2], torch.zeros(16, dtype=torch.float64))
    assert torch.equal(grouped[3], torch.zeros(16, dtype=torch.float64))


@pytest.mark.parametrize(
    "rewards",
    [
        torch.tensor([0.0, 0.5, 1.0, 0.0]),
        torch.tensor([0.0, float("nan"), 1.0, 0.0]),
        torch.tensor([0.0, float("inf"), 1.0, 0.0]),
    ],
)
def test_practical_maxrl_rejects_nonbinary_or_nonfinite_rewards(rewards):
    maxrl = _maxrl()

    with pytest.raises(ValueError):
        maxrl.compute_practical_maxrl_advantages(rewards, group_size=4)


@pytest.mark.parametrize("group_size", [0, 1, -4])
def test_practical_maxrl_rejects_invalid_group_size(group_size):
    maxrl = _maxrl()

    with pytest.raises(ValueError):
        maxrl.compute_practical_maxrl_advantages(torch.zeros(16), group_size=group_size)


def test_practical_maxrl_rejects_grouping_mismatch():
    maxrl = _maxrl()

    with pytest.raises(ValueError):
        maxrl.compute_practical_maxrl_advantages(torch.zeros(17), group_size=16)


def test_practical_maxrl_rejects_nonvector_rewards():
    maxrl = _maxrl()

    with pytest.raises(ValueError):
        maxrl.compute_practical_maxrl_advantages(torch.zeros((2, 16)), group_size=16)


def test_practical_maxrl_metadata_records_effective_order_n_minus_one():
    maxrl = _maxrl()

    metadata = maxrl.practical_maxrl_metadata(group_size=16)

    assert metadata == {
        "advantage_estimator": "practical_maxrl",
        "rollouts_per_prompt": 16,
        "effective_maxrl_order": 15,
        "all_failure_behavior": "zero_group_gradient",
        "maxrl_denominator_epsilon": 0.0,
    }
