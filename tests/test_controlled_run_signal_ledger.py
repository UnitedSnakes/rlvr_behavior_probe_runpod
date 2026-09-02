from __future__ import annotations

import importlib
import json
import math

import torch


def test_sequence_log_rho_keeps_raw_tail_and_matches_dapo_effective_ratio():
    ledger = importlib.import_module("controlled_run.signal_ledger")

    old = torch.tensor([[-50.0, -50.0], [math.log(2.0), math.log(2.0)]])
    sampling = torch.zeros_like(old)
    mask = torch.ones_like(old)
    # In sequence_mask mode both numerical underflow and an upper-cap rejection
    # can appear as a post-mask ratio of exactly zero. The raw log ratio keeps
    # those cases distinguishable.
    post_mask_ratio = torch.tensor([[0.0], [0.0]])

    raw = ledger.compute_raw_sequence_log_rho(old, sampling, mask)
    effective = ledger.compute_effective_log_rho(post_mask_ratio)
    upper_masked = ledger.infer_upper_cap_mask(raw, clip_max=3.0)

    assert torch.allclose(raw[:, 0], torch.tensor([-100.0, math.log(4.0)]), atol=1e-6)
    assert torch.isneginf(effective[:, 0]).tolist() == [True, True]
    assert upper_masked[:, 0].tolist() == [False, True]


def test_token_logprob_summary_is_mask_aware_and_supports_trimmed_slope_and_ess():
    ledger = importlib.import_module("controlled_run.signal_ledger")

    delta = torch.zeros((1, 101), dtype=torch.float64)
    delta[0, 0] = 2.0
    delta[0, 1] = -0.5
    delta[0, 2] = 0.2
    delta[0, 100] = 100.0  # masked: must not enter any statistic
    old = delta.clone()
    sampling = torch.zeros_like(old)
    mask = torch.ones_like(old)
    mask[0, 100] = 0

    summary = ledger.compute_token_logprob_summary(
        old,
        sampling,
        mask,
        clip_max=3.0,
        trim_fraction=0.01,
    )

    active = delta[0, :100]
    expected_ratio_sum = torch.exp(active).sum()
    expected_ratio_sq_sum = torch.exp(2 * active).sum()

    assert summary["token_delta_count"][0].item() == 100
    assert torch.allclose(summary["token_delta_mean"][0], active.mean())
    assert torch.allclose(summary["token_delta_std"][0], active.std(correction=0))
    assert summary["token_delta_min"][0].item() == -0.5
    assert summary["token_delta_max"][0].item() == 2.0
    assert math.isclose(summary["token_delta_positive_fraction"][0].item(), 0.02)
    assert torch.allclose(summary["token_ratio_sum"][0], expected_ratio_sum)
    assert torch.allclose(summary["token_ratio_sq_sum"][0], expected_ratio_sq_sum)
    assert math.isclose(summary["token_ratio_gt_clip_fraction"][0].item(), 0.01)
    assert math.isclose(summary["token_abs_delta_gt_1_fraction"][0].item(), 0.01)
    # floor(1% * 100) = 1: remove the largest |delta| token, +2.0.
    assert summary["trimmed_token_count_1pct"][0].item() == 99
    assert math.isclose(summary["trimmed_raw_log_rho_1pct"][0].item(), -0.3, abs_tol=1e-12)


def test_signal_ledger_jsonl_schema_is_complete(tmp_path):
    ledger = importlib.import_module("controlled_run.signal_ledger")
    row = {
        "generation_global_step": 12,
        "rank": 1,
        "dataset_index": 345,
        "correct": True,
        "terminated": True,
        "canonical_reward": 1.0,
        "completion_length": 731,
        "group_successes": 5,
        "group_size": 16,
        "group_reward_std": 0.5,
        "advantage": 1.2,
        "raw_log_rho": -4.3,
        "effective_log_rho": -4.3,
        "importance_sampling_ratio": 0.0136,
        "upper_cap_masked": False,
        "token_delta_count": 731,
        "token_delta_mean": -0.0059,
        "token_delta_std": 0.11,
        "token_delta_min": -0.8,
        "token_delta_max": 0.7,
        "token_delta_positive_fraction": 0.49,
        "token_ratio_sum": 729.0,
        "token_ratio_sq_sum": 735.0,
        "token_ratio_gt_clip_fraction": 0.0,
        "token_abs_delta_gt_1_fraction": 0.0,
        "trimmed_token_count_1pct": 724,
        "trimmed_raw_log_rho_1pct": -3.9,
    }

    path = tmp_path / "signal_ledger.jsonl"
    ledger.append_ledger_rows(path, [row])
    payload = json.loads(path.read_text(encoding="utf-8").strip())

    assert set(payload) == set(ledger.LEDGER_FIELDS)
    assert payload["dataset_index"] == 345
