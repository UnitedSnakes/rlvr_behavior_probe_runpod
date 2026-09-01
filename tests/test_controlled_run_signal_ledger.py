from __future__ import annotations

import importlib
import json
import math

import torch


def test_sequence_log_rho_keeps_raw_tail_and_matches_effective_trl_clamp():
    ledger = importlib.import_module("controlled_run.signal_ledger")

    old = torch.tensor([[-50.0, -50.0], [math.log(2.0), math.log(2.0)]])
    sampling = torch.zeros_like(old)
    mask = torch.ones_like(old)
    post_mask_ratio = torch.tensor([[0.0], [0.0]])

    raw = ledger.compute_raw_sequence_log_rho(old, sampling, mask)
    effective = ledger.compute_effective_log_rho(post_mask_ratio)
    upper_masked = ledger.infer_upper_cap_mask(raw, clip_max=3.0)

    assert torch.allclose(raw[:, 0], torch.tensor([-100.0, math.log(4.0)]), atol=1e-6)
    assert torch.allclose(
        effective[:, 0],
        torch.tensor([math.log(1e-8), math.log(1e-8)]),
        atol=1e-6,
    )
    assert upper_masked[:, 0].tolist() == [False, True]


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
    }

    path = tmp_path / "signal_ledger.jsonl"
    ledger.append_ledger_rows(path, [row])
    payload = json.loads(path.read_text(encoding="utf-8").strip())

    assert set(payload) == set(ledger.LEDGER_FIELDS)
    assert payload["dataset_index"] == 345
