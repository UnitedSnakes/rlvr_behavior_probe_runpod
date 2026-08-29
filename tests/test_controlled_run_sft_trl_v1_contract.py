from __future__ import annotations

from pathlib import Path

import controlled_run.train_sft as train_sft
from controlled_run.config import load_config


ROOT = Path(__file__).resolve().parents[1]


class FakeSFTConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_sft_adapter_maps_frozen_warmup_ratio_to_transformers_v5_warmup_steps(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        train_sft,
        "_load_sft_classes",
        lambda: (FakeSFTConfig, object),
    )
    config = load_config(ROOT / "controlled_run/configs/sft_qwen3_0_6b.yaml")

    args = train_sft.build_sft_arguments(config, tmp_path / "trainer")

    assert config["warmup_ratio"] == 0.03
    assert args.kwargs["warmup_steps"] == 0.03
    assert "warmup_ratio" not in args.kwargs
