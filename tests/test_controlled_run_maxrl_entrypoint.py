from __future__ import annotations

from pathlib import Path

from controlled_run.config import load_config
from controlled_run.maxrl import MaxRLRewardBatchRecorder
import controlled_run.train_maxrl as train_maxrl


ROOT = Path(__file__).resolve().parents[1]


def test_maxrl_objective_metadata_freezes_g16_as_practical_order15():
    config = load_config(ROOT / "controlled_run/configs/grpo_qwen3_0_6b.yaml")

    metadata = train_maxrl.build_maxrl_objective_metadata(config)

    assert metadata["objective_family"] == "MaxRL"
    assert metadata["objective_intervention"] == "replace_group_advantages_only"
    assert metadata["advantage_estimator"] == "practical_maxrl"
    assert metadata["rollouts_per_prompt"] == 16
    assert metadata["effective_maxrl_order"] == 15
    assert metadata["all_failure_behavior"] == "zero_group_gradient"
    assert metadata["maxrl_denominator_epsilon"] == 0.0
    assert metadata["trainer_composition"] == [
        "trl.GRPOTrainer",
        "PracticalMaxRLTrainer",
        "SignalLedgerGRPOTrainer",
    ]
    assert metadata["grouping_semantics"] == "trl_global_reward_order_grouped_by_num_generations"


def test_run_maxrl_delegates_to_shared_core_with_frozen_objective_hooks(monkeypatch, tmp_path):
    config_path = ROOT / "controlled_run/configs/grpo_qwen3_0_6b.yaml"
    config = load_config(config_path)
    captured = {}
    sentinel_result = {"status": "delegated"}

    monkeypatch.setattr(train_maxrl, "load_config", lambda _path: dict(config))

    def fake_core(**kwargs):
        captured.update(kwargs)
        return sentinel_result

    monkeypatch.setattr(train_maxrl, "_run_controlled_grpo", fake_core)

    result = train_maxrl.run_maxrl(
        config_path=config_path,
        pi0_dir=tmp_path / "pi0",
        output_dir=tmp_path / "maxrl",
        mode="pilot",
        pilot_steps=20,
    )

    assert result is sentinel_result
    assert captured["config_path"] == config_path
    assert captured["pi0_dir"] == tmp_path / "pi0"
    assert captured["output_dir"] == tmp_path / "maxrl"
    assert captured["mode"] == "pilot"
    assert captured["pilot_steps"] == 20
    assert captured["recorder_factory"] is MaxRLRewardBatchRecorder
    assert captured["manifest_filename"] == "maxrl_run_manifest.json"

    objective = captured["manifest_extra"]["objective"]
    assert objective["advantage_estimator"] == "practical_maxrl"
    assert objective["rollouts_per_prompt"] == 16
    assert objective["effective_maxrl_order"] == 15
    assert captured["result_extra"] == {"objective": objective}

    recorder = MaxRLRewardBatchRecorder()

    class FakeBaseTrainer:
        pass

    wrapped = captured["trainer_transform"](FakeBaseTrainer, recorder=recorder)
    assert wrapped.__name__ == "PracticalMaxRLTrainer"
    assert issubclass(wrapped, FakeBaseTrainer)


def test_maxrl_entrypoint_uses_distinct_default_output_directory():
    assert train_maxrl.DEFAULT_OUTPUT_DIR == Path("controlled_run_outputs/maxrl")
    assert train_maxrl.DEFAULT_CONFIG == Path("controlled_run/configs/grpo_qwen3_0_6b.yaml")
