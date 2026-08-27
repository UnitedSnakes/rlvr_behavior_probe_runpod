from __future__ import annotations

import json
from pathlib import Path

import controlled_run.train_sft as train_sft


ROOT = Path(__file__).resolve().parents[1]


class FakeSFTConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _write_record(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "uuid": "u1",
                "prompt": [{"role": "user", "content": "1+1?"}],
                "completion": [{"role": "assistant", "content": "\\boxed{2}"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_sources(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "base_model": {
                    "repo_id": "Qwen/Qwen3-0.6B-Base",
                    "sha": "base-sha",
                },
                "sft_dataset": {
                    "repo_id": "open-r1/OpenR1-Math-220k",
                    "sha": "sft-sha",
                },
            }
        ),
        encoding="utf-8",
    )


def test_run_sft_canonical_freezes_exact_pi0_and_writes_run_manifest(
    monkeypatch,
    tmp_path,
):
    config_path = ROOT / "controlled_run/configs/sft_qwen3_0_6b.yaml"
    records = tmp_path / "records.jsonl"
    sources = tmp_path / "source_revisions.json"
    sft_manifest = tmp_path / "sft_10k_manifest.jsonl"
    _write_record(records)
    _write_sources(sources)
    sft_manifest.write_text('{"uuid":"u1"}\n', encoding="utf-8")

    monkeypatch.setattr(
        train_sft,
        "validate_record_count",
        lambda path, canonical: 10_000 if canonical else 1,
    )

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, repo_id, **kwargs):
            return "MODEL"

    class FakeTokenizer:
        def save_pretrained(self, output_dir):
            path = Path(output_dir)
            path.mkdir(parents=True, exist_ok=True)
            (path / "tokenizer.json").write_text("{}", encoding="utf-8")

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, repo_id, **kwargs):
            return FakeTokenizer()

    class FakeTrainer:
        def __init__(self, **kwargs):
            pass

        def train(self):
            pass

        def save_model(self, output_dir):
            path = Path(output_dir)
            path.mkdir(parents=True, exist_ok=True)
            (path / "model.safetensors").write_bytes(b"canonical-final")

    monkeypatch.setattr(train_sft, "AutoModelForCausalLM", FakeAutoModel)
    monkeypatch.setattr(train_sft, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(
        train_sft,
        "_load_sft_classes",
        lambda: (FakeSFTConfig, FakeTrainer),
    )

    output_dir = tmp_path / "out"
    result = train_sft.run_sft(
        config_path=config_path,
        records_path=records,
        source_revisions_path=sources,
        sft_manifest_path=sft_manifest,
        output_dir=output_dir,
    )

    assert result["mode"] == "canonical"
    assert result["record_count"] == 10_000
    assert result["pi0_manifest"]["policy_name"] == "pi_0"
    assert (output_dir / "pi_0" / "model.safetensors").read_bytes() == b"canonical-final"
    assert (output_dir / "pi_0" / "pi0_manifest.json").exists()

    run_manifest = json.loads(
        (output_dir / "sft_run_manifest.json").read_text(encoding="utf-8")
    )
    assert run_manifest["mode"] == "canonical"
    assert run_manifest["record_count"] == 10_000
    assert run_manifest["config"]["model_name"] == "Qwen/Qwen3-0.6B-Base"
    assert run_manifest["lineage"] == {
        key: result["pi0_manifest"][key]
        for key in (
            "base_model_sha",
            "sft_dataset_sha",
            "sft_data_manifest_sha256",
            "sft_config_sha256",
        )
    }


def test_main_forwards_explicit_paths_and_smoke_steps(monkeypatch, tmp_path):
    calls = []

    def fake_run_sft(**kwargs):
        calls.append(kwargs)
        return {"mode": "smoke", "record_count": 1, "smoke_steps": 2}

    monkeypatch.setattr(train_sft, "run_sft", fake_run_sft)

    train_sft.main(
        [
            "--config",
            str(tmp_path / "sft.yaml"),
            "--records",
            str(tmp_path / "records.jsonl"),
            "--source-revisions",
            str(tmp_path / "revisions.json"),
            "--sft-manifest",
            str(tmp_path / "manifest.jsonl"),
            "--output-dir",
            str(tmp_path / "out"),
            "--smoke-steps",
            "2",
        ]
    )

    assert calls == [
        {
            "config_path": tmp_path / "sft.yaml",
            "records_path": tmp_path / "records.jsonl",
            "source_revisions_path": tmp_path / "revisions.json",
            "sft_manifest_path": tmp_path / "manifest.jsonl",
            "output_dir": tmp_path / "out",
            "smoke_steps": 2,
        }
    ]
