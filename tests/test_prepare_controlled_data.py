from __future__ import annotations

import json

import controlled_run.prepare_data as prepare_module
from controlled_run.constants import BASE_MODEL, GSM8K_DATASET, SFT_DATASET


def make_openr1_row(uuid: str, problem: str):
    return {
        "uuid": uuid,
        "problem": problem,
        "source": "synthetic",
        "generations": [f"reasoning {uuid} \\boxed{{1}}"],
        "is_reasoning_complete": [True],
        "correctness_math_verify": [True],
    }


class FakeTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
    ):
        assert tokenize is True
        assert add_generation_prompt is False
        return list(range(100))


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_prepare_data_pins_sources_and_writes_manifest_audit_and_full_records(
    monkeypatch,
    tmp_path,
):
    resolve_calls = []

    def fake_resolve(repo_id, revision, repo_type):
        resolve_calls.append((repo_id, revision, repo_type))
        return {
            BASE_MODEL: "base-sha",
            SFT_DATASET: "openr1-sha",
            GSM8K_DATASET: "gsm8k-sha",
        }[repo_id]

    monkeypatch.setattr(prepare_module, "resolve_hf_revision", fake_resolve)

    tokenizer_calls = []

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, repo_id, revision):
            tokenizer_calls.append((repo_id, revision))
            return FakeTokenizer()

    monkeypatch.setattr(prepare_module, "AutoTokenizer", FakeAutoTokenizer)

    openr1_rows = [
        make_openr1_row("u1", "safe problem one"),
        make_openr1_row("u2", "safe problem two"),
        make_openr1_row("u3", "safe problem three"),
    ]
    gsm_train = [{"question": "unrelated train question", "answer": "#### 1"}]
    gsm_test = [{"question": "unrelated test question", "answer": "#### 1"}]
    dataset_calls = []

    def fake_load_dataset(repo_id, config_name, *, split, revision):
        dataset_calls.append((repo_id, config_name, split, revision))
        if repo_id == SFT_DATASET:
            assert config_name == "default"
            assert split == "train"
            assert revision == "openr1-sha"
            return openr1_rows
        assert repo_id == GSM8K_DATASET
        assert config_name == "main"
        assert revision == "gsm8k-sha"
        return gsm_train if split == "train" else gsm_test

    monkeypatch.setattr(prepare_module, "load_dataset", fake_load_dataset)

    manifests_dir = tmp_path / "manifests"
    generated_dir = tmp_path / "generated"

    result = prepare_module.prepare_data(
        manifests_dir=manifests_dir,
        generated_dir=generated_dir,
        target_size=2,
        validation_size=1,
        seed=42,
    )

    assert resolve_calls == [
        (BASE_MODEL, "main", "model"),
        (SFT_DATASET, "main", "dataset"),
        (GSM8K_DATASET, "main", "dataset"),
    ]
    assert tokenizer_calls == [(BASE_MODEL, "base-sha")]
    assert dataset_calls == [
        (SFT_DATASET, "default", "train", "openr1-sha"),
        (GSM8K_DATASET, "main", "train", "gsm8k-sha"),
        (GSM8K_DATASET, "main", "test", "gsm8k-sha"),
    ]

    manifest = read_jsonl(manifests_dir / "sft_10k_manifest.jsonl")
    validation_manifest = read_jsonl(manifests_dir / "sft_val_512_manifest.jsonl")
    records = read_jsonl(generated_dir / "sft_10k_records.jsonl")
    validation_records = read_jsonl(generated_dir / "sft_val_512_records.jsonl")
    audit = json.loads(
        (manifests_dir / "contamination_audit.json").read_text(encoding="utf-8")
    )
    revisions = json.loads(
        (manifests_dir / "source_revisions.json").read_text(encoding="utf-8")
    )

    assert len(manifest) == 2
    assert len(validation_manifest) == 1
    assert len(records) == 2
    assert len(validation_records) == 1
    assert {row["source_index"] for row in manifest}.isdisjoint(
        {row["source_index"] for row in validation_manifest}
    )
    assert all("problem" not in row and "completion" not in row for row in manifest)
    assert all("prompt" in row and "completion" in row for row in records)
    assert audit["final_count"] == 3
    assert audit["audit_only"] is False
    assert audit["requested_total_count"] == 3
    assert audit["selected_total_count"] == 3
    assert audit["train_count"] == 2
    assert audit["validation_count"] == 1
    assert revisions == {
        "base_model": {
            "repo_id": BASE_MODEL,
            "requested_revision": "main",
            "sha": "base-sha",
        },
        "tokenizer": {
            "repo_id": BASE_MODEL,
            "requested_revision": "main",
            "sha": "base-sha",
        },
        "sft_dataset": {
            "repo_id": SFT_DATASET,
            "requested_revision": "main",
            "sha": "openr1-sha",
        },
        "gsm8k_dataset": {
            "repo_id": GSM8K_DATASET,
            "requested_revision": "main",
            "sha": "gsm8k-sha",
        },
        "seed": 42,
        "target_size": 2,
        "validation_size": 1,
        "source_identity": "pinned_dataset_revision_plus_source_index",
    }
    assert result == revisions


def test_main_forwards_cli_paths_target_size_and_seed(monkeypatch, tmp_path):
    calls = []

    def fake_prepare_data(**kwargs):
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(prepare_module, "prepare_data", fake_prepare_data)

    prepare_module.main(
        [
            "--manifests-dir",
            str(tmp_path / "m"),
            "--generated-dir",
            str(tmp_path / "g"),
            "--target-size",
            "17",
            "--seed",
            "9",
        ]
    )

    assert calls == [
        {
            "manifests_dir": tmp_path / "m",
            "generated_dir": tmp_path / "g",
            "target_size": 17,
            "validation_size": 512,
            "seed": 9,
            "audit_only": False,
        }
    ]
