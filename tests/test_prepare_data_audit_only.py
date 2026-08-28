from __future__ import annotations

import json

import controlled_run.prepare_data as prepare_module
from controlled_run.constants import BASE_MODEL, GSM8K_DATASET, SFT_DATASET


class FakeTokenizer:
    pass


def test_prepare_data_audit_only_writes_diagnostics_without_canonical_records(
    monkeypatch,
    tmp_path,
):
    def fake_resolve(repo_id, revision, repo_type):
        return {
            BASE_MODEL: "base-sha",
            SFT_DATASET: "openr1-sha",
            GSM8K_DATASET: "gsm8k-sha",
        }[repo_id]

    monkeypatch.setattr(prepare_module, "resolve_hf_revision", fake_resolve)

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, repo_id, revision):
            return FakeTokenizer()

    monkeypatch.setattr(prepare_module, "AutoTokenizer", FakeAutoTokenizer)

    def fake_load_dataset(repo_id, config_name, *, split, revision):
        if repo_id == SFT_DATASET:
            return [{"uuid": "u1", "problem": "p"}]
        return [{"question": "q", "answer": "#### 1"}]

    monkeypatch.setattr(prepare_module, "load_dataset", fake_load_dataset)

    calls = []

    def fake_build(rows, refs, tokenizer, *, target_size, seed):
        calls.append((target_size, seed))
        return [
            {
                "source_index": 0,
                "uuid": "u1",
                "generation_index": 0,
                "source": "synthetic",
                "problem_sha256": "p",
                "completion_sha256": "c",
                "formatted_token_count": 100,
            }
        ], {
            "candidate_count": 93733,
            "pre_length_filter_count": 64968,
            "removed_too_long": 58259,
            "eligible_after_filters": 6709,
            "formatted_token_percentiles": {"p50": 3000.0},
            "formatted_token_tail_fractions": {"gt_2048": 0.8967},
            "final_count": 1,
        }

    monkeypatch.setattr(prepare_module, "build_sft_manifest", fake_build)

    manifests_dir = tmp_path / "manifests"
    generated_dir = tmp_path / "generated"

    result = prepare_module.prepare_data(
        manifests_dir=manifests_dir,
        generated_dir=generated_dir,
        target_size=10_000,
        validation_size=512,
        seed=42,
        audit_only=True,
    )

    assert calls == [(1, 42)]
    audit = json.loads(
        (manifests_dir / "contamination_audit.json").read_text(encoding="utf-8")
    )
    revisions = json.loads(
        (manifests_dir / "source_revisions.json").read_text(encoding="utf-8")
    )

    assert audit["audit_only"] is True
    assert audit["requested_total_count"] == 10512
    assert audit["selected_total_count"] == 0
    assert audit["train_count"] == 0
    assert audit["validation_count"] == 0
    assert audit["eligible_after_filters"] == 6709
    assert revisions["target_size"] == 10_000
    assert revisions["validation_size"] == 512
    assert result == {"source_revisions": revisions, "audit": audit}

    assert not (manifests_dir / "sft_10k_manifest.jsonl").exists()
    assert not (manifests_dir / "sft_val_512_manifest.jsonl").exists()
    assert not (generated_dir / "sft_10k_records.jsonl").exists()
    assert not (generated_dir / "sft_val_512_records.jsonl").exists()
    assert not (manifests_dir / "data_bundle_manifest.json").exists()
