from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from controlled_run.config import load_config, validate_grpo_config
from controlled_run.data import assert_prompt_token_limit, build_gsm8k_rl_rows
from controlled_run.provenance import sha256_file, write_json
from controlled_run.rewards import (
    gsm8k_binary_reward,
    is_truncated_completion,
    resolve_terminal_token_ids,
)
from controlled_run.sample_p0 import (
    DEFAULT_CONFIG,
    P0_DATASET,
    _prompt_token_ids,
    canonical_sampling_settings,
    collect_runtime_metadata,
    select_indexed_rows,
)


DEFAULT_OUTPUT_DIR = Path("controlled_run_outputs/snapshot_eval")
SNAPSHOT_SEED_OFFSET = 75_000


def snapshot_question_seed(seed: int, dataset_index: int) -> int:
    if dataset_index < 0:
        raise ValueError("dataset_index must be non-negative")
    return int(seed) * 100_000 + int(dataset_index) + SNAPSHOT_SEED_OFFSET


def _load_json(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def resolve_snapshot_policy(canonical_run_dir: Path, snapshot_pct: int) -> dict:
    run_dir = Path(canonical_run_dir)
    percentage = int(snapshot_pct)
    if percentage not in range(5, 101, 5):
        raise ValueError("snapshot percentage must be one of 5,10,...,100")

    manifest = _load_json(run_dir / "grpo_run_manifest.json")
    if manifest.get("mode") != "canonical" or manifest.get("scientific_use") is not True:
        raise ValueError("snapshot evaluation requires a scientific canonical GRPO run")
    lineage = manifest.get("pi0_lineage_id")
    if not isinstance(lineage, str) or not lineage:
        raise ValueError("canonical manifest is missing pi0 lineage")

    schedule = _load_json(run_dir / "policy_snapshot_schedule.json")
    if schedule.get("pi0_lineage_id") != lineage:
        raise ValueError("snapshot schedule lineage does not match canonical run lineage")
    percentage_to_step = schedule.get("percentage_to_step")
    if not isinstance(percentage_to_step, dict) or str(percentage) not in percentage_to_step:
        raise ValueError(f"snapshot percentage {percentage} is missing from canonical schedule")
    expected_step = int(percentage_to_step[str(percentage)])

    policy_dir = run_dir / f"pi_{percentage:03d}"
    metadata = _load_json(policy_dir / "policy_metadata.json")
    if metadata.get("pi0_lineage_id") != lineage:
        raise ValueError("snapshot policy lineage does not match canonical run lineage")
    if int(metadata.get("target_percentage", -1)) != percentage:
        raise ValueError("snapshot metadata percentage does not match requested percentage")
    actual_step = int(metadata.get("actual_step", -1))
    if actual_step != expected_step:
        raise ValueError(
            f"snapshot actual step {actual_step} does not match canonical schedule {expected_step}"
        )

    return {
        "policy_dir": policy_dir,
        "actual_step": actual_step,
        "target_percentage": percentage,
        "pi0_lineage_id": lineage,
        "canonical_manifest": manifest,
    }


def resolve_panel(panel: str, dataset_indices: list[int] | None = None) -> dict:
    if panel == "train256":
        if dataset_indices is not None:
            raise ValueError("train256 panel does not accept explicit dataset indices")
        return {"name": "train256", "split": "train", "indices": list(range(256))}

    if panel == "heldout":
        if not dataset_indices:
            raise ValueError("heldout panel requires explicit dataset indices")
        indices = [int(index) for index in dataset_indices]
        if len(set(indices)) != len(indices):
            raise ValueError("heldout dataset indices must not contain duplicates")
        if any(index < 0 for index in indices):
            raise ValueError("heldout dataset indices must be non-negative")
        return {"name": "heldout", "split": "test", "indices": sorted(indices)}

    raise ValueError("panel must be 'train256' or 'heldout'")


def score_snapshot_completions(
    *,
    texts: list[str],
    token_ids: list[list[int]],
    finish_reasons: list[str | None],
    answer: str,
    terminal_token_ids,
) -> dict:
    texts = list(texts)
    token_ids = [[int(token_id) for token_id in ids] for ids in token_ids]
    finish_reasons = list(finish_reasons)
    if not (len(texts) == len(token_ids) == len(finish_reasons)):
        raise ValueError("text, token-id, and finish-reason counts must match")

    correctness = gsm8k_binary_reward(texts, answer)
    terminated = [
        not is_truncated_completion(ids, terminal_token_ids) for ids in token_ids
    ]
    rewards = [
        int(score > 0.0 and did_terminate)
        for score, did_terminate in zip(correctness, terminated, strict=True)
    ]

    rollouts = []
    for rollout_index, (text, ids, finish_reason, score, did_terminate, reward) in enumerate(
        zip(
            texts,
            token_ids,
            finish_reasons,
            correctness,
            terminated,
            rewards,
            strict=True,
        )
    ):
        rollouts.append(
            {
                "rollout": rollout_index,
                "correct": bool(score > 0.0),
                "terminated": bool(did_terminate),
                "canonical_reward": int(reward),
                "completion_length": len(ids),
                "finish_reason": finish_reason,
                "token_ids": ids,
                "text": text,
            }
        )

    count = len(rollouts)
    n_correct = sum(int(item["correct"]) for item in rollouts)
    n_terminated = sum(int(item["terminated"]) for item in rollouts)
    n_reward = sum(int(item["canonical_reward"]) for item in rollouts)
    return {
        "n_rollouts": count,
        "n_correct": n_correct,
        "n_terminated": n_terminated,
        "n_reward": n_reward,
        "p_correct": n_correct / count if count else 0.0,
        "p_terminated": n_terminated / count if count else 0.0,
        "p_reward": n_reward / count if count else 0.0,
        "rollouts": rollouts,
    }


def _append_jsonl(path: Path, record: dict) -> None:
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def sample_snapshot_rows(
    *,
    llm,
    tokenizer,
    indexed_rows: list[tuple[int, dict]],
    settings: dict,
    output_path: Path,
    sampling_params_cls,
    tokens_prompt_cls,
) -> None:
    terminal_token_ids = resolve_terminal_token_ids(tokenizer)
    for dataset_index, row in indexed_rows:
        encoded_prompt = tokenizer.apply_chat_template(
            row["prompt"],
            tokenize=True,
            add_generation_prompt=True,
        )
        prompt = tokens_prompt_cls(prompt_token_ids=_prompt_token_ids(encoded_prompt))
        seed = snapshot_question_seed(settings["seed"], dataset_index)
        sampling_params = sampling_params_cls(
            n=settings["num_generations"],
            temperature=settings["temperature"],
            top_p=settings["top_p"],
            top_k=settings["top_k"],
            repetition_penalty=settings["repetition_penalty"],
            max_tokens=settings["max_completion_length"],
            seed=seed,
        )
        request_outputs = llm.generate(
            [prompt],
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        if len(request_outputs) != 1:
            raise RuntimeError(f"Expected one vLLM request output, got {len(request_outputs)}")
        outputs = list(request_outputs[0].outputs)
        if len(outputs) != settings["num_generations"]:
            raise RuntimeError(
                f"Requested {settings['num_generations']} completions but received {len(outputs)}"
            )

        scored = score_snapshot_completions(
            texts=[output.text for output in outputs],
            token_ids=[list(output.token_ids) for output in outputs],
            finish_reasons=[getattr(output, "finish_reason", None) for output in outputs],
            answer=row["answer"],
            terminal_token_ids=terminal_token_ids,
        )
        record = {
            "dataset_index": dataset_index,
            "question_seed": seed,
            "question": row["prompt"][-1]["content"],
            "gold": row["answer"],
            **scored,
        }
        _append_jsonl(output_path, record)
        print(
            f"[snapshot] dataset_index={dataset_index} "
            f"reward={record['n_reward']}/{record['n_rollouts']} "
            f"correct={record['n_correct']}/{record['n_rollouts']} "
            f"terminated={record['n_terminated']}/{record['n_rollouts']}"
        )


def build_snapshot_manifest(
    *,
    canonical_run_dir: Path,
    snapshot: dict,
    panel: dict,
    dataset_sha: str,
    dataset_config: str,
    config_path: Path,
    sampling_settings: dict,
    prompt_audit: dict,
    runtime: dict,
) -> dict:
    snapshot_record = {
        "policy_dir": str(Path(snapshot["policy_dir"])),
        "actual_step": int(snapshot["actual_step"]),
        "target_percentage": int(snapshot["target_percentage"]),
    }
    return {
        "mode": "canonical_snapshot_eval",
        "canonical_run_dir": str(Path(canonical_run_dir)),
        "snapshot": snapshot_record,
        "pi0_lineage_id": str(snapshot["pi0_lineage_id"]),
        "panel": {
            "name": panel["name"],
            "split": panel["split"],
            "indices": [int(index) for index in panel["indices"]],
        },
        "dataset": {
            "name": P0_DATASET,
            "config": str(dataset_config),
            "split": panel["split"],
            "sha": str(dataset_sha),
        },
        "grpo_config_sha256": sha256_file(Path(config_path)),
        "sampling": dict(sampling_settings),
        "prompt_length_audit": dict(prompt_audit),
        "runtime": dict(runtime),
        "termination_semantics": "completion token ids end in tokenizer eos/pad id",
        "finish_reason_semantics": "descriptive_only",
    }


def run_snapshot_eval(
    *,
    canonical_run_dir: Path,
    snapshot_pct: int,
    panel_name: str,
    output_dir: Path,
    dataset_indices: list[int] | None = None,
    num_generations: int | None = None,
    config_path: Path = DEFAULT_CONFIG,
    gpu_memory_utilization: float = 0.50,
) -> dict:
    snapshot = resolve_snapshot_policy(canonical_run_dir, snapshot_pct)
    panel = resolve_panel(panel_name, dataset_indices)

    config_path = Path(config_path)
    config = load_config(config_path)
    validate_grpo_config(config)
    canonical_config = snapshot["canonical_manifest"].get("config")
    if canonical_config is not None and canonical_config != config:
        raise ValueError("current GRPO config does not match the canonical run manifest")
    settings = canonical_sampling_settings(
        config,
        num_generations_override=num_generations,
    )

    dataset_sha = snapshot["canonical_manifest"].get("gsm8k_dataset_sha")
    if not isinstance(dataset_sha, str) or not dataset_sha:
        raise ValueError("canonical run manifest is missing the pinned GSM8K dataset SHA")

    policy_dir = Path(snapshot["policy_dir"])
    tokenizer = AutoTokenizer.from_pretrained(str(policy_dir))
    raw_dataset = load_dataset(
        config["dataset_name"],
        config["dataset_config"],
        revision=dataset_sha,
        split=panel["split"],
    )
    rows = build_gsm8k_rl_rows(raw_dataset)
    indexed_rows = select_indexed_rows(rows, panel["indices"])
    selected_rows = [row for _, row in indexed_rows]
    prompt_audit = assert_prompt_token_limit(
        selected_rows,
        tokenizer,
        max_tokens=settings["max_prompt_tokens"],
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    raw_path = destination / "snapshot_raw.jsonl"
    if raw_path.exists():
        raise FileExistsError(f"Refusing to append to existing snapshot output: {raw_path}")

    manifest = build_snapshot_manifest(
        canonical_run_dir=canonical_run_dir,
        snapshot=snapshot,
        panel=panel,
        dataset_sha=dataset_sha,
        dataset_config=config["dataset_config"],
        config_path=config_path,
        sampling_settings=settings,
        prompt_audit=prompt_audit,
        runtime=collect_runtime_metadata(),
    )
    write_json(destination / "prompt_length_audit.json", prompt_audit)
    write_json(destination / "snapshot_eval_manifest.json", manifest)

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(
        model=str(policy_dir),
        tokenizer=str(policy_dir),
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=float(gpu_memory_utilization),
        max_model_len=settings["vllm_max_model_length"],
    )
    sample_snapshot_rows(
        llm=llm,
        tokenizer=tokenizer,
        indexed_rows=indexed_rows,
        settings=settings,
        output_path=raw_path,
        sampling_params_cls=SamplingParams,
        tokens_prompt_cls=TokensPrompt,
    )
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a canonical GRPO policy snapshot")
    parser.add_argument("--canonical-run-dir", type=Path, required=True)
    parser.add_argument("--snapshot-pct", type=int, required=True)
    parser.add_argument("--panel", choices=("train256", "heldout"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset-indices",
        type=str,
        default=None,
        help="Comma-separated explicit GSM8K test indices; required for heldout panel",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=None,
        help="Evaluation K; defaults to the frozen GRPO num_generations",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.50)
    args = parser.parse_args(argv)

    dataset_indices = None
    if args.dataset_indices is not None:
        dataset_indices = [
            int(token) for token in args.dataset_indices.split(",") if token.strip()
        ]

    manifest = run_snapshot_eval(
        canonical_run_dir=args.canonical_run_dir,
        snapshot_pct=args.snapshot_pct,
        panel_name=args.panel,
        output_dir=args.output_dir,
        dataset_indices=dataset_indices,
        num_generations=args.num_generations,
        config_path=args.config,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
