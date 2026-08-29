from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from controlled_run.checkpointing import PI0_MANIFEST_NAME, load_pi0_manifest
from controlled_run.config import load_config, validate_grpo_config
from controlled_run.data import assert_prompt_token_limit, build_gsm8k_rl_rows
from controlled_run.provenance import resolve_hf_revision, sha256_file, write_json
from controlled_run.rewards import gsm8k_binary_reward


DEFAULT_CONFIG = Path("controlled_run/configs/grpo_qwen3_0_6b.yaml")
DEFAULT_OUTPUT_DIR = Path("controlled_run_outputs/p0")
P0_DATASET = "openai/gsm8k"
P0_SPLIT = "test"


def canonical_sampling_settings(config: dict) -> dict:
    validate_grpo_config(config)
    return {
        "num_generations": config["num_generations"],
        "temperature": config["temperature"],
        "top_p": config["top_p"],
        "top_k": config["top_k"],
        "repetition_penalty": config["repetition_penalty"],
        "max_completion_length": config["max_completion_length"],
        "max_prompt_tokens": config["max_prompt_tokens"],
        "vllm_max_model_length": config["vllm_max_model_length"],
        "seed": config["seed"],
    }


def slice_shard(
    rows: list[dict],
    start_index: int = 0,
    end_index: int | None = None,
) -> list[tuple[int, dict]]:
    count = len(rows)
    if start_index < 0 or start_index >= count:
        raise ValueError(
            f"start_index must lie in [0, {count}), got {start_index}"
        )
    resolved_end = count if end_index is None else int(end_index)
    if resolved_end < 0 or resolved_end > count:
        raise ValueError(
            f"end_index must lie in [0, {count}], got {resolved_end}"
        )
    if resolved_end <= start_index:
        raise ValueError(
            f"end_index must be greater than start_index; got "
            f"{resolved_end} <= {start_index}"
        )
    return [(index, rows[index]) for index in range(start_index, resolved_end)]


def question_seed(seed: int, dataset_index: int) -> int:
    if dataset_index < 0:
        raise ValueError("dataset_index must be non-negative")
    return int(seed) * 100_000 + int(dataset_index)


def prepare_p0_rows(raw_dataset, tokenizer, max_prompt_tokens: int) -> tuple[list[dict], dict]:
    rows = build_gsm8k_rl_rows(raw_dataset)
    audit = assert_prompt_token_limit(
        rows,
        tokenizer,
        max_tokens=max_prompt_tokens,
    )
    return rows, audit


def verify_policy(policy_dir: Path) -> dict:
    directory = Path(policy_dir)
    manifest = load_pi0_manifest(directory)
    return {
        "manifest": manifest,
        "lineage_id": sha256_file(directory / PI0_MANIFEST_NAME),
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_runtime_metadata() -> dict:
    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    return {
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "gpu_name": gpu_name,
        "packages": {
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "datasets": _package_version("datasets"),
            "vllm": _package_version("vllm"),
        },
    }


def build_p0_manifest(
    *,
    policy_dir: Path,
    pi0_manifest: dict,
    pi0_lineage_id: str,
    dataset_sha: str,
    dataset_config: str,
    config_path: Path,
    sampling_settings: dict,
    prompt_audit: dict,
    start_index: int,
    end_index: int,
    record_count: int,
    runtime: dict,
) -> dict:
    return {
        "mode": "canonical_p0",
        "policy_dir": str(Path(policy_dir)),
        "pi0_manifest": pi0_manifest,
        "pi0_lineage_id": str(pi0_lineage_id),
        "dataset": {
            "name": P0_DATASET,
            "config": str(dataset_config),
            "split": P0_SPLIT,
            "sha": str(dataset_sha),
        },
        "grpo_config_sha256": sha256_file(Path(config_path)),
        "sampling": dict(sampling_settings),
        "prompt_length_audit": dict(prompt_audit),
        "shard": {
            "start_index": int(start_index),
            "end_index": int(end_index),
            "record_count": int(record_count),
        },
        "runtime": dict(runtime),
    }


def _append_jsonl(path: Path, record: dict) -> None:
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def sample_indexed_rows(
    *,
    llm,
    tokenizer,
    indexed_rows: list[tuple[int, dict]],
    settings: dict,
    output_path: Path,
    sampling_params_cls,
    tokens_prompt_cls,
) -> None:
    output_path = Path(output_path)
    for dataset_index, row in indexed_rows:
        prompt_token_ids = tokenizer.apply_chat_template(
            row["prompt"],
            tokenize=True,
            add_generation_prompt=True,
        )
        prompt = tokens_prompt_cls(prompt_token_ids=prompt_token_ids)
        seed = question_seed(settings["seed"], dataset_index)
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
            raise RuntimeError(
                f"Expected one vLLM request output, got {len(request_outputs)}"
            )
        texts = [output.text for output in request_outputs[0].outputs]
        if len(texts) != settings["num_generations"]:
            raise RuntimeError(
                f"Requested {settings['num_generations']} completions but received "
                f"{len(texts)}"
            )
        rewards = gsm8k_binary_reward(texts, row["answer"])
        if len(rewards) != len(texts):
            raise RuntimeError("Reward count does not match completion count")

        rollouts = [
            {
                "rollout": rollout_index,
                "correct": bool(reward > 0.0),
                "text": text,
            }
            for rollout_index, (text, reward) in enumerate(zip(texts, rewards))
        ]
        question = row["prompt"][-1]["content"]
        record = {
            "dataset_index": dataset_index,
            "question_seed": seed,
            "question": question,
            "gold": row["answer"],
            "n_correct": sum(int(item["correct"]) for item in rollouts),
            "n_rollouts": len(rollouts),
            "rollouts": rollouts,
        }
        _append_jsonl(output_path, record)
        print(
            f"[p0] dataset_index={dataset_index} "
            f"correct={record['n_correct']}/{record['n_rollouts']}"
        )


def run_p0(
    *,
    policy_dir: Path,
    output_dir: Path,
    config_path: Path = DEFAULT_CONFIG,
    start_index: int = 0,
    end_index: int | None = None,
    gpu_memory_utilization: float = 0.50,
) -> dict:
    policy_dir = Path(policy_dir)
    destination = Path(output_dir)
    config_path = Path(config_path)

    policy = verify_policy(policy_dir)

    config = load_config(config_path)
    settings = canonical_sampling_settings(config)
    tokenizer = AutoTokenizer.from_pretrained(str(policy_dir))

    dataset_sha = resolve_hf_revision(
        P0_DATASET,
        revision="main",
        repo_type="dataset",
    )
    raw_dataset = load_dataset(
        P0_DATASET,
        config["dataset_config"],
        revision=dataset_sha,
        split=P0_SPLIT,
    )
    rows, prompt_audit = prepare_p0_rows(
        raw_dataset,
        tokenizer,
        settings["max_prompt_tokens"],
    )
    indexed_rows = slice_shard(rows, start_index, end_index)
    resolved_end = indexed_rows[-1][0] + 1

    destination.mkdir(parents=True, exist_ok=True)
    raw_path = destination / "p0_raw.jsonl"
    if raw_path.exists():
        raise FileExistsError(
            f"Refusing to append to existing canonical p0 output: {raw_path}"
        )

    runtime = collect_runtime_metadata()
    manifest = build_p0_manifest(
        policy_dir=policy_dir,
        pi0_manifest=policy["manifest"],
        pi0_lineage_id=policy["lineage_id"],
        dataset_sha=dataset_sha,
        dataset_config=config["dataset_config"],
        config_path=config_path,
        sampling_settings=settings,
        prompt_audit=prompt_audit,
        start_index=indexed_rows[0][0],
        end_index=resolved_end,
        record_count=len(indexed_rows),
        runtime=runtime,
    )
    write_json(destination / "prompt_length_audit.json", prompt_audit)
    write_json(destination / "p0_run_manifest.json", manifest)

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
    sample_indexed_rows(
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
    parser = argparse.ArgumentParser(
        description="Sample canonical pre-RL p0 from exact frozen pi_0."
    )
    parser.add_argument("--policy-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.50)
    args = parser.parse_args(argv)

    result = run_p0(
        policy_dir=args.policy_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        start_index=args.start_index,
        end_index=args.end_index,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
