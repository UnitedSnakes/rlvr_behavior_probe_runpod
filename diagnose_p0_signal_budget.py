"""Collect the canonical GSM8K-train K=32 p0 rebaseline.

Each prompt is sampled in two independent K=16 vLLM calls.  p0 is defined by
canonical terminated-and-correct reward; correctness and termination are also
recorded separately.  K=32 improves probability estimation only: canonical
GRPO group-level quantities remain G=16.
"""

import os

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from controlled_run.config import load_config
from controlled_run.data import build_gsm8k_rl_rows
from controlled_run.rewards import (
    gsm8k_binary_reward,
    is_truncated_completion,
    resolve_terminal_token_ids,
)

CONFIG = "controlled_run/configs/grpo_qwen3_0_6b.yaml"
GSM8K_SHA = "740312add88f781978c0658806c59bc2815b9866"
HALF_SIZE = 16
SECOND_HALF_SEED_OFFSET = 50_000

THINK_START = 151667
THINK_END = 151668
EOT = 151643

BIN_FRACTIONS = [
    ("0", 0.0, 0.0),
    ("(0, 1/4]", 0.0, 0.25),
    ("(1/4, 1/2]", 0.25, 0.50),
    ("(1/2, 3/4]", 0.50, 0.75),
    ("(3/4, 1)", 0.75, 1.0),
    ("1", 1.0, 1.0),
]


def assign_bin(p: float) -> str:
    if p <= 0.0:
        return "0"
    if p >= 1.0:
        return "1"
    for label, low, high in BIN_FRACTIONS[1:-1]:
        if low < p <= high:
            return label
    return "(3/4, 1)"


def pct(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else float("nan")


def half_seed(base_seed: int, dataset_index: int, half: str) -> int:
    if half == "A":
        offset = 0
    elif half == "B":
        offset = SECOND_HALF_SEED_OFFSET
    else:
        raise ValueError("half must be 'A' or 'B'")
    return int(base_seed) * 100_000 + int(dataset_index) + offset


def model_live_probability(p: float, *, group_size: int) -> float:
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    p = float(p)
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must lie in [0, 1]")
    return 1.0 - p**group_size - (1.0 - p) ** group_size


def _sampling_params(settings, dataset_indices, half, *, half_size, sampling_params_cls):
    return [
        sampling_params_cls(
            n=half_size,
            temperature=settings["temperature"],
            top_p=settings["top_p"],
            top_k=settings["top_k"],
            repetition_penalty=settings["repetition_penalty"],
            max_tokens=settings["max_completion_length"],
            seed=half_seed(settings["seed"], idx, half),
        )
        for idx in dataset_indices
    ]


def generate_independent_halves(
    *,
    llm,
    prompts,
    dataset_indices,
    settings,
    sampling_params_cls=SamplingParams,
    half_size: int = HALF_SIZE,
    use_tqdm: bool = True,
):
    """Generate A and B in separate calls so the cross-fit halves are independent."""
    if len(prompts) != len(dataset_indices):
        raise ValueError("prompts and dataset_indices must have matching lengths")
    if half_size <= 0:
        raise ValueError("half_size must be positive")

    params_a = _sampling_params(
        settings, dataset_indices, "A", half_size=half_size, sampling_params_cls=sampling_params_cls
    )
    params_b = _sampling_params(
        settings, dataset_indices, "B", half_size=half_size, sampling_params_cls=sampling_params_cls
    )
    results_a = llm.generate(prompts, sampling_params=params_a, use_tqdm=use_tqdm)
    results_b = llm.generate(prompts, sampling_params=params_b, use_tqdm=use_tqdm)
    return results_a, results_b


def score_half_outputs(
    outputs,
    *,
    answer,
    terminal_token_ids,
    half_label: str,
    store_token_ids: bool,
):
    outputs = list(outputs)
    correctness = gsm8k_binary_reward([output.text for output in outputs], answer)
    rollouts = []
    for rollout_index, (output, correct_score) in enumerate(zip(outputs, correctness, strict=True)):
        token_ids = list(output.token_ids)
        correct = bool(correct_score)
        terminated = not is_truncated_completion(token_ids, terminal_token_ids)
        canonical_reward = int(correct and terminated)
        rollout = {
            "half": half_label,
            "rollout": rollout_index,
            "finish_reason": output.finish_reason,
            "n_tokens": len(token_ids),
            "correct": correct,
            "terminated": terminated,
            "canonical_reward": canonical_reward,
            "capped": not terminated,
            "has_think_start": THINK_START in token_ids,
            "has_think_end": THINK_END in token_ids,
            "ends_on_eot": bool(token_ids) and token_ids[-1] == EOT,
            "text": output.text,
        }
        if store_token_ids:
            rollout["token_ids"] = token_ids
        rollouts.append(rollout)
    return rollouts


def _empirical_live(rollouts, group_size: int) -> bool:
    if len(rollouts) != group_size:
        raise ValueError(
            f"canonical empirical group must contain G={group_size} rollouts, got {len(rollouts)}"
        )
    successes = sum(int(row["canonical_reward"]) for row in rollouts)
    return 0 < successes < group_size


def combine_halves(
    *,
    dataset_index: int,
    question,
    gold,
    half_a,
    half_b,
    group_size: int,
):
    half_a = list(half_a)
    half_b = list(half_b)
    if len(half_a) != group_size or len(half_b) != group_size:
        raise ValueError("each cross-fit half must contain exactly one canonical G-sized group")

    all_rollouts = half_a + half_b
    n = len(all_rollouts)
    successes_a = sum(int(r["canonical_reward"]) for r in half_a)
    successes_b = sum(int(r["canonical_reward"]) for r in half_b)
    p0_a = successes_a / group_size
    p0_b = successes_b / group_size
    p0 = (successes_a + successes_b) / n

    return {
        "dataset_index": int(dataset_index),
        "question": question,
        "gold": gold,
        "group_size": group_size,
        "half_size": group_size,
        "n_rollouts": n,
        "canonical_successes_A": successes_a,
        "canonical_successes_B": successes_b,
        "p0_A": p0_a,
        "p0_B": p0_b,
        "p0": p0,
        "correctness_p0": sum(bool(r["correct"]) for r in all_rollouts) / n,
        "termination_rate": sum(bool(r["terminated"]) for r in all_rollouts) / n,
        "cap_rate": sum(bool(r["capped"]) for r in all_rollouts) / n,
        "empirical_live_A": _empirical_live(half_a, group_size),
        "empirical_live_B": _empirical_live(half_b, group_size),
        "model_live_probability_G16": model_live_probability(p0, group_size=group_size),
        "rollouts_A": half_a,
        "rollouts_B": half_b,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--questions", type=int, default=256)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--store-token-ids", action="store_true")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    group_size = int(cfg["num_generations"])
    if group_size != HALF_SIZE:
        raise ValueError(
            f"canonical K=32 rebaseline requires GRPO G={HALF_SIZE}; config has {group_size}"
        )
    if cfg.get("reward") != "binary_terminated_final_answer_correctness":
        raise ValueError("rebaseline requires the canonical terminated-and-correct reward")
    if bool(cfg.get("mask_truncated_completions")):
        raise ValueError("rebaseline requires canonical mask_truncated_completions=false")

    cap = int(cfg["max_completion_length"])
    max_prompt = int(cfg["max_prompt_tokens"])
    settings = {
        "temperature": cfg["temperature"],
        "top_p": cfg["top_p"],
        "top_k": cfg["top_k"],
        "repetition_penalty": cfg["repetition_penalty"],
        "max_completion_length": cap,
        "seed": cfg["seed"],
    }

    out_dir = Path(args.out_dir or f"{args.policy.rstrip('/')}/p0_train_k32_rebaseline")
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.num_shards == 1 else f"_shard{args.shard}of{args.num_shards}"

    print("policy:             ", args.policy)
    print("GRPO group size G:  ", group_size)
    print("bank K:             ", 2 * group_size)
    print("completion cap:     ", cap)
    print("temperature/top_p:  ", cfg["temperature"], cfg["top_p"])
    print("reward:              ", cfg["reward"])
    print("shard:               ", f"{args.shard}/{args.num_shards}")
    print("out_dir:             ", out_dir)

    tok = AutoTokenizer.from_pretrained(args.policy)
    terminal_token_ids = resolve_terminal_token_ids(tok)
    print("terminal token ids:  ", terminal_token_ids)

    raw = load_dataset(
        cfg["dataset_name"],
        cfg["dataset_config"],
        revision=GSM8K_SHA,
        split=f"{cfg['dataset_split']}[:{args.questions}]",
    )
    rows = build_gsm8k_rl_rows(raw)
    indexed = [(i, row) for i, row in enumerate(rows) if i % args.num_shards == args.shard]
    print(f"prompts in this shard: {len(indexed)} of {len(rows)}")

    llm = LLM(
        model=args.policy,
        tokenizer=args.policy,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=cfg["vllm_max_model_length"],
        seed=cfg["seed"],
    )

    prompts = []
    kept = []
    over_length = []
    for idx, row in indexed:
        rendered = tok.apply_chat_template(
            row["prompt"], tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tok(rendered, add_special_tokens=False)["input_ids"]
        if len(prompt_ids) > max_prompt:
            over_length.append((idx, len(prompt_ids)))
            continue
        prompts.append(TokensPrompt(prompt_token_ids=prompt_ids))
        kept.append((idx, row))

    if over_length:
        print(f"WARNING: {len(over_length)} prompts exceed max_prompt_tokens={max_prompt}")
        for idx, length in over_length[:5]:
            print(f"  idx={idx} prompt_tokens={length}")

    dataset_indices = [idx for idx, _ in kept]
    print(
        f"\ngenerating two independent halves: "
        f"2 x {len(prompts)} x {group_size} = {2 * len(prompts) * group_size} rollouts ..."
    )
    results_a, results_b = generate_independent_halves(
        llm=llm,
        prompts=prompts,
        dataset_indices=dataset_indices,
        settings=settings,
        half_size=group_size,
        use_tqdm=True,
    )

    records = []
    for (idx, row), result_a, result_b in zip(kept, results_a, results_b, strict=True):
        half_a = score_half_outputs(
            result_a.outputs,
            answer=row["answer"],
            terminal_token_ids=terminal_token_ids,
            half_label="A",
            store_token_ids=args.store_token_ids,
        )
        half_b = score_half_outputs(
            result_b.outputs,
            answer=row["answer"],
            terminal_token_ids=terminal_token_ids,
            half_label="B",
            store_token_ids=args.store_token_ids,
        )
        question = row["prompt"][-1]["content"] if isinstance(row["prompt"], list) else None
        records.append(
            combine_halves(
                dataset_index=idx,
                question=question,
                gold=row["answer"],
                half_a=half_a,
                half_b=half_b,
                group_size=group_size,
            )
        )

    all_rollouts = [
        rollout
        for rec in records
        for key in ("rollouts_A", "rollouts_B")
        for rollout in rec[key]
    ]
    n_all = len(all_rollouts)
    n_success = sum(r["canonical_reward"] for r in all_rollouts)
    n_correct = sum(r["correct"] for r in all_rollouts)
    n_terminated = sum(r["terminated"] for r in all_rollouts)
    n_capped = sum(r["capped"] for r in all_rollouts)

    print("\n=== AGGREGATE K=32 CANONICAL BANK ===")
    print(f"prompts:             {len(records)}")
    print(f"rollouts:            {n_all}")
    print(f"canonical success:   {pct(n_success, n_all):.2f}%")
    print(f"correctness-only:    {pct(n_correct, n_all):.2f}%")
    print(f"termination:         {pct(n_terminated, n_all):.2f}%")
    print(f"cap/nontermination:  {pct(n_capped, n_all):.2f}%")

    hist = Counter(round(rec["p0"] * (2 * group_size)) for rec in records)
    print("\n=== K=32 CANONICAL p0 HISTOGRAM ===")
    for k in range(2 * group_size + 1):
        count = hist.get(k, 0)
        if count:
            print(f"  {k:>2}/{2 * group_size}  {count:>4}")

    bins = {}
    for rec in records:
        bins.setdefault(assign_bin(rec["p0"]), []).append(rec)

    print("\n=== CANONICAL p0 BIN REBASELINE (G=16 live quantities) ===")
    print(
        f"  {'bin':<12} {'n':>4} {'mean p0':>9} {'cap%':>8} {'term%':>8} "
        f"{'correct%':>9} {'model live%':>11} {'live A%':>9} {'live B%':>9}"
    )
    rows_out = []
    for label, _, _ in BIN_FRACTIONS:
        group = bins.get(label, [])
        if not group:
            continue
        row_out = {
            "bin": label,
            "n_prompts": len(group),
            "mean_p0": float(np.mean([r["p0"] for r in group])),
            "mean_cap_rate": float(np.mean([r["cap_rate"] for r in group])),
            "mean_termination_rate": float(np.mean([r["termination_rate"] for r in group])),
            "mean_correctness_p0": float(np.mean([r["correctness_p0"] for r in group])),
            "mean_model_live_probability_G16": float(
                np.mean([r["model_live_probability_G16"] for r in group])
            ),
            "empirical_live_A_rate": float(np.mean([r["empirical_live_A"] for r in group])),
            "empirical_live_B_rate": float(np.mean([r["empirical_live_B"] for r in group])),
        }
        rows_out.append(row_out)
        print(
            f"  {label:<12} {len(group):>4} {row_out['mean_p0']:>9.3f} "
            f"{100*row_out['mean_cap_rate']:>7.2f}% "
            f"{100*row_out['mean_termination_rate']:>7.2f}% "
            f"{100*row_out['mean_correctness_p0']:>8.2f}% "
            f"{100*row_out['mean_model_live_probability_G16']:>10.2f}% "
            f"{100*row_out['empirical_live_A_rate']:>8.2f}% "
            f"{100*row_out['empirical_live_B_rate']:>8.2f}%"
        )

    think_start = sum(x["has_think_start"] for x in all_rollouts)
    think_end = sum(x["has_think_end"] for x in all_rollouts)
    summary = {
        "policy": args.policy,
        "config": args.config,
        "gsm8k_revision": GSM8K_SHA,
        "dataset_split": cfg["dataset_split"],
        "reward": cfg["reward"],
        "group_size": group_size,
        "half_size": group_size,
        "bank_k": 2 * group_size,
        "crossfit_halves": {
            "independent_generate_calls": True,
            "half_A_seed_formula": "seed*100000+dataset_index",
            "half_B_seed_formula": "seed*100000+dataset_index+50000",
        },
        "max_completion_length": cap,
        "max_prompt_tokens": max_prompt,
        "terminal_token_ids": list(terminal_token_ids),
        "sampling": settings,
        "shard": args.shard,
        "num_shards": args.num_shards,
        "n_prompts": len(records),
        "n_rollouts": n_all,
        "canonical_success_rate": n_success / n_all if n_all else None,
        "correctness_rate": n_correct / n_all if n_all else None,
        "termination_rate": n_terminated / n_all if n_all else None,
        "cap_rate": n_capped / n_all if n_all else None,
        "p0_histogram_K32": {str(k): hist.get(k, 0) for k in range(2 * group_size + 1)},
        "by_p0_bin": rows_out,
        "think_start_rate": think_start / n_all if n_all else None,
        "think_end_rate": think_end / n_all if n_all else None,
        "over_length_prompts": over_length,
    }

    summary_path = out_dir / f"summary{suffix}.json"
    raw_path = out_dir / f"rollouts{suffix}.jsonl"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with raw_path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, allow_nan=False) + "\n")

    print("\nsaved:", summary_path)
    print("saved:", raw_path)


if __name__ == "__main__":
    main()
