"""Measure the p0 distribution and the GRPO learning-signal budget of a policy.

This answers one question the canonical design already made decisive:

    Under the post-amendment 2048 completion cap and
    mask_truncated_completions=true, what fraction of GSM8K train prompts
    actually produce a nonzero policy gradient, and how does that fraction
    vary with the pre-RL success probability p0?

A GRPO group contributes gradient only if, AFTER truncated completions are
masked out, the surviving completions still contain both a correct and an
incorrect rollout. Groups that are fully masked, or whose survivors are all
correct / all incorrect, have zero advantage and are dead weight.

If the dead-group rate is systematically higher in low-p0 bins, the training
signal is skewed by difficulty, which is exactly the quantity the GRPO vs
MaxRL signal-allocation experiment is trying to measure.

Sampling matches the frozen GRPO recipe, not the evaluation protocol, because
the point is to predict GRPO behavior.
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
from controlled_run.rewards import gsm8k_binary_reward

CONFIG = "controlled_run/configs/grpo_qwen3_0_6b.yaml"
GSM8K_SHA = "740312add88f781978c0658806c59bc2815b9866"

THINK_START = 151667
THINK_END = 151668
EOT = 151643

# p0 bins expressed as inclusive rollout-count ranges, assuming G = 16.
# Recomputed proportionally if num_generations differs.
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

    n_gen = cfg["num_generations"]
    cap = cfg["max_completion_length"]
    max_prompt = cfg["max_prompt_tokens"]

    out_dir = Path(args.out_dir or f"{args.policy.rstrip('/')}/p0_signal_budget")
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.num_shards == 1 else f"_shard{args.shard}of{args.num_shards}"

    print("policy:            ", args.policy)
    print("num_generations:   ", n_gen)
    print("completion cap:    ", cap)
    print("temperature/top_p: ", cfg["temperature"], cfg["top_p"])
    print("shard:             ", f"{args.shard}/{args.num_shards}")
    print("out_dir:           ", out_dir)

    tok = AutoTokenizer.from_pretrained(args.policy)

    raw = load_dataset(
        cfg["dataset_name"],
        cfg["dataset_config"],
        revision=GSM8K_SHA,
        split=f"{cfg['dataset_split']}[:{args.questions}]",
    )
    rows = build_gsm8k_rl_rows(raw)

    indexed = [
        (i, row)
        for i, row in enumerate(rows)
        if i % args.num_shards == args.shard
    ]
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
    params = []
    over_length = []

    for idx, row in indexed:
        rendered = tok.apply_chat_template(
            row["prompt"],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tok(rendered, add_special_tokens=False)["input_ids"]

        if len(prompt_ids) > max_prompt:
            over_length.append((idx, len(prompt_ids)))
            continue

        prompts.append(TokensPrompt(prompt_token_ids=prompt_ids))
        params.append(
            SamplingParams(
                n=n_gen,
                temperature=cfg["temperature"],
                top_p=cfg["top_p"],
                top_k=cfg["top_k"],
                repetition_penalty=cfg["repetition_penalty"],
                max_tokens=cap,
                seed=cfg["seed"] * 100_000 + idx,
            )
        )

    if over_length:
        print(f"WARNING: {len(over_length)} prompts exceed max_prompt_tokens={max_prompt}")
        for idx, length in over_length[:5]:
            print(f"  idx={idx} prompt_tokens={length}")

    kept = [pair for pair in indexed if pair[0] not in {i for i, _ in over_length}]

    print(f"\ngenerating {len(prompts)} x {n_gen} = {len(prompts) * n_gen} rollouts ...")
    results = llm.generate(prompts, sampling_params=params, use_tqdm=True)

    records = []

    for (idx, row), result in zip(kept, results):
        rollouts = []

        for r, o in enumerate(result.outputs):
            token_ids = list(o.token_ids)
            text = o.text
            correct = bool(gsm8k_binary_reward([text], row["answer"])[0])
            truncated = o.finish_reason == "length"

            rollout = {
                "rollout": r,
                "finish_reason": o.finish_reason,
                "truncated": truncated,
                "n_tokens": len(token_ids),
                "correct": correct,
                "has_think_start": THINK_START in token_ids,
                "has_think_end": THINK_END in token_ids,
                "ends_on_eot": bool(token_ids) and token_ids[-1] == EOT,
                "text": text,
            }
            if args.store_token_ids:
                rollout["token_ids"] = token_ids
            rollouts.append(rollout)

        n_correct = sum(x["correct"] for x in rollouts)
        survivors = [x for x in rollouts if not x["truncated"]]
        n_surv = len(survivors)
        n_surv_correct = sum(x["correct"] for x in survivors)

        if n_surv == 0:
            group_state = "dead_all_masked"
        elif n_surv_correct == 0:
            group_state = "dead_all_wrong"
        elif n_surv_correct == n_surv:
            group_state = "dead_all_correct"
        else:
            group_state = "live"

        records.append({
            "dataset_index": idx,
            "question": row["prompt"][-1]["content"] if isinstance(row["prompt"], list) else None,
            "gold": row["answer"],
            "n_rollouts": n_gen,
            "n_correct": n_correct,
            "p0": n_correct / n_gen,
            "n_truncated": n_gen - n_surv,
            "clip_rate": (n_gen - n_surv) / n_gen,
            "n_survivors": n_surv,
            "n_survivor_correct": n_surv_correct,
            "group_state": group_state,
            "rollouts": rollouts,
        })

    # ---------------------------------------------------------------- summary

    all_rollouts = [x for rec in records for x in rec["rollouts"]]
    n_all = len(all_rollouts)
    n_clip = sum(x["truncated"] for x in all_rollouts)
    stopped_lengths = [x["n_tokens"] for x in all_rollouts if not x["truncated"]]

    print("\n=== AGGREGATE ===")
    print(f"prompts:            {len(records)}")
    print(f"rollouts:           {n_all}")
    print(f"clipped_ratio:      {n_clip}/{n_all} = {pct(n_clip, n_all):.2f}%")
    print(f"sample accuracy:    {pct(sum(x['correct'] for x in all_rollouts), n_all):.2f}%")
    if stopped_lengths:
        q = np.percentile(stopped_lengths, [50, 75, 90, 99])
        print(f"stopped length p50/p75/p90/p99: {q[0]:.0f} / {q[1]:.0f} / {q[2]:.0f} / {q[3]:.0f}")

    states = Counter(rec["group_state"] for rec in records)
    print("\n=== GROUP STATE (mask_truncated_completions=true) ===")
    for state in ["live", "dead_all_masked", "dead_all_wrong", "dead_all_correct"]:
        n = states.get(state, 0)
        print(f"  {state:<18} {n:>4}/{len(records)} = {pct(n, len(records)):.2f}%")

    print("\n=== p0 HISTOGRAM ===")
    hist = Counter(rec["n_correct"] for rec in records)
    for k in range(n_gen + 1):
        n = hist.get(k, 0)
        bar = "#" * int(40 * n / max(1, max(hist.values())))
        print(f"  {k:>2}/{n_gen}  {n:>4}  {bar}")

    interior = sum(n for k, n in hist.items() if 0 < k < n_gen)
    print(f"\n  interior (neither 0 nor {n_gen}): {interior}/{len(records)} = "
          f"{pct(interior, len(records)):.2f}%")

    print("\n=== CLIP RATE x p0  (the decisive table) ===")
    print(f"  {'bin':<12} {'n':>4} {'mean p0':>9} {'clip%':>8} {'live%':>8} {'allmask%':>9}")
    bins = {}
    for rec in records:
        bins.setdefault(assign_bin(rec["p0"]), []).append(rec)

    rows_out = []
    for label, _, _ in BIN_FRACTIONS:
        group = bins.get(label, [])
        if not group:
            continue
        mean_p0 = float(np.mean([r["p0"] for r in group]))
        mean_clip = float(np.mean([r["clip_rate"] for r in group]))
        live = sum(r["group_state"] == "live" for r in group)
        allmask = sum(r["group_state"] == "dead_all_masked" for r in group)
        print(f"  {label:<12} {len(group):>4} {mean_p0:>9.3f} "
              f"{100*mean_clip:>7.2f}% {pct(live, len(group)):>7.2f}% "
              f"{pct(allmask, len(group)):>8.2f}%")
        rows_out.append({
            "bin": label,
            "n_prompts": len(group),
            "mean_p0": mean_p0,
            "mean_clip_rate": mean_clip,
            "live_rate": live / len(group),
            "all_masked_rate": allmask / len(group),
        })

    think_start = sum(x["has_think_start"] for x in all_rollouts)
    think_end = sum(x["has_think_end"] for x in all_rollouts)
    print("\n=== THINK TAGS (token-id based) ===")
    print(f"  has <think>:  {think_start}/{n_all} = {pct(think_start, n_all):.2f}%")
    print(f"  has </think>: {think_end}/{n_all} = {pct(think_end, n_all):.2f}%")

    summary = {
        "policy": args.policy,
        "config": args.config,
        "gsm8k_revision": GSM8K_SHA,
        "num_generations": n_gen,
        "max_completion_length": cap,
        "max_prompt_tokens": max_prompt,
        "sampling": {
            "temperature": cfg["temperature"],
            "top_p": cfg["top_p"],
            "top_k": cfg["top_k"],
            "repetition_penalty": cfg["repetition_penalty"],
            "seed": cfg["seed"],
        },
        "shard": args.shard,
        "num_shards": args.num_shards,
        "n_prompts": len(records),
        "n_rollouts": n_all,
        "clipped_ratio": n_clip / n_all if n_all else None,
        "sample_accuracy": sum(x["correct"] for x in all_rollouts) / n_all if n_all else None,
        "group_states": dict(states),
        "p0_histogram": {str(k): hist.get(k, 0) for k in range(n_gen + 1)},
        "interior_fraction": interior / len(records) if records else None,
        "clip_by_p0_bin": rows_out,
        "think_start_rate": think_start / n_all if n_all else None,
        "think_end_rate": think_end / n_all if n_all else None,
        "over_length_prompts": over_length,
    }

    summary_path = out_dir / f"summary{suffix}.json"
    raw_path = out_dir / f"rollouts{suffix}.jsonl"

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with raw_path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec) + "\n")

    print("\nsaved:", summary_path)
    print("saved:", raw_path)


if __name__ == "__main__":
    main()