from __future__ import annotations

"""Legacy multi-question batched sampler.

This script is kept for reproducing the earlier 8-rollout batched runs.
For high-depth trajectory sampling (for example 256+ rollouts per question),
use ``run_probe.py`` instead. This implementation materializes
``question_batch_size * rollouts`` sequences in one ``generate`` call, which
can become prohibitively memory-hungry at large rollout counts.
"""

import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from probe.data import prepare_questions
from probe.model import SYSTEM_PROMPT, TOKENIZER_NAME, resolve_checkpoint_revision
from probe.scoring import extract_numeric_answer, numeric_equal, _to_number
from probe.utils import (
    append_jsonl,
    empty_device_cache,
    read_jsonl,
    resolve_device,
    resolve_dtype,
    set_seed,
)


DEFAULT_SFT = "ns-0/qwen-2.5-1.5b-instruct-reasoning-sft"
DEFAULT_RL = "expx/qwen-2.5-1.5b-rlvr-ppo"


def completed_qids(path):
    return {int(row["qid"]) for row in read_jsonl(path)}


class BatchedSampler:
    def __init__(self, model_name, revision, device, dtype):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "left"

        print(f"Loading model {model_name} @ {revision}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        self.model.eval()

    def format_prompt(self, question):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @torch.inference_mode()
    def sample_batch(
        self,
        questions,
        rollouts,
        max_new_tokens,
        temperature,
        top_p,
        seed,
    ):
        prompts = []
        owners = []

        for question in questions:
            prompt = self.format_prompt(question["question"])
            prompts.extend([prompt] * rollouts)
            owners.extend([int(question["qid"])] * rollouts)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
            add_special_tokens=False,
        )
        encoded = {
            name: tensor.to(self.device)
            for name, tensor in encoded.items()
        }
        prompt_len = encoded["input_ids"].shape[1]

        output = self.model.generate(
            **encoded,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )

        grouped = {int(question["qid"]): [] for question in questions}
        for owner, sequence in zip(owners, output):
            text = self.tokenizer.decode(
                sequence[prompt_len:],
                skip_special_tokens=True,
            )
            grouped[owner].append(text)

        return grouped


def run_one(
    alias,
    model_name,
    revision,
    questions,
    out_path,
    args,
    device,
    dtype,
):
    print("\n" + "=" * 72)
    print(f"{alias.upper()}: {model_name} @ {revision}")
    print("=" * 72)

    done = completed_qids(out_path) if args.resume else set()

    if out_path.exists() and not args.resume:
        out_path.unlink()

    sampler = BatchedSampler(model_name, revision, device, dtype)
    pending = [
        question
        for question in questions
        if int(question["qid"]) not in done
    ]

    for start in range(0, len(pending), args.question_batch_size):
        batch = pending[start : start + args.question_batch_size]
        qids = [int(question["qid"]) for question in batch]
        effective_batch = len(batch) * args.rollouts

        print(
            f"[{alias}] questions={qids} | "
            f"effective batch={effective_batch}"
        )

        generations = sampler.sample_batch(
            questions=batch,
            rollouts=args.rollouts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed * 100000 + qids[0],
        )

        for question in batch:
            qid = int(question["qid"])
            gold_value = _to_number(question["gold"])
            scored_rollouts = []

            for rollout_idx, text in enumerate(generations[qid]):
                pred_value, pred_token, method = extract_numeric_answer(text)
                scored_rollouts.append(
                    {
                        "rollout": rollout_idx,
                        "pred_value": pred_value,
                        "pred_token": pred_token,
                        "extract_method": method,
                        "correct": bool(
                            numeric_equal(pred_value, gold_value)
                        ),
                        "text": text,
                    }
                )

            n_correct = sum(
                int(rollout["correct"])
                for rollout in scored_rollouts
            )

            append_jsonl(
                out_path,
                {
                    "model_alias": alias,
                    "model_name": model_name,
                    "model_revision": revision,
                    "qid": qid,
                    "question": question["question"],
                    "gold": question["gold"],
                    "gold_value": gold_value,
                    "n_correct": n_correct,
                    "n_rollouts": args.rollouts,
                    "question_batch_size": args.question_batch_size,
                    "rollouts": scored_rollouts,
                },
            )

            print(
                f"[{alias}] qid={qid:02d} "
                f"correct={n_correct}/{args.rollouts} "
                f"gold={question['gold']}"
            )

        empty_device_cache()

    del sampler
    gc.collect()
    empty_device_cache()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Legacy multi-question batched sampler. "
            "Use run_probe.py for high-depth rollout sampling."
        )
    )

    parser.add_argument("--sft-model", default=DEFAULT_SFT)
    parser.add_argument("--rl-model", default=DEFAULT_RL)
    parser.add_argument("--sft-revision", default="auto")
    parser.add_argument("--rl-revision", default="main")
    parser.add_argument("--questions", type=int, default=30)
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument("--question-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument(
        "--question-file",
        default="data/gsm8k_subset.jsonl",
    )
    parser.add_argument(
        "--result-dir",
        default="results_2048_batched",
    )
    parser.add_argument("--resume", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    effective_batch = args.rollouts * args.question_batch_size

    print(
        f"Device={device}, dtype={dtype}, K={args.rollouts}, "
        f"question_batch={args.question_batch_size}, "
        f"effective_batch={effective_batch}"
    )
    print(
        "WARNING: run_probe_batched.py is legacy. "
        "Use run_probe.py for high-depth rollout sampling."
    )

    questions = prepare_questions(
        args.question_file,
        args.questions,
        args.seed,
    )

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    sft_revision = resolve_checkpoint_revision(
        args.sft_model,
        args.sft_revision,
    )

    run_one(
        "sft",
        args.sft_model,
        sft_revision,
        questions,
        result_dir / "sft_raw.jsonl",
        args,
        device,
        dtype,
    )
    run_one(
        "rl",
        args.rl_model,
        args.rl_revision,
        questions,
        result_dir / "rl_raw.jsonl",
        args,
        device,
        dtype,
    )

    config = vars(args).copy()
    config["effective_sequence_batch"] = effective_batch
    config["sft_revision_resolved"] = sft_revision

    config_path = result_dir / "run_config.json"
    config_path.write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print("\nGeneration complete.")
    print(
        "Summarize with:\n"
        f"  python summarize_results.py --result-dir {result_dir}"
    )


if __name__ == "__main__":
    main()
