from __future__ import annotations

import argparse
import gc
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from probe.data import prepare_questions
from probe.model import Sampler, resolve_checkpoint_revision
from probe.results_upload import (
    build_remote_run_path,
    format_run_started_at,
    upload_result_dir,
)
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


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def parse_args():
    parser = argparse.ArgumentParser()

    # Models
    parser.add_argument("--sft-model", default=DEFAULT_SFT)
    parser.add_argument("--rl-model", default=DEFAULT_RL)
    parser.add_argument("--sft-revision", default="auto")
    parser.add_argument("--rl-revision", default="main")

    # Sampling
    parser.add_argument("--questions", type=int, default=30)
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument("--batch-rollouts", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)

    # Runtime
    parser.add_argument(
        "--engine",
        choices=["hf", "vllm"],
        default="hf",
        help="Inference backend. vLLM ignores --batch-rollouts and schedules requests internally.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.90,
        help="Fraction of GPU memory vLLM may reserve for its executor and KV cache.",
    )

    # I/O
    parser.add_argument(
        "--question-file",
        default="data/gsm8k_subset.jsonl",
    )
    parser.add_argument(
        "--result-dir",
        default="results",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--upload-repo",
        default=None,
        help=(
            "Optional pre-existing Hugging Face Dataset repo to receive the "
            "completed result directory. Authentication uses HF_TOKEN."
        ),
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--only-sft", action="store_true")
    mode_group.add_argument("--only-rl", action="store_true")

    return parser.parse_args()


def configure_runtime(args) -> None:
    if args.engine == "vllm":
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def completed_qids(path):
    return {int(row["qid"]) for row in read_jsonl(path)}


def build_sampler(model_name, revision, args, device, dtype):
    if args.engine == "hf":
        return Sampler(
            model_name,
            device,
            dtype,
            revision=revision,
        )

    from probe.vllm_model import VLLMSampler

    return VLLMSampler(
        model_name,
        device,
        dtype,
        revision=revision,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
    )


def maybe_empty_device_cache(engine: str) -> None:
    if engine == "hf":
        empty_device_cache()


def run_one_checkpoint(
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

    if args.resume:
        done = completed_qids(out_path)
    else:
        done = set()
        if out_path.exists():
            out_path.unlink()

    sampler = build_sampler(
        model_name=model_name,
        revision=revision,
        args=args,
        device=device,
        dtype=dtype,
    )

    for question in questions:
        qid = int(question["qid"])

        if qid in done:
            print(f"[{alias}] skip qid={qid} (resume)")
            continue

        # Deterministic random stream for each question.
        question_seed = args.seed * 100000 + qid

        generations = sampler.sample(
            question=question["question"],
            n=args.rollouts,
            batch_rollouts=args.batch_rollouts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=question_seed,
        )

        gold_value = _to_number(question["gold"])
        if gold_value is None:
            raise ValueError(
                f"Could not parse gold answer: {question['gold']}"
            )

        scored_rollouts = []

        for rollout_idx, text in enumerate(generations):
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

        result = {
            "model_alias": alias,
            "model_name": model_name,
            "model_revision": revision,
            "qid": qid,
            "question_seed": question_seed,
            "question": question["question"],
            "gold": question["gold"],
            "gold_value": gold_value,
            "n_correct": n_correct,
            "n_rollouts": args.rollouts,
            "rollouts": scored_rollouts,
        }

        append_jsonl(out_path, result)

        print(
            f"[{alias}] "
            f"qid={qid:02d} "
            f"correct={n_correct}/{args.rollouts} "
            f"gold={question['gold']}"
        )

        maybe_empty_device_cache(args.engine)

    del sampler
    gc.collect()
    maybe_empty_device_cache(args.engine)


def main():
    args = parse_args()
    run_started_at = utc_now()
    configure_runtime(args)

    # vLLM receives a per-question seed through SamplingParams. Avoid touching
    # CUDA in the parent process before its worker subprocesses are created.
    if args.engine == "hf":
        set_seed(args.seed)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)

    print(f"Engine: {args.engine}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")

    if args.engine == "hf":
        batching = f"batch={args.batch_rollouts}, "
    else:
        batching = "batch=vLLM scheduler, "

    print(
        "Sampling: "
        f"K={args.rollouts}, "
        f"{batching}"
        f"temperature={args.temperature}, "
        f"top_p={args.top_p}"
    )

    questions = prepare_questions(
        args.question_file,
        args.questions,
        args.seed,
    )

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    if args.upload_repo:
        upload_path = build_remote_run_path(result_dir, run_started_at)
    else:
        upload_path = None

    rl_revision = args.rl_revision
    sft_revision = None

    if not args.only_rl:
        sft_revision = resolve_checkpoint_revision(
            args.sft_model,
            args.sft_revision,
        )
        print(f"Resolved SFT revision: {sft_revision}")

    print(f"Resolved RL revision:  {rl_revision}")

    if not args.only_rl:
        run_one_checkpoint(
            alias="sft",
            model_name=args.sft_model,
            revision=sft_revision,
            questions=questions,
            out_path=result_dir / "sft_raw.jsonl",
            args=args,
            device=device,
            dtype=dtype,
        )

    if not args.only_sft:
        run_one_checkpoint(
            alias="rl",
            model_name=args.rl_model,
            revision=rl_revision,
            questions=questions,
            out_path=result_dir / "rl_raw.jsonl",
            args=args,
            device=device,
            dtype=dtype,
        )

    config = vars(args).copy()
    config["device_resolved"] = device
    config["dtype_resolved"] = str(dtype)
    config["sft_revision_resolved"] = sft_revision
    config["rl_revision_resolved"] = rl_revision
    config["run_started_at"] = format_run_started_at(run_started_at)
    config["upload_repo"] = args.upload_repo
    config["upload_path"] = upload_path

    config_path = result_dir / "run_config.json"
    config_path.write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    if args.upload_repo:
        try:
            upload_result_dir(
                result_dir=result_dir,
                repo_id=args.upload_repo,
                remote_path=upload_path,
            )
        except Exception as error:
            raise RuntimeError(
                f"Experiment completed and local results are in {result_dir}, "
                f"but Hugging Face backup failed: {error}"
            ) from error

    print("\nGeneration complete.")
    if args.only_sft:
        print("SFT-only run complete; paired SFT/RL summarization is not applicable.")
    elif args.only_rl:
        print("RL-only run complete; paired SFT/RL summarization is not applicable.")
    else:
        print(
            "Summarize with:\n"
            f"  python summarize_results.py --result-dir {result_dir}"
        )


if __name__ == "__main__":
    main()
