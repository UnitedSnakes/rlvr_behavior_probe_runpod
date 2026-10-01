"""Camera-ready evaluation wrapper (approved deviation, PROGRESS.md "Decisions" 1).

Runs the unchanged canonical sequential evaluator (`controlled_run.eval_snapshot`
from checkout 1c26b1f) on a run trained with seed S, with runtime changes only:

1. `GRPO_INVARIANTS["seed"] = S`, and the evaluator is given the run's own
   derived config (canonical YAML with only `seed:` changed), so its existing
   manifest-equality check still runs. Per-question seed = S*100000 + i + 75000.
2. Extra endpoint batch b ∈ {1,2,3}: `SNAPSHOT_SEED_OFFSET = 75000 + b*1_000_000`,
   i.e. per-question seed = S*100000 + i + 75000 + b*1,000,000.
3. Smoke only (`--smoke-policy-dir`, `--smoke-questions n`): evaluate a smoke
   snapshot on the first n panel questions.

    python camera_ready/eval_run.py --run-dir <run> --seed 43 --pct 100 --output-dir <out> [--extra-batch 1]
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

EVAL_COMMIT = "1c26b1f0f3c5f6ea1187fd00318587388a891272"
BASE_OFFSET = 75_000
EXTRA_OFFSET = 1_000_000


def eval_tree() -> Path:
    import controlled_run

    tree = Path(controlled_run.__file__).resolve().parent.parent
    head = subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(tree), "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    if head != EVAL_COMMIT or dirty:
        raise SystemExit(f"evaluation tree {tree} at {head} (dirty={bool(dirty)}), expected clean {EVAL_COMMIT}")
    return tree


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pct", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--extra-batch", type=int, choices=(1, 2, 3), default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.50)
    parser.add_argument("--smoke-policy-dir", type=Path, default=None)
    parser.add_argument("--smoke-questions", type=int, default=None)
    args = parser.parse_args(argv)
    if args.seed not in (42, 43, 44, 45):
        raise SystemExit("seed must be 42..45")

    tree = eval_tree()
    from controlled_run import config as cfg
    from controlled_run import eval_snapshot

    config_path = args.run_dir / f"camera_ready_config_seed{args.seed}.yaml"
    if args.seed == 42 and not config_path.exists():
        config_path = tree / "controlled_run/configs/grpo_qwen3_0_6b.yaml"
    if not config_path.is_file():
        raise SystemExit(f"missing run config {config_path}")

    if cfg.GRPO_INVARIANTS.get("seed") != 42:
        raise SystemExit("unexpected frozen seed invariant")
    cfg.GRPO_INVARIANTS["seed"] = args.seed
    offset = BASE_OFFSET + (args.extra_batch or 0) * EXTRA_OFFSET
    if eval_snapshot.SNAPSHOT_SEED_OFFSET != BASE_OFFSET:
        raise SystemExit("unexpected evaluator seed offset")
    eval_snapshot.SNAPSHOT_SEED_OFFSET = offset

    patches = [
        "controlled_run.config.GRPO_INVARIANTS['seed'] = seed",
        f"controlled_run.eval_snapshot.SNAPSHOT_SEED_OFFSET = {offset}",
    ]
    if args.smoke_policy_dir is not None:
        def smoke_resolve(run_dir, pct):
            manifest_path, manifest = eval_snapshot._resolve_canonical_run_manifest(run_dir)
            meta = json.loads((args.smoke_policy_dir / "policy_metadata.json").read_text())
            if meta.get("pi0_lineage_id") != manifest.get("pi0_lineage_id"):
                raise SystemExit("smoke snapshot lineage mismatch")
            return {
                "policy_dir": args.smoke_policy_dir,
                "actual_step": int(meta["actual_step"]),
                "target_percentage": 5,
                "pi0_lineage_id": manifest["pi0_lineage_id"],
                "canonical_manifest": manifest,
                "canonical_manifest_filename": manifest_path.name,
                "objective_family": "MaxRL" if manifest_path.name == "maxrl_run_manifest.json" else "GRPO",
            }

        eval_snapshot.resolve_snapshot_policy = smoke_resolve
        patches.append("smoke: policy dir overridden")
    if args.smoke_questions is not None:
        n = int(args.smoke_questions)
        eval_snapshot.resolve_panel = lambda panel, dataset_indices=None: {
            "name": "train256", "split": "train", "indices": list(range(n))
        }
        patches.append(f"smoke: first {n} questions only")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cr_tree = Path(__file__).resolve().parent.parent
    cr_head = subprocess.run(["git", "-C", str(cr_tree), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (args.output_dir / "camera_ready_eval_provenance.json").write_text(
        json.dumps(
            {
                "run_dir": str(args.run_dir),
                "seed": args.seed,
                "pct": args.pct,
                "extra_batch": args.extra_batch,
                "question_seed_rule": f"{args.seed}*100000 + dataset_index + {offset}",
                "evaluation_commit": EVAL_COMMIT,
                "evaluation_tree": str(tree),
                "camera_ready_commit": cr_head,
                "config_path": str(config_path),
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "runtime_patches": patches,
            },
            indent=1,
        )
    )
    manifest = eval_snapshot.run_snapshot_eval(
        canonical_run_dir=args.run_dir,
        snapshot_pct=args.pct,
        panel_name="train256",
        output_dir=args.output_dir,
        config_path=config_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    print(json.dumps({"snapshot": manifest["snapshot"], "sampling": manifest["sampling"]}, indent=1))


if __name__ == "__main__":
    main()
