"""Camera-ready training launcher (approved deviation, see PROGRESS.md "Decisions" 1).

Runs the unchanged seed-42 training code (checkout of 9814757) with exactly two
runtime changes, both applied in this process only:

1. Seed: the frozen invariant `GRPO_INVARIANTS["seed"]` (42) is replaced by the
   requested seed, and the config file is the canonical YAML with only the
   `seed:` line changed. Every other invariant, manifest, lineage and data check
   still runs unchanged.
2. Logging only (camera_ready/INSTRUMENTATION.diff): `logging_steps = 1`, and a
   rank-0 callback that appends every Trainer log record to
   `<output-dir>/camera_ready_step_log.jsonl`.

Smoke mode (`--smoke-stop-step N`, scratch output only) additionally saves the
policy at step N and stops training there; the LR schedule is still built for
the full 3,736 steps.

Usage (cwd anywhere; PYTHONPATH must put the training checkout first):
    torchrun --nproc_per_node=2 camera_ready/launch_train.py \
        --objective grpo --seed 43 --pi0-dir <pi0> --output-dir <run>
    python camera_ready/launch_train.py --resolve-only --objective grpo --seed 42 \
        --pi0-dir <pi0> --output-dir <tmp>   # writes the resolved config + GRPOConfig, no training
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

TRAIN_COMMIT = "981475795538eee391c7e86aa022ee609b539770"
CANONICAL_CONFIG_REL = Path("controlled_run/configs/grpo_qwen3_0_6b.yaml")
CANONICAL_CONFIG_SHA256 = "c18b6656c50abdc139fda2c15e890dae5cd0b425caab9b1b011aff143633a71f"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def training_tree() -> Path:
    import controlled_run

    tree = Path(controlled_run.__file__).resolve().parent.parent
    head = subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(tree), "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    if head != TRAIN_COMMIT:
        raise SystemExit(f"training tree {tree} is at {head}, expected {TRAIN_COMMIT}")
    if dirty:
        raise SystemExit(f"training tree {tree} is dirty:\n{dirty}")
    return tree


def derived_config(tree: Path, seed: int, destination: Path) -> Path:
    """Canonical YAML with only the `seed:` line replaced."""
    source = tree / CANONICAL_CONFIG_REL
    if _sha256(source) != CANONICAL_CONFIG_SHA256:
        raise SystemExit("canonical config hash mismatch")
    lines = source.read_text(encoding="utf-8").splitlines(keepends=True)
    seed_lines = [k for k, line in enumerate(lines) if line.startswith("seed:")]
    if len(seed_lines) != 1 or lines[seed_lines[0]].strip() != "seed: 42":
        raise SystemExit("unexpected seed line in canonical config")
    k = seed_lines[0]
    newline = "\n" if lines[k].endswith("\n") else ""
    lines[k] = f"seed: {int(seed)}{newline}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(f".{destination.name}.rank{os.environ.get('RANK', '0')}")
    tmp.write_text("".join(lines), encoding="utf-8")
    os.replace(tmp, destination)  # atomic: every rank writes identical bytes
    return destination


def patch_seed_invariant(seed: int) -> None:
    from controlled_run import config as cfg

    if cfg.GRPO_INVARIANTS.get("seed") != 42:
        raise SystemExit("unexpected frozen seed invariant")
    cfg.GRPO_INVARIANTS["seed"] = int(seed)


def patch_logging(output_dir: Path, smoke_stop_step: int | None) -> None:
    from transformers import TrainerCallback  # noqa: F401  (import check)

    from controlled_run import train_grpo

    original_build = train_grpo.build_grpo_arguments

    def build_with_step_logging(*args, **kwargs):
        grpo_args = original_build(*args, **kwargs)
        grpo_args.logging_steps = 1
        return grpo_args

    train_grpo.build_grpo_arguments = build_with_step_logging

    base_callback = train_grpo.PolicySnapshotCallback
    log_path = Path(output_dir) / "camera_ready_step_log.jsonl"

    class LoggingPolicySnapshotCallback(base_callback):
        def on_train_begin(self, args, state, control, **kwargs):
            control = super().on_train_begin(args, state, control, **kwargs)
            if getattr(state, "is_world_process_zero", False):
                resolved = {f.name: getattr(args, f.name) for f in dataclasses.fields(args)}
                resolved["_world_size"] = args.world_size
                (Path(output_dir) / "camera_ready_trainer_args.json").write_text(
                    json.dumps(resolved, indent=1, sort_keys=True, default=str)
                )
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            if getattr(state, "is_world_process_zero", False) and logs is not None:
                import torch

                record = {"step": int(state.global_step)}
                record.update({k: v for k, v in logs.items() if isinstance(v, (int, float, str))})
                if torch.cuda.is_available():  # read-only memory counters (rank 0)
                    record["cr_max_memory_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
                    record["cr_max_memory_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
            return control

        def on_step_end(self, args, state, control, model=None, **kwargs):
            control = super().on_step_end(args, state, control, model=model, **kwargs)
            if smoke_stop_step is not None and int(state.global_step) >= smoke_stop_step:
                if getattr(state, "is_world_process_zero", False):
                    destination = self.output_dir / f"smoke_step{int(state.global_step):04d}"
                    model.save_pretrained(str(destination))
                    self.tokenizer.save_pretrained(str(destination))
                    (destination / "policy_metadata.json").write_text(
                        json.dumps(
                            {
                                "actual_step": int(state.global_step),
                                "target_percentage": None,
                                "pi0_lineage_id": self.pi0_lineage_id,
                                "smoke_only": True,
                            }
                        )
                    )
                control.should_training_stop = True
            return control

    train_grpo.PolicySnapshotCallback = LoggingPolicySnapshotCallback


def write_provenance(output_dir: Path, args, config_path: Path, tree: Path) -> None:
    cr_tree = Path(__file__).resolve().parent.parent
    cr_head = subprocess.run(["git", "-C", str(cr_tree), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    payload = {
        "objective": args.objective,
        "seed": args.seed,
        "training_commit": TRAIN_COMMIT,
        "training_tree": str(tree),
        "camera_ready_commit": cr_head,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "pi0_dir": str(args.pi0_dir),
        "output_dir": str(args.output_dir),
        "smoke_stop_step": args.smoke_stop_step,
        "runtime_patches": [
            "controlled_run.config.GRPO_INVARIANTS['seed'] = seed (only the seed equality check is relaxed)",
            "train_grpo.build_grpo_arguments(...).logging_steps = 1 (logging only)",
            "train_grpo.PolicySnapshotCallback += on_log JSONL writer and on_train_begin dump of the resolved Trainer args (logging only)",
        ]
        + (["smoke: save at stop step and stop training (scratch only)"] if args.smoke_stop_step else []),
        "env": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "WORLD_SIZE", "HF_HUB_OFFLINE", "WANDB_MODE")},
        "argv": sys.argv,
    }
    if int(os.environ.get("RANK", "0")) == 0:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        (Path(output_dir) / "camera_ready_launch.json").write_text(json.dumps(payload, indent=1))


def resolve_only(args, config_path: Path) -> None:
    """Build the GRPOConfig exactly as training would and dump it for comparison."""
    from controlled_run import train_grpo
    from controlled_run.config import load_config, validate_grpo_config

    config = load_config(config_path)
    validate_grpo_config(config)
    grpo_args = train_grpo.build_grpo_arguments(config, Path(args.output_dir) / "trainer")
    resolved = {
        f.name: getattr(grpo_args, f.name)
        for f in dataclasses.fields(grpo_args)
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "resolved_config.json").write_text(json.dumps(config, indent=1, sort_keys=True))
    (out / "resolved_grpo_args.json").write_text(json.dumps(resolved, indent=1, sort_keys=True, default=str))
    print(f"resolved config and GRPOConfig written to {out}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", choices=("grpo", "maxrl"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pi0-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke-stop-step", type=int, default=None)
    parser.add_argument("--resolve-only", action="store_true")
    args = parser.parse_args(argv)
    if args.seed not in (42, 43, 44, 45):
        raise SystemExit("seed must be 42 (verification/smoke) or 43/44/45")

    tree = training_tree()
    config_path = derived_config(tree, args.seed, Path(args.output_dir) / f"camera_ready_config_seed{args.seed}.yaml")
    patch_seed_invariant(args.seed)
    patch_logging(args.output_dir, args.smoke_stop_step)
    if args.resolve_only:
        resolve_only(args, config_path)
        return
    write_provenance(args.output_dir, args, config_path, tree)

    if args.objective == "grpo":
        from controlled_run.train_grpo import run_grpo as run
    else:
        from controlled_run.train_maxrl import run_maxrl as run
    result = run(config_path=config_path, pi0_dir=args.pi0_dir, output_dir=args.output_dir, mode="canonical")
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
