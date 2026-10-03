"""Private HF backup of camera-ready artifacts, run ON THE BOX in its own shell (Sam's instructions, PROGRESS 10-03).

Token: HF_TOKEN environment variable of this process only (passed via stdin by the launcher); never printed or written.
Usage:
  python hf_upload.py --round 1     # create repos, check privacy, upload seeds 43/44 data + bridge, then checkpoints
  python hf_upload.py --round 2     # seed 45 data + checkpoints, then re-sync all staging and bridge (deferred evals)
Each round ends with a HF-side SHA-256 verification written to /root/autodl-tmp/logs/hf_upload_round{N}_verify.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi

CR = Path("/root/autodl-tmp")
DATA_REPO = "HKReporter/rlvr-behavior-probe-camera-ready-2026-10"
CKPT_REPO = "HKReporter/rlvr-behavior-probe-camera-ready-2026-10-checkpoints"
LOG = CR / "logs" / "hf_upload.log"
RUN_FILES = ("grpo_run_manifest.json", "maxrl_run_manifest.json", "policy_snapshot_schedule.json",
             "prompt_length_audit.json", "camera_ready_launch.json", "camera_ready_trainer_args.json",
             "camera_ready_integrity.json")


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def paused() -> bool:
    return (CR / "queue" / "PAUSE_UPLOAD").exists()


def upload_dir(api: HfApi, repo: str, rtype: str, local: Path, remote: str, plan: list) -> None:
    for attempt in range(1, 6):
        while paused():
            log("paused by PAUSE_UPLOAD flag; waiting")
            time.sleep(60)
        try:
            t0 = time.time()
            size = sum(p.stat().st_size for p in local.rglob("*") if p.is_file())
            api.upload_folder(repo_id=repo, repo_type=rtype, folder_path=str(local), path_in_repo=remote,
                              commit_message=f"camera-ready backup: {remote}")
            dt = time.time() - t0
            log(f"uploaded {local} -> {repo}:{remote} ({size/1e6:.1f} MB in {dt:.0f} s, {size/1e6/max(dt,1):.2f} MB/s)")
            for p in local.rglob("*"):
                if p.is_file():
                    plan.append((repo, rtype, str(p), f"{remote}/{p.relative_to(local).as_posix()}"))
            return
        except Exception as exc:  # retry; upload_folder skips content already on the Hub
            log(f"upload attempt {attempt} failed for {local}: {type(exc).__name__}: {str(exc)[:200]}")
            time.sleep(min(300, 30 * attempt))
    raise SystemExit(f"giving up on {local}")


def upload_run_files(api: HfApi, seed: int, arm: str, plan: list) -> None:
    run = CR / "runs" / f"seed{seed}" / arm
    for name in RUN_FILES + (f"camera_ready_config_seed{seed}.yaml",):
        p = run / name
        if p.is_file():
            api.upload_file(path_or_fileobj=str(p), path_in_repo=f"seed{seed}/{arm}/{name}", repo_id=CKPT_REPO,
                            repo_type="model", commit_message=f"camera-ready backup: seed{seed}/{arm}/{name}")
            plan.append((CKPT_REPO, "model", str(p), f"seed{seed}/{arm}/{name}"))


def verify(api: HfApi, plan: list, out: Path) -> bool:
    remote = {}
    for repo, rtype in {(r, t) for r, t, _, _ in plan}:
        info = api.repo_info(repo, repo_type=rtype, files_metadata=True)
        remote[repo] = {s.rfilename: s for s in info.siblings}
    rows, bad = [], 0
    for repo, rtype, local, rpath in plan:
        s = remote[repo].get(rpath)
        lp = Path(local)
        if s is None:
            rows.append({"repo": repo, "path": rpath, "status": "MISSING"}); bad += 1; continue
        if s.lfs is not None and getattr(s.lfs, "sha256", None):
            ok = s.lfs.sha256 == sha256(lp); kind = "sha256"
        else:
            ok = s.blob_id == git_blob_sha1(lp); kind = "git-blob-sha1"
        ok = ok and (s.size == lp.stat().st_size)
        rows.append({"repo": repo, "path": rpath, "status": "OK" if ok else "MISMATCH", "check": kind, "size": s.size})
        bad += 0 if ok else 1
    out.write_text(json.dumps({"files": len(rows), "bad": bad, "rows": rows}, indent=1))
    log(f"verification: {len(rows)} files, {bad} not OK -> {out}")
    return bad == 0


def ensure_private(api: HfApi) -> None:
    for repo, rtype in ((DATA_REPO, "dataset"), (CKPT_REPO, "model")):
        api.create_repo(repo, repo_type=rtype, private=True, exist_ok=True)
        info = api.repo_info(repo, repo_type=rtype)
        log(f"repo {rtype} {repo}: private={info.private}")
        if info.private is not True:
            raise SystemExit(f"{repo} is not private; stopping")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, choices=(1, 2), required=True)
    args = ap.parse_args()
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN missing")
    api = HfApi(token=os.environ["HF_TOKEN"])
    ensure_private(api)
    plan: list = []
    if args.round == 1:
        seeds = (43, 44)
        for s in seeds:
            upload_dir(api, DATA_REPO, "dataset", CR / "results_staging" / f"seed{s}", f"seed{s}", plan)
        upload_dir(api, DATA_REPO, "dataset", CR / "bridge", "bridge_box", plan)
    else:
        seeds = (45,)
        for s in (43, 44, 45):
            upload_dir(api, DATA_REPO, "dataset", CR / "results_staging" / f"seed{s}", f"seed{s}", plan)
        upload_dir(api, DATA_REPO, "dataset", CR / "bridge", "bridge_box", plan)
    # checkpoints: step 3736 of every run first, then the other EVAL_STEPS
    for pct in (100, 65, 45, 25):
        for s in seeds:
            for arm in ("grpo", "maxrl"):
                upload_dir(api, CKPT_REPO, "model", CR / "runs" / f"seed{s}" / arm / f"pi_{pct:03d}",
                           f"seed{s}/{arm}/pi_{pct:03d}", plan)
    for s in seeds:
        for arm in ("grpo", "maxrl"):
            upload_run_files(api, s, arm, plan)
    ok = verify(api, plan, CR / "logs" / f"hf_upload_round{args.round}_verify.json")
    log(f"ROUND {args.round} {'COMPLETE' if ok else 'VERIFY FAILED'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
