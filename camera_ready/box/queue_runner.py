"""Server-side camera-ready queue (E3). Runs on the box under `screen`; does not depend on the laptop.

Per seed (43, 44, 45 in order):
  1. time rule (PREREG_RUNS.md §5; seed 43 unconditional), 2. both arms concurrently on GPU pairs
  (alternating), 3. integrity checks + structural audit, 4. both arms' K=16 endpoint evaluations
  (plus seed-42 bridge snapshot evaluations on the two idle GPUs), 5. staging, 6. snapshot retention.
After all training: the deferred K=16 evaluations (934/1681/2428), remaining bridge evaluations,
then extra endpoint batches while the time budget allows.

State: /root/autodl-tmp/queue/state.json ; log: /root/autodl-tmp/queue/queue.log
Stop gracefully: touch /root/autodl-tmp/queue/STOP (checked between jobs).
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

CR = Path("/root/autodl-tmp")
REPOS = CR / "repos"
TRAIN_TREE = REPOS / "train_9814757"
EVAL_TREE = REPOS / "eval_1c26b1f"
CR_TREE = REPOS / "cr"
PY = CR / "envs/rlvr/bin/python"
TORCHRUN = CR / "envs/rlvr/bin/torchrun"
PI0 = CR / "a40/pi0/pi_0"
RUNS = CR / "runs"
STAGING = CR / "results_staging"
QDIR = CR / "queue"
STATE = QDIR / "state.json"
LOG = QDIR / "queue.log"

UTC = dt.timezone.utc
TRAIN_CUTOFF = dt.datetime(2026, 10, 3, 16, 0, tzinfo=UTC)
RESULTS_DUE = dt.datetime(2026, 10, 4, 0, 0, tzinfo=UTC)
ANALYSIS_H, FACTCHECK_H, BUFFER_H = 1.5, 3.0, 2.0
SEEDS = (43, 44, 45)
STEPS = 3736
EVAL_PCTS_DEFERRED = (25, 45, 65)
GPU_PAIRS = ("0,1", "2,3")
BRIDGE_ITEMS = [(arm, pct) for pct in (100, 65, 45, 25) for arm in ("grpo", "maxrl")]

sys.path.insert(0, str(CR_TREE))


def now() -> dt.datetime:
    return dt.datetime.now(tz=UTC)


def log(msg: str) -> None:
    line = f"[{now().strftime('%Y-%m-%dT%H:%M:%SZ')}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def load_state() -> dict:
    if STATE.is_file():
        return json.loads(STATE.read_text())
    return {"pairs": {}, "measured": {}, "events": [], "bridge_done": [], "extra_done": [], "deferred_done": []}


STATE_LOCK = threading.RLock()
_last_smi = [0.0]


def save_state(state: dict) -> None:
    with STATE_LOCK:
        tmp = STATE.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1, default=str))
        os.replace(tmp, STATE)
    if time.time() - _last_smi[0] > 1800:
        _last_smi[0] = time.time()
        try:
            _smi_snapshot()
        except Exception as exc:  # diagnostics must never stop the queue
            log(f"nvidia-smi snapshot failed: {exc}")


def _smi_snapshot() -> None:
    if True:
        smi = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True,
        ).stdout
        gpus = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader"],
            capture_output=True, text=True,
        ).stdout
        with (QDIR / "nvidia_smi.log").open("a") as f:
            f.write(f"--- {now().isoformat()}\n{gpus}{smi}")


def base_env() -> dict:
    env = dict(os.environ)
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "HF_TOKEN"):
        env.pop(k, None)
    env.update(
        {
            "HF_HOME": str(CR / "hf-cache"),
            "HUGGINGFACE_HUB_CACHE": str(CR / "hf-cache/hub"),
            "TORCH_HOME": str(CR / "torch-cache"),
            "XDG_CACHE_HOME": str(CR / "cache"),
            "TMPDIR": str(CR / "tmp"),
            "TRITON_CACHE_DIR": str(CR / "cache/triton"),
            "FLASHINFER_WORKSPACE_BASE": str(CR / "cache/flashinfer"),
            "VLLM_CACHE_ROOT": str(CR / "cache/vllm"),
            "CUDA_CACHE_PATH": str(CR / "cache/nv"),
            "WANDB_MODE": "offline",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "PYTHONUNBUFFERED": "1",
            "PATH": f"{CR / 'envs/rlvr/bin'}:{env.get('PATH', '')}",
        }
    )
    return env


def train_env(gpus: str) -> dict:
    env = base_env()
    # The unchanged trainer resolves the GSM8K `main` revision online at launch; the academic proxy
    # stays off, so the lookup goes to the HF mirror. The resolved SHA is verified after launch.
    env.update({"CUDA_VISIBLE_DEVICES": gpus, "HF_ENDPOINT": "https://hf-mirror.com", "PYTHONPATH": str(TRAIN_TREE),
                # resource-only: avoids allocator fragmentation OOM on 32 GB cards (no numerical effect)
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    return env


def eval_env(gpu: str) -> dict:
    env = base_env()
    env.update({"CUDA_VISIBLE_DEVICES": gpu, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "PYTHONPATH": str(EVAL_TREE)})
    return env


# ----------------------------------------------------------------------------- jobs

def start_training(seed: int, arm: str, gpus: str, port: int, attempt: int) -> subprocess.Popen:
    out = RUNS / f"seed{seed}" / arm
    if out.exists():
        shutil.rmtree(out)  # never resume: a restart begins from scratch
    out.mkdir(parents=True)
    logf = open(RUNS / f"seed{seed}" / f"{arm}_train_attempt{attempt}.log", "w")
    cmd = [
        str(TORCHRUN), "--nproc_per_node=2", f"--master_port={port}",
        str(CR_TREE / "camera_ready/launch_train.py"),
        "--objective", arm, "--seed", str(seed), "--pi0-dir", str(PI0), "--output-dir", str(out),
    ]
    log(f"launch seed {seed} {arm} on GPUs {gpus} (attempt {attempt}): {' '.join(cmd)}")
    return subprocess.Popen(cmd, cwd=str(TRAIN_TREE), env=train_env(gpus), stdout=logf, stderr=subprocess.STDOUT)


def start_eval(kind: str, gpu: str, **kw) -> tuple[subprocess.Popen, Path]:
    if kind == "protocol" or kind == "extra":
        seed, arm, pct = kw["seed"], kw["arm"], kw["pct"]
        run = RUNS / f"seed{seed}" / arm
        if kind == "protocol":
            out = run / "eval" / f"pi_{pct:03d}"
            extra = []
        else:
            out = run / "eval_extra" / f"pi_100_b{kw['batch']}"
            extra = ["--extra-batch", str(kw["batch"])]
        cmd = [str(PY), str(CR_TREE / "camera_ready/eval_run.py"), "--run-dir", str(run), "--seed", str(seed),
               "--pct", str(pct), "--output-dir", str(out)] + extra
    elif kind == "bridge":
        arm, pct = kw["arm"], kw["pct"]
        out = CR / "bridge" / f"{arm}_seed42" / f"pi_{pct:03d}"
        cmd = [str(PY), "-m", "controlled_run.eval_snapshot", "--canonical-run-dir", str(CR / "a40" / f"{arm}_seed42"),
               "--snapshot-pct", str(pct), "--panel", "train256", "--output-dir", str(out)]
    else:
        raise ValueError(kind)
    if out.exists():
        shutil.rmtree(out)  # an interrupted evaluation restarts from scratch (evaluator refuses to append)
    out.parent.mkdir(parents=True, exist_ok=True)
    logf = open(str(out) + ".log", "w")
    log(f"eval {kind} {kw} on GPU {gpu}")
    p = subprocess.Popen(cmd, cwd=str(EVAL_TREE), env=eval_env(gpu), stdout=logf, stderr=subprocess.STDOUT)
    return p, out


def run_eval_batch(jobs: list[dict], state: dict, gpus=("0", "1", "2", "3")) -> list[dict]:
    """Run evaluation jobs, at most one per GPU; returns per-job results with wall times."""
    pending = list(jobs)
    running: dict[str, tuple] = {}
    results = []
    while pending or running:
        for gpu in gpus:
            if gpu not in running and pending:
                job = pending.pop(0)
                p, out = start_eval(job["kind"], gpu, **{k: v for k, v in job.items() if k != "kind"})
                running[gpu] = (p, out, job, time.time())
        time.sleep(20)
        for gpu in list(running):
            p, out, job, t0 = running[gpu]
            rc = p.poll()
            if rc is None:
                continue
            wall = time.time() - t0
            ok = rc == 0 and (out / "snapshot_raw.jsonl").is_file() and sum(1 for _ in open(out / "snapshot_raw.jsonl")) == 256
            if not ok and not job.get("retried"):
                log(f"eval failed rc={rc} {job}; retrying once from scratch")
                job = dict(job, retried=True)
                pending.insert(0, job)
            else:
                results.append({**job, "rc": rc, "ok": ok, "wall_s": wall, "out": str(out)})
                log(f"eval done ok={ok} wall={wall/60:.1f} min {job}")
                if ok:
                    state["measured"].setdefault("eval_wall_s", []).append(wall)
                    save_state(state)
            del running[gpu]
    return results


# ----------------------------------------------------------------------------- checks

def integrity(seed: int, arm: str) -> dict:
    from camera_ready.analysis import core

    run = RUNS / f"seed{seed}" / arm
    problems = []
    manifest_name = "grpo_run_manifest.json" if arm == "grpo" else "maxrl_run_manifest.json"
    manifest = json.loads((run / manifest_name).read_text())
    cfg = manifest["config"]
    if cfg.get("seed") != seed:
        problems.append(f"manifest seed {cfg.get('seed')}")
    if manifest.get("gsm8k_dataset_sha") != "740312add88f781978c0658806c59bc2815b9866":
        problems.append("gsm8k sha")
    if manifest.get("pi0_lineage_id") != "f89fc90226a67a6a3c7374f9c13abadfcecda88f397ab812fa4130f1f425605b":
        problems.append("pi0 lineage")
    sched = json.loads((run / "policy_snapshot_schedule.json").read_text())
    for pct, step in ((25, 934), (45, 1681), (65, 2428), (100, 3736)):
        meta = run / f"pi_{pct:03d}" / "policy_metadata.json"
        if not meta.is_file() or json.loads(meta.read_text()).get("actual_step") != step:
            problems.append(f"snapshot pi_{pct:03d}")
        if sched["percentage_to_step"].get(str(pct)) != step:
            problems.append(f"schedule {pct}")
    ledger = core.load_ledger(run / "signal_ledger")
    audit = core.structural_audit(ledger, arm)
    if audit["status"] != "PASS":
        problems.append(f"audit {audit['problems']}")
    if audit["aggregate_token_is_ess_fraction"] < 0.95:
        problems.append(f"ESS stop {audit['aggregate_token_is_ess_fraction']}")
    alert = audit["aggregate_token_is_ess_fraction"] < 0.99
    steps_logged = 0
    log_path = run / "camera_ready_step_log.jsonl"
    if log_path.is_file():
        steps_logged = len({json.loads(l)["step"] for l in open(log_path) if '"grad_norm"' in l})
    result = {"seed": seed, "arm": arm, "problems": problems, "ess_alert": alert, "audit": audit,
              "grad_norm_steps_logged": steps_logged, "status": "PASS" if not problems else "FAIL"}
    (run / "camera_ready_integrity.json").write_text(json.dumps(result, indent=1, default=str))
    return result


def stage(seed: int) -> None:
    dest = STAGING / f"seed{seed}"
    for arm in ("grpo", "maxrl"):
        run = RUNS / f"seed{seed}" / arm
        d = dest / arm
        (d / "eval").mkdir(parents=True, exist_ok=True)
        for name in ("grpo_run_manifest.json", "maxrl_run_manifest.json", "policy_snapshot_schedule.json",
                     "prompt_length_audit.json", "camera_ready_launch.json", f"camera_ready_config_seed{seed}.yaml",
                     "camera_ready_integrity.json", "camera_ready_step_log.jsonl"):
            if (run / name).is_file():
                shutil.copy2(run / name, d / name)
        if (run / "camera_ready_step_log.jsonl").is_file():
            shutil.copy2(run / "camera_ready_step_log.jsonl", d / "step_log.jsonl")
        shutil.copytree(run / "signal_ledger", d / "ledger", dirs_exist_ok=True)
        for sub in ("eval", "eval_extra"):
            if (run / sub).is_dir():
                shutil.copytree(run / sub, d / sub, dirs_exist_ok=True)
        for f in (RUNS / f"seed{seed}").glob(f"{arm}_train_attempt*.log"):
            shutil.copy2(f, d / f.name)
        ts = sorted((run / "trainer").glob("checkpoint-*/trainer_state.json"))
        for t in ts:
            shutil.copy2(t, d / f"trainer_state_{t.parent.name}.json")
    subprocess.run(f"cd {dest} && find . -type f ! -name SHA256SUMS -exec sha256sum {{}} + > SHA256SUMS", shell=True)
    log(f"staged seed {seed} to {dest}")


def retain(seed: int) -> None:
    for arm in ("grpo", "maxrl"):
        run = RUNS / f"seed{seed}" / arm
        for snap in run.glob("pi_*"):
            if snap.name not in ("pi_025", "pi_045", "pi_065", "pi_100"):
                shutil.rmtree(snap)
        for ck in (run / "trainer").glob("checkpoint-*"):
            for f in ck.iterdir():
                if f.name != "trainer_state.json":
                    if f.is_dir():
                        shutil.rmtree(f)
                    else:
                        f.unlink()
    log(f"retention applied to seed {seed} (kept pi_025/045/065/100 and trainer_state.json)")


# ----------------------------------------------------------------------------- timing

def ledger_steps(run: Path) -> int:
    log_path = run / "camera_ready_step_log.jsonl"
    if not log_path.is_file():
        return 0
    last = 0
    with open(log_path) as f:
        for line in f:
            try:
                last = max(last, int(json.loads(line).get("step", 0)))
            except Exception:
                pass
    return last


def projected_train_hours(state: dict) -> float:
    m = state["measured"]
    if m.get("train_hours_completed"):
        return m["train_hours_completed"][-1]  # most recent completed pair (PREREG_RUNS §5)
    return float(m["train_hours_projected_from_smoke"])


def eval_hours(state: dict) -> float:
    walls = state["measured"].get("eval_wall_s") or [state["measured"]["eval_wall_s_initial"]]
    return max(walls) / 3600.0


def time_rule(state: dict, k: int, t0: dt.datetime) -> tuple[bool, str]:
    T = projected_train_hours(state)
    E = eval_hours(state)
    train_end = t0 + dt.timedelta(hours=T)
    finish = train_end + dt.timedelta(hours=E + math.ceil(6 * k / 4) * E + ANALYSIS_H + FACTCHECK_H + BUFFER_H)
    ok = train_end <= TRAIN_CUTOFF and finish <= RESULTS_DUE
    msg = (f"time rule k={k}: T_train={T:.2f}h E_eval={E:.2f}h train_end={train_end:%Y-%m-%dT%H:%MZ} "
           f"(cutoff {TRAIN_CUTOFF:%m-%dT%H:%MZ}) finish={finish:%Y-%m-%dT%H:%MZ} (due {RESULTS_DUE:%m-%dT%H:%MZ}) -> {ok}")
    return ok, msg


# ----------------------------------------------------------------------------- main loop

class Fatal(Exception):
    pass


def wait_all(procs: dict) -> None:
    for arm, p in procs.items():
        if p.poll() is None:
            log(f"waiting for running job {arm} to exit before stopping")
            p.wait()


def train_pair(seed: int, state: dict) -> list[dict]:
    """Train both arms; as soon as an arm finishes, its integrity check and K=16 endpoint evaluation
    (plus one bridge evaluation) run on that arm's freed GPUs. Returns evaluation results."""
    k_index = SEEDS.index(seed)
    arms = ("grpo", "maxrl") if k_index % 2 == 0 else ("maxrl", "grpo")
    assign = {arms[0]: GPU_PAIRS[0], arms[1]: GPU_PAIRS[1]}
    ports = {arms[0]: 29511, arms[1]: 29522}
    pstate = state["pairs"].setdefault(str(seed), {"attempts": {"grpo": 0, "maxrl": 0}, "gpus": assign})
    procs, t_start, last_step, last_move = {}, {}, {}, {}
    for arm in ("grpo", "maxrl"):
        pstate["attempts"][arm] += 1
        procs[arm] = start_training(seed, arm, assign[arm], ports[arm], pstate["attempts"][arm])
        t_start[arm] = last_move[arm] = time.time()
        last_step[arm] = 0
    pstate["started_at"] = now().isoformat()
    pstate["status"] = "training"
    save_state(state)
    finished: dict[str, float] = {}
    threads, eval_results = [], []

    def post_arm(arm: str) -> None:
        res = integrity(seed, arm)
        with STATE_LOCK:
            pstate.setdefault("integrity", {})[arm] = res["status"]
        log(f"integrity seed {seed} {arm}: {res['status']} {res['problems']} "
            f"ess={res['audit']['aggregate_token_is_ess_fraction']:.5f} alert={res['ess_alert']} "
            f"grad_norm_steps={res['grad_norm_steps_logged']}")
        if res["status"] != "PASS":
            return
        jobs = [{"kind": "protocol", "seed": seed, "arm": arm, "pct": 100}]
        with STATE_LOCK:
            todo = [b for b in BRIDGE_ITEMS if f"{b[0]}_{b[1]}" not in state["bridge_done"] + state.setdefault("bridge_claimed", [])]
            if todo:
                state["bridge_claimed"].append(f"{todo[0][0]}_{todo[0][1]}")
                jobs.append({"kind": "bridge", "arm": todo[0][0], "pct": todo[0][1]})
        out = run_eval_batch(jobs, state, gpus=tuple(assign[arm].split(",")))
        with STATE_LOCK:
            for r in out:
                if r["kind"] == "bridge":
                    state["bridge_claimed"].remove(f"{r['arm']}_{r['pct']}")
                    if r["ok"]:
                        state["bridge_done"].append(f"{r['arm']}_{r['pct']}")
            eval_results.extend(out)
            save_state(state)

    while len(finished) < 2:
        time.sleep(60)
        for arm, p in list(procs.items()):
            if arm in finished:
                continue
            rc = p.poll()
            steps = ledger_steps(RUNS / f"seed{seed}" / arm)
            if steps != last_step[arm]:
                last_step[arm], last_move[arm] = steps, time.time()
            with STATE_LOCK:
                pstate.setdefault("progress", {})[arm] = {
                    "step": steps, "elapsed_h": (time.time() - t_start[arm]) / 3600,
                    "s_per_step": (time.time() - t_start[arm]) / steps if steps else None}
            if rc is None and time.time() - last_move[arm] > 1800:
                log(f"watchdog: seed {seed} {arm} made no progress for 30 min at step {steps}; killing")
                p.kill()
                p.wait()
                rc = -9
            if rc is None:
                continue
            if rc == 0:
                finished[arm] = time.time() - t_start[arm]
                log(f"seed {seed} {arm} training exited 0 after {finished[arm]/3600:.2f} h")
                th = threading.Thread(target=post_arm, args=(arm,), daemon=False)
                th.start()
                threads.append(th)
                continue
            log(f"seed {seed} {arm} training died rc={rc} at step {steps}")
            pstate.setdefault("deaths", []).append({"arm": arm, "rc": rc, "step": steps, "at": now().isoformat()})
            if pstate["attempts"][arm] >= 2:
                pstate["status"] = "stopped_double_death"
                save_state(state)
                log(f"STOP: seed {seed} {arm} died twice")
                wait_all(procs)
                for th in threads:
                    th.join()
                raise Fatal(f"seed {seed} {arm} died twice")
            T = projected_train_hours(state)
            if now() + dt.timedelta(hours=T) > TRAIN_CUTOFF:
                pstate["status"] = "stopped_no_time_for_restart"
                save_state(state)
                log(f"STOP: no time to restart seed {seed} {arm}")
                wait_all(procs)
                for th in threads:
                    th.join()
                raise Fatal(f"no time to restart seed {seed} {arm}")
            pstate["attempts"][arm] += 1
            procs[arm] = start_training(seed, arm, assign[arm], ports[arm], pstate["attempts"][arm])
            t_start[arm] = last_move[arm] = time.time()
            last_step[arm] = 0
        save_state(state)
    for th in threads:
        th.join()
    pstate["train_hours"] = {arm: h / 3600 for arm, h in finished.items()}
    state["measured"].setdefault("train_hours_completed", []).append(max(finished.values()) / 3600)
    pstate["status"] = "trained"
    save_state(state)
    return eval_results


def main() -> None:
    QDIR.mkdir(parents=True, exist_ok=True)
    state = load_state()
    log("queue runner start")
    launched = []
    for seed in SEEDS:
        if (QDIR / "STOP").exists():
            log("STOP file present; not launching further pairs")
            break
        if state["pairs"].get(str(seed), {}).get("status") in ("done",):
            launched.append(seed)
            continue
        if seed != SEEDS[0]:
            ok, msg = time_rule(state, len(launched) + 1, now())
            log(msg)
            state["events"].append({"at": now().isoformat(), "time_rule": msg})
            save_state(state)
            if not ok:
                state["pairs"][str(seed)] = {"status": "not_launched_time_rule", "reason": msg}
                save_state(state)
                break
        launched.append(seed)
        try:
            res = train_pair(seed, state)
        except Fatal as exc:
            state["stopped"] = str(exc)
            save_state(state)
            log(f"QUEUE STOPPED: {exc}")
            return
        pstate = state["pairs"][str(seed)]
        if any(v != "PASS" for v in pstate.get("integrity", {}).values()) or len(pstate.get("integrity", {})) < 2:
            pstate["status"] = "integrity_fail"
            state["stopped"] = f"integrity failure seed {seed}"
            save_state(state)
            log(f"QUEUE STOPPED: integrity failure for seed {seed}")
            return
        pstate["endpoint_eval_ok"] = all(r["ok"] for r in res if r["kind"] == "protocol") and \
            sum(1 for r in res if r["kind"] == "protocol") == 2
        stage(seed)
        retain(seed)
        pstate["status"] = "done"
        save_state(state)
    log("training phase over; deferred evaluations")
    deferred = [{"kind": "protocol", "seed": s, "arm": a, "pct": p}
                for s in launched for p in EVAL_PCTS_DEFERRED for a in ("grpo", "maxrl")
                if state["pairs"].get(str(s), {}).get("status") == "done"]
    remaining_bridge = [b for b in BRIDGE_ITEMS if f"{b[0]}_{b[1]}" not in state["bridge_done"]]
    deferred += [{"kind": "bridge", "arm": a, "pct": p} for a, p in remaining_bridge]
    res = run_eval_batch(deferred, state)
    for r in res:
        if r["kind"] == "bridge" and r["ok"]:
            state["bridge_done"].append(f"{r['arm']}_{r['pct']}")
        if r["kind"] == "protocol" and r["ok"]:
            state["deferred_done"].append(f"{r['seed']}_{r['arm']}_{r['pct']}")
    save_state(state)
    for s in launched:
        if state["pairs"].get(str(s), {}).get("status") == "done":
            stage(s)
    # extra endpoint batches, only while each job can still finish before the analysis/fact-check window
    extra = [{"kind": "extra", "seed": s, "arm": a, "pct": 100, "batch": b}
             for s in launched for b in (1, 2, 3) for a in ("grpo", "maxrl")
             if state["pairs"].get(str(s), {}).get("status") == "done"]
    latest_start = RESULTS_DUE - dt.timedelta(hours=ANALYSIS_H + FACTCHECK_H + BUFFER_H)
    while extra and not (QDIR / "STOP").exists():
        if now() + dt.timedelta(hours=eval_hours(state)) > latest_start:
            log(f"no time for {len(extra)} remaining extra batches; skipped")
            state["events"].append({"at": now().isoformat(), "extra_skipped": len(extra)})
            break
        batch, extra = extra[:4], extra[4:]
        res = run_eval_batch(batch, state)
        for r in res:
            if r["ok"]:
                state["extra_done"].append(f"{r['seed']}_{r['arm']}_b{r['batch']}")
        save_state(state)
    for s in launched:
        if state["pairs"].get(str(s), {}).get("status") == "done":
            stage(s)
    state["queue_finished_at"] = now().isoformat()
    save_state(state)
    log("QUEUE FINISHED: GPUs idle")


if __name__ == "__main__":
    main()
