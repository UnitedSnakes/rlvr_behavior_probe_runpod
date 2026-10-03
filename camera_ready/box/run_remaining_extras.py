"""Run the extra endpoint batches that the queue skipped at its pre-registered cut-off (Sam, 2026-10-03 ~20:00 UTC).

Seed 45, batch 3, both arms. Uses the queue's own evaluation path (`queue_runner.run_eval_batch`: same command,
environment, one retry, 256-line check) and re-stages seed 45 afterwards. Hard stop 2026-10-04 02:00 UTC
(22:00 ET): a job is not started if it cannot finish by then, and running jobs are killed at that time.
Run on the box from /root/autodl-tmp/repos/cr: `envs/rlvr/bin/python camera_ready/box/run_remaining_extras.py`.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import subprocess
import sys
import threading
from pathlib import Path

CR_TREE = Path("/root/autodl-tmp/repos/cr")
sys.path.insert(0, str(CR_TREE))
spec = importlib.util.spec_from_file_location("queue_runner", CR_TREE / "camera_ready/box/queue_runner.py")
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)

HARD_STOP = dt.datetime(2026, 10, 4, 2, 0, tzinfo=dt.timezone.utc)
JOBS = [{"kind": "extra", "seed": 45, "arm": a, "pct": 100, "batch": 3} for a in ("grpo", "maxrl")]


def kill_at_hard_stop() -> None:
    q.log("HARD STOP reached (2026-10-04 02:00 UTC): killing remaining extra-batch evaluations")
    subprocess.run(["pkill", "-f", "[e]val_run.py --run-dir /root/autodl-tmp/runs/seed45"], check=False)


def main() -> None:
    state = q.load_state()
    todo = [j for j in JOBS if f"{j['seed']}_{j['arm']}_b{j['batch']}" not in state.get("extra_done", [])]
    if not todo:
        q.log("remaining extras: nothing to do")
        return
    projected = q.now() + dt.timedelta(hours=q.eval_hours(state))
    if projected > HARD_STOP:
        q.log(f"remaining extras NOT started: projected end {projected:%H:%MZ} after hard stop 02:00Z")
        return
    q.log(f"remaining extras (Sam, after the pre-registered cut-off; hard stop 02:00Z): {todo}; "
          f"projected end {projected:%H:%MZ}")
    timer = threading.Timer((HARD_STOP - q.now()).total_seconds(), kill_at_hard_stop)
    timer.daemon = True
    timer.start()
    res = q.run_eval_batch(todo, state, gpus=("0", "1"))
    timer.cancel()
    for r in res:
        if r["ok"]:
            state["extra_done"].append(f"{r['seed']}_{r['arm']}_b{r['batch']}")
    state.setdefault("events", []).append({"at": q.now().isoformat(), "remaining_extras": [
        {k: r[k] for k in ("seed", "arm", "batch", "ok", "wall_s")} for r in res]})
    q.save_state(state)
    q.stage(45)
    ok = all(r["ok"] for r in res) and len(res) == len(todo)
    q.log(f"REMAINING EXTRAS {'DONE' if ok else 'INCOMPLETE'}: {[(r['arm'], r['ok']) for r in res]}")


if __name__ == "__main__":
    main()
