#!/bin/bash
# Camera-ready status from the laptop: queue state, step counts, ETA, last log lines, GPU and disk usage.
# usage: camera_ready/status.sh [host]   (default host: autodl-4080-4)
HOST=${1:-autodl-4080-4}
ssh -o ConnectTimeout=20 "$HOST" 'bash -s' <<'REMOTE'
CR=/root/autodl-tmp
echo "== box time (UTC): $(date -u +%FT%TZ)"
echo "== screens"; screen -ls | grep -E "^\s+[0-9]+\." || echo "  none"
echo "== queue state"
if [ -f $CR/queue/state.json ]; then
  $CR/envs/rlvr/bin/python - <<'PY'
import json, datetime as dt
s = json.load(open("/root/autodl-tmp/queue/state.json"))
now = dt.datetime.now(dt.timezone.utc)
for seed, p in s.get("pairs", {}).items():
    print(f"  seed {seed}: status={p.get('status')} gpus={p.get('gpus')} attempts={p.get('attempts')} integrity={p.get('integrity')}")
    for arm, pr in (p.get("progress") or {}).items():
        step, sps = pr.get("step") or 0, pr.get("s_per_step")
        eta = f"{(3736 - step) * sps / 3600:.1f} h" if sps else "?"
        print(f"     {arm:5s} step {step:4d}/3736  elapsed {pr.get('elapsed_h', 0):.2f} h  s/step {sps if sps is None else round(sps, 1)}  ETA {eta}")
    if p.get("deaths"): print("     deaths:", p["deaths"])
print("  bridge done:", s.get("bridge_done"), "| deferred done:", len(s.get("deferred_done", [])), "| extra done:", len(s.get("extra_done", [])))
if s.get("stopped"): print("  STOPPED:", s["stopped"])
if s.get("queue_finished_at"): print("  QUEUE FINISHED at", s["queue_finished_at"])
PY
  echo "== queue log (last 8)"; tail -8 $CR/queue/queue.log
else
  echo "  queue not started"
fi
for f in $(ls -t $CR/runs/seed*/*_train_attempt*.log 2>/dev/null | head -2); do
  echo "== $f (last line)"; tail -c 3000 $f | tr '\r' '\n' | grep -v '^\s*$' | tail -1 | cut -c1-220
done
echo "== GPUs"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader
echo "== disk"; df -h / $CR | tail -2
REMOTE
