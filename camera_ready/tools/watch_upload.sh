#!/bin/bash
# Laptop-side: watch upload progress and seed training step time; pause the uploader (SIGSTOP) if the
# recent mean step_time exceeds the pre-upload baseline by more than 10 %. Exits after $1 seconds or on completion.
HOST=${HOST:-autodl-4080-4}
BASE=${BASE:-15.9}
sshq() { perl -e 'alarm shift; exec @ARGV' 45 ssh -o ConnectTimeout=20 -o ConnectionAttempts=1 -o BatchMode=yes "$HOST" "$@"; }
end=$((SECONDS+${1:-1200}))
while [ $SECONDS -lt $end ]; do
  sleep 120
  out=$(sshq 'for a in grpo maxrl; do f=$(ls -t /root/autodl-tmp/runs/seed4*/$a/camera_ready_step_log.jsonl 2>/dev/null | head -1); [ -n "$f" ] && tail -20 $f | /root/autodl-tmp/envs/rlvr/bin/python -c "import sys,json; r=[json.loads(l) for l in sys.stdin if \"step_time\" in l]; print(round(sum(x[\"step_time\"] for x in r)/max(len(r),1),2))"; done | sort -n | tail -1; grep -c "uploaded " /root/autodl-tmp/logs/hf_upload.log; grep -E "ROUND|giving up|failed" /root/autodl-tmp/logs/hf_upload.log | tail -2' 2>/dev/null)
  [ -z "$out" ] && continue
  st=$(echo "$out" | sed -n 1p)
  if [ -n "$st" ] && awk "BEGIN{exit !($st > $BASE*1.10)}"; then
    sshq 'pkill -STOP -f "[h]f_upload.py"'; echo "PAUSED upload: step_time $st > 1.1 x $BASE"; exit 0
  fi
  if echo "$out" | grep -q -E "ROUND [12] (COMPLETE|VERIFY FAILED)|giving up"; then echo "UPLOAD END"; echo "$out"; exit 0; fi
done
echo "TIMER"; echo "$out"
