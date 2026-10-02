#!/bin/bash
# Laptop-side helper: exit after $1 seconds (TIMER), as soon as the box queue log records an important event (EVENT),
# or report UNREACHABLE/BACK transitions. Every ssh call has a hard 45 s limit (macOS has no `timeout`).
HOST=${HOST:-autodl-4080-4}
sshq() { perl -e 'alarm shift; exec @ARGV' 45 ssh -o ConnectTimeout=20 -o ConnectionAttempts=1 -o ServerAliveInterval=10 -o ServerAliveCountMax=2 -o BatchMode=yes "$HOST" "$@"; }
cnt() { sshq 'grep -c -E "died|STOP|integrity|exited|eval done|QUEUE|time rule|watchdog|failed|staged" /root/autodl-tmp/queue/queue.log; true' 2>/dev/null | head -1; }
end=$((SECONDS+${1:-1400}))
base=$(cnt); down=0; [ -z "$base" ] && down=1
while [ $SECONDS -lt $end ]; do
  sleep 60
  n=$(cnt)
  if [ -z "$n" ]; then down=$((down+1)); continue; fi
  if [ -z "$base" ]; then echo "BACK after $down failed polls"; sshq 'tail -3 /root/autodl-tmp/queue/queue.log'; exit 0; fi
  if [ "$n" != "$base" ]; then echo "EVENT ($base -> $n)"; sshq 'tail -6 /root/autodl-tmp/queue/queue.log'; exit 0; fi
done
if [ -z "$base" ] || [ $down -gt 0 ]; then echo "TIMER (unreachable polls: $down)"; else echo "TIMER"; fi
