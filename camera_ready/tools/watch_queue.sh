#!/bin/bash
# Laptop-side helper: exit after $1 seconds (TIMER) or as soon as the box queue log records an important event (EVENT).
HOST=${HOST:-autodl-4080-4}
cnt() { ssh -o ConnectTimeout=20 "$HOST" 'grep -c -E "died|STOP|integrity|exited|eval done|QUEUE|time rule|watchdog|failed|staged" /root/autodl-tmp/queue/queue.log; true' 2>/dev/null | head -1; }
end=$((SECONDS+${1:-1500}))
base=$(cnt); [ -z "$base" ] && base=0
while [ $SECONDS -lt $end ]; do
  sleep 60
  n=$(cnt)
  if [ -n "$n" ] && [ "$n" != "$base" ]; then echo "EVENT ($base -> $n)"; ssh -o ConnectTimeout=20 "$HOST" 'tail -6 /root/autodl-tmp/queue/queue.log'; exit 0; fi
done
echo "TIMER"
