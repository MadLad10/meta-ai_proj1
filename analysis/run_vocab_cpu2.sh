#!/bin/bash
# Same-image vocabulary test, resumable: re-runs each job until it has 240 rows (the CPU process is killed when it reaches
# its memory cap; vocab_cpu.py resumes from its jsonl). Waits for any running vocab_cpu.py first.
cd "$(dirname "$0")/.."
export HF_HUB_DISABLE_TELEMETRY=1 MALLOC_ARENA_MAX=2 ONEDNN_PRIMITIVE_CACHE_CAPACITY=8
for job in "gdino ruod10" "owlv2 ruod10" "gdino duo4rev"; do
  set -- $job; f=analysis/vocab_cpu/$1_$2_pc60.jsonl
  for attempt in $(seq 1 15); do
    while pgrep -f "analysis/vocab_cpu.py" > /dev/null; do sleep 10; done
    n=$( [ -f "$f" ] && wc -l < "$f" || echo 0 ); [ "$n" -ge 240 ] && break
    echo "$(date +%H:%M) $1 $2 attempt $attempt from $n rows"
    systemd-run --user --scope -q -p MemoryMax=3500M -p MemorySwapMax=0 .venv_cpu/bin/python analysis/vocab_cpu.py $1 $2 --per_class 60
  done
  echo "$(date +%H:%M) $1 $2 done: $(wc -l < "$f") rows"
done
echo ALL_DONE
