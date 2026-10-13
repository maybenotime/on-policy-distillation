#!/bin/bash
# kill orphaned vLLM EngineCore children of the killed fork_rollout (they keep the GPU memory)
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
  c=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-120)
  if echo "$c" | grep -qE "EngineCore|multiprocessing" && ! echo "$c" | grep -q "fork_entropy"; then
    ppid=$(ps -o ppid= -p $p | tr -d ' '); echo "kill $p (ppid $ppid): $c"; kill -9 $p
  fi
done
sleep 5; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
