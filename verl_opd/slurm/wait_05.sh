#!/bin/bash
# login-node waiter: when the 6-GPU 05 holder starts, verify visible GPUs and launch Qwen3-1.7B <- Qwen3-4B (19-style: student 2 GPUs, teacher 1)
J=$1
until [ "$(squeue -j $J -h -o '%T')" = "RUNNING" ]; do sleep 120; done
echo "holder $J running $(date -Is)"; sleep 30
n=$(srun --jobid=$J --overlap -c 1 --ntasks=1 nvidia-smi -L 2>/dev/null | wc -l); echo "visible GPUs: $n"
srun --jobid=$J --overlap -c 1 --ntasks=1 nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader
if [ "$n" -ge 3 ]; then
  sed -i "s/--jobid=[0-9]*/--jobid=$J/" /home/sya225/opd/scripts/chain05_q3i4b.sh
  tmux new -d -s q4b05 "srun --jobid=$J --overlap -c 48 --ntasks=1 bash /home/sya225/opd/scripts/chain05_q3i4b.sh > /home/sya225/logs/q4b05.log 2>&1"
  echo "4B run launched on $J $(date -Is)"
else
  echo "NOT ENOUGH GPUS ($n) - not launching"
fi
