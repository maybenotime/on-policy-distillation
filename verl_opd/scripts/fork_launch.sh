#!/bin/bash
# 4 students in parallel, one GPU each (teacher is 8B bf16 = ~16 GB, fits alongside).
L=/localscratch/sya225/logs
CK=/localscratch/sya225/checkpoints
bash /home/sya225/opd/scripts/fork_worker.sh Qwen3-4B    Qwen/Qwen3-4B          0 > $L/fork_Qwen3-4B.log   2>&1 &
bash /home/sya225/opd/scripts/fork_worker.sh Qwen3-1.7B  Qwen/Qwen3-1.7B        1 > $L/fork_Qwen3-1.7B.log 2>&1 &
bash /home/sya225/opd/scripts/fork_worker.sh opd-4b      $CK/hf_opd30k_step424  2 > $L/fork_opd-4b.log     2>&1 &
bash /home/sya225/opd/scripts/fork_worker.sh opd-1.7b    $CK/hf_opd17b_step424  3 > $L/fork_opd-1.7b.log   2>&1 &
wait
echo "FORK_ALL_DONE $(date -Is)"
