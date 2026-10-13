#!/bin/bash
# Code-300 16k student: MathArena at 16k, concurrency 64, vLLM gpu util 0.9 (HE+/LCB outputs already on disk -> skipped).
source /home/sya225/opd/env.sh none
MAX_TOKENS=16384 CONCURRENCY=100 GPU_UTIL=0.9 bash /home/sya225/opd/scripts/post_run.sh opd-1.7b-fromK4Bcode300-16k $SCRATCH/checkpoints/opd_code_17b_fromK4Bcode300_16k 3 8033 > $SCRATCH/logs/post_run_K4Bcode300_16k_v2.log 2>&1
echo "CODE300_EVAL_DONE rc=$? $(date -Is)"
