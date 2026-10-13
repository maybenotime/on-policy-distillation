#!/bin/bash
# 19 GPU 3: rerun the untrained Qwen3-1.7B-Base MathArena eval (server had died; matharena skips finished problems)
source /home/sya225/opd/env.sh none; S=$OPD_HOME/scripts; L=$SCRATCH/logs
export MAX_TOKENS=16384 CONCURRENCY=100 GPU_UTIL=0.9 MATH_ONLY=1 MAX_MODEL_LEN=32768
export COMPS="aime/aime_2024_I aime/aime_2024_II aime/aime_2025 custom/amc custom/beyondaime hmmt/hmmt_feb_2025 hmmt/hmmt_feb_2026"
bash $S/teacher_eval.sh Q3-1.7B-Base-16k Qwen/Qwen3-1.7B-Base 3 8066 > $L/teval_Q3-1.7B-Base-16k.rerun.log 2>&1
echo "EVAL_DONE Q3-1.7B-Base-16k rc=$? $(date -Is)"
