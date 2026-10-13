#!/bin/bash
# 19: MathArena-only (16k) evals of the k3 Base-student step 50 (GPU 0) and untrained Qwen3-1.7B-Base (GPU 3), in parallel.
# Qwen3-*-Base has max_position_embeddings 32768 -> MAX_MODEL_LEN=32768 (eval_v5 default 40960 fails validation).
source /home/sya225/opd/env.sh none; S=$OPD_HOME/scripts; L=$SCRATCH/logs
export MAX_TOKENS=16384 CONCURRENCY=100 GPU_UTIL=0.9 MATH_ONLY=1 MAX_MODEL_LEN=32768
export COMPS="aime/aime_2024_I aime/aime_2024_II aime/aime_2025 custom/amc custom/beyondaime hmmt/hmmt_feb_2025 hmmt/hmmt_feb_2026"
te () { bash $S/teacher_eval.sh $1 $2 $3 $4 > $L/teval_$1.log 2>&1; echo "EVAL_DONE $1 rc=$? $(date -Is)"; }
( te q3b17-BaseGRPO-k3-s50-16k $SCRATCH/checkpoints/hf_q3b17_BaseGRPO_k3_step50 0 8064 ) &
sleep 20; ( te Q3-1.7B-Base-16k Qwen/Qwen3-1.7B-Base 3 8065 ) &
wait
echo "EVAL_Q3B_DONE $(date -Is)"
