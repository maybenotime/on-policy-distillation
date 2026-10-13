#!/bin/bash
# 19 GPU 0 only: MathArena (7 comps, 8k, n=4) evals of the Qwen3-1.7B <- 8B student, step 200 then step 50, sequential.
source /home/USER/opd/env.sh none; S=$OPD_HOME/scripts; L=$SCRATCH/logs
export MAX_TOKENS=8192 CONCURRENCY=100 GPU_UTIL=0.9 MATH_ONLY=1
export COMPS="aime/aime_2024_I aime/aime_2024_II aime/aime_2025 custom/amc custom/beyondaime hmmt/hmmt_feb_2025 hmmt/hmmt_feb_2026"
te () { bash $S/teacher_eval.sh $1 $2 0 8076 > $L/teval_$1.log 2>&1; echo "EVAL_DONE $1 rc=$? $(date -Is)"; }
te q3i17-8B-s200-8k $SCRATCH/checkpoints/hf_q3i17_from8B_step200
te q3i17-8B-s50-8k $SCRATCH/checkpoints/hf_q3i17_from8B_step50
echo "EVAL_Q8B_DONE $(date -Is)"
