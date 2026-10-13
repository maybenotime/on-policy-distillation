#!/bin/bash
# 19 GPUs 0/1 (free): run the last two items of the k3 8k chain in parallel now (same tags; the serial chain on GPU 3
# will find them finished and skip). s50 on GPU 0, untrained Qwen3-1.7B-Base on GPU 1.
source /home/sya225/opd/env.sh none; S=$OPD_HOME/scripts; L=$SCRATCH/logs
export MAX_TOKENS=8192 CONCURRENCY=100 GPU_UTIL=0.9 MATH_ONLY=1 MAX_MODEL_LEN=32768
export COMPS="aime/aime_2024_I aime/aime_2024_II aime/aime_2025 custom/amc custom/beyondaime"
CK=$SCRATCH/checkpoints/opd_q3b17_fromQ3B4Bgrpo_dapo_16k_k3; BASE=Qwen/Qwen3-1.7B-Base
te () { bash $S/teacher_eval.sh $1 $2 $3 $4 > $L/teval_$1.par.log 2>&1; echo "EVAL_DONE $1 rc=$? $(date -Is)"; }
( te q3b17-BaseGRPO-k3-s50-8k $SCRATCH/checkpoints/hf_q3b17_BaseGRPO_k3_step50 0 8073 ) &
sleep 20; ( te Q3-1.7B-Base-8k $BASE 1 8074 ) &
wait
echo "EVAL_K3_REST_DONE $(date -Is)"
