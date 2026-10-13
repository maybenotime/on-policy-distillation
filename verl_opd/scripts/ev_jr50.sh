#!/bin/bash
# ev_jr50.sh <SUFFIX> <GPU> <PORT> <COMPS...>   — JustRL DAPO s50, 16k, split by competition
set -u
source /home/USER/opd/env.sh none
S=$OPD_HOME/scripts; L=$SCRATCH/logs
SUF=$1; G=$2; PORT=$3; shift 3
export MAX_TOKENS=16384 CONCURRENCY=100 GPU_UTIL=0.9 MATH_ONLY=1
export COMPS="$*"
HF=$SCRATCH/checkpoints/hf_ds15b_fromJustRL_step50
T=ds15b-JustRL-s50${SUF}-16k
echo "EVAL_START $T gpu=$G port=$PORT comps=[$COMPS] $(date -Is)"
bash $S/teacher_eval.sh $T $HF $G $PORT deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B > $L/teval_$T.log 2>&1
echo "EVAL_DONE $T rc=$? $(date -Is)"
