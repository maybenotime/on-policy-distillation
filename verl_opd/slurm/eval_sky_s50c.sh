#!/bin/bash
# 19 GPU 2: the two HMMT comps of the Skywork s50 eval, in parallel (separate tag).
source /home/sya225/opd/env.sh none; S=$OPD_HOME/scripts; L=$SCRATCH/logs
export MAX_TOKENS=16384 CONCURRENCY=100 GPU_UTIL=0.9 MATH_ONLY=1
export COMPS="hmmt/hmmt_feb_2025 hmmt/hmmt_feb_2026"
HF=$SCRATCH/checkpoints/hf_ds15b_fromSky7B_step50
T=ds15b-Sky7B-s50c-16k
echo "EVAL_START $T $(date -Is)"
bash $S/teacher_eval.sh $T $HF 2 8080 > $L/teval_$T.log 2>&1
echo "EVAL_DONE $T rc=$? $(date -Is)"
