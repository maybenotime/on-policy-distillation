#!/bin/bash
# 19 GPU 1: the slow tail comps of the Skywork s50 eval, run in parallel with the main s50 job (separate tag -> separate files).
source /home/sya225/opd/env.sh none; S=$OPD_HOME/scripts; L=$SCRATCH/logs
export MAX_TOKENS=16384 CONCURRENCY=100 GPU_UTIL=0.9 MATH_ONLY=1
export COMPS="custom/beyondaime"
HF=$SCRATCH/checkpoints/hf_ds15b_fromSky7B_step50
T=ds15b-Sky7B-s50b-16k
echo "EVAL_START $T $(date -Is)"
bash $S/teacher_eval.sh $T $HF 1 8079 > $L/teval_$T.log 2>&1
echo "EVAL_DONE $T rc=$? $(date -Is)"
