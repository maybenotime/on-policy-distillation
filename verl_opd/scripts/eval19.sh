#!/bin/bash
# job 265251 (node-B, 4 GPUs): the evals not yet done, 16k everywhere.
#  GPU0: code-data Math-500 student (ckpt on 19)   GPU1: Math-1200 teacher, then Code-1200 teacher
#  GPU2: Math-500 teacher, then Code-300 teacher   GPU3: RLCR-4B teacher
source /home/USER/opd/env.sh none; S=$OPD_HOME/scripts; L=$SCRATCH/logs
export MAX_TOKENS=16384 CONCURRENCY=100 GPU_UTIL=0.9
te () { bash $S/teacher_eval.sh $1 $2 $3 $4 > $L/teval_$1.log 2>&1; echo "TEVAL_DONE $1 rc=$? $(date -Is)"; }
( bash $S/post_run.sh opd-1.7b-fromK4Bmath500-16k $SCRATCH/checkpoints/opd_code_17b_fromK4Bmath500_16k 0 8050 > $L/post_run_K4Bmath500_16k.log 2>&1; echo "STUDENT_EVAL_DONE K4Bmath500_16k rc=$? $(date -Is)" ) &
( te K4Bmath1200-16k Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step1200 1 8051; te K4Bcode1200-16k Keven16/Qwen3-4B-Non-Thinking-RL-Code-Step1200 1 8051 ) &
( te K4Bmath500-16k  Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500  2 8052; te K4Bcode300-16k  Keven16/Qwen3-4B-Non-Thinking-RL-Code-Step300  2 8052 ) &
( te RLCR4B-16k resistz/RLCR-Calibrated-Qwen3-4B-DAPO-Math14k-3Epoch-330Step 3 8053 ) &
wait
echo "EVAL19_DONE $(date -Is)"
