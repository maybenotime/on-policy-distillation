#!/bin/bash
# cs-venus-19 job 271952: same three checkpoints as driver19x but only AIME24 I/II, AIME25, AMC (user, 9-24 22:50).
# Kills driver19x's evals first; MathArena resumes per-problem, so finished problems of these comps are not redone.
set -u
source /home/sya225/opd/env.sh none
S=$OPD_HOME/scripts; L=$SCRATCH/logs; H=/home/sya225/logs; C=$SCRATCH/checkpoints
Q3=Qwen/Qwen3-1.7B; DS=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
export CONCURRENCY=100 GPU_UTIL=0.9 MATH_ONLY=1
C3="aime/aime_2024_I aime/aime_2024_II aime/aime_2025 custom/amc"
echo "DRIVER19Y_START $(date -Is)"
for pat in driver19x.sh "teacher_eval.sh q3i17_fromM500-s1" "teacher_eval.sh ds15b_fromJustRL-s150" "fromM500_step1" "fromJustRL_step150" "--port 832"; do pkill -f -- "$pat" 2>/dev/null; done
sleep 20; for pat in "fromM500_step1" "fromJustRL_step150" "--port 832"; do pkill -9 -f -- "$pat" 2>/dev/null; done
for i in $(seq 1 40); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1); [ "$m" -lt 3000 ] && break; sleep 15; done
echo "GPUS_FREE mem=$m $(date -Is)"
ev () { local T=$1 HF=$2 G=$3 P=$4 BASE=$5 MT=$6; shift 6
  MAX_TOKENS=$MT COMPS="$*" bash $S/teacher_eval.sh $T $HF $G $P $BASE > $L/teval_$T.log 2>&1
  echo "EVAL_DONE $T rc=$? $(date -Is)"; }
( ev q3i17_fromM500-s100-16k $C/hf_q3i17_fromM500_step100 0 8320 $Q3 8192 $C3 ) > $H/y19_0.out 2>&1 &
( ev q3i17_fromM500-s150-16k $C/hf_q3i17_fromM500_step150 1 8321 $Q3 8192 $C3 ) > $H/y19_1.out 2>&1 &
( ev ds15b_fromJustRL-s150-16k  $C/hf_ds15b_fromJustRL_step150 2 8322 $DS 16384 aime/aime_2024_I aime/aime_2024_II aime/aime_2025 ) > $H/y19_2.out 2>&1 &
( ev ds15b_fromJustRL-s150b-16k $C/hf_ds15b_fromJustRL_step150 3 8323 $DS 16384 custom/amc ) > $H/y19_3.out 2>&1 &
wait
cat $H/y19_?.out
echo "DRIVER19Y_DONE $(date -Is)"
