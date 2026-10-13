#!/bin/bash
# M500Fix s100 eval split over all 4 GPUs of cs-venus-19 (training is finished, GPUs 0-2 idle); replaces the single-GPU
# driver19s eval.  Tag s100 keeps the comps the single-GPU run had started (MathArena resumes them).
set -u
source /home/sya225/opd/env.sh none
S=$OPD_HOME/scripts; L=$SCRATCH/logs; H=/home/sya225/logs
STU=Qwen/Qwen3-1.7B; HF=$SCRATCH/checkpoints/hf_q3i17_tpM500_shapeStuFix_step100
echo "DRIVER19T_START $(date -Is)"
pkill -f driver19s.sh; pkill -f "q3TP_M500_shapeStuFix"; pkill -f "hf_q3i17_tpM500_shapeStuFix_step100"; sleep 15
pkill -9 -f "q3TP_M500_shapeStuFix"; pkill -9 -f "hf_q3i17_tpM500_shapeStuFix_step100"; sleep 5
for i in $(seq 1 40); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1); [ "$m" -lt 3000 ] && break; sleep 15; done
echo "GPUS_FREE mem=$m $(date -Is)"
[ -s $HF/config.json ] || { echo "NO HF $HF"; exit 1; }
export MAX_TOKENS=8192 CONCURRENCY=100 GPU_UTIL=0.9 MATH_ONLY=1
ev () { local T=$1 G=$2 P=$3; shift 3
  COMPS="$*" bash $S/teacher_eval.sh $T $HF $G $P $STU > $L/teval_$T.log 2>&1
  echo "EVAL_DONE $T rc=$? $(date -Is)"; }
( ev q3TP_M500_shapeStuFix-s100-16k  0 8270 custom/amc hmmt/hmmt_feb_2026 ) > $H/ev_m500fix100_a.out 2>&1 &
( ev q3TP_M500_shapeStuFix-s100b-16k 1 8271 custom/beyondaime )            > $H/ev_m500fix100_b.out 2>&1 &
( ev q3TP_M500_shapeStuFix-s100c-16k 2 8272 aime/aime_2024_I aime/aime_2024_II aime/aime_2025 ) > $H/ev_m500fix100_c.out 2>&1 &
( ev q3TP_M500_shapeStuFix-s100d-16k 3 8273 hmmt/hmmt_feb_2025 )           > $H/ev_m500fix100_d.out 2>&1 &
wait
cat $H/ev_m500fix100_?.out
echo "M500FIX_EVAL_DONE $(date -Is)"
