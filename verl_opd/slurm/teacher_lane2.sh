#!/bin/bash
# One GPU lane of teacher benchmarks: bash teacher_lane.sh <GPU> <PORT> <TAG:MODEL:MAXTOK[:MAXMODELLEN]> ...
# Same MathArena protocol as every student eval (7 comps, T0.7/top_p0.8/top_k20, n=4; enable_thinking=false is a no-op for DS).
set -u
source /home/sya225/opd/env.sh none
S=$OPD_HOME/scripts; L=$SCRATCH/logs
G=$1; P=$2; shift 2
export CONCURRENCY=100 GPU_UTIL=0.9 MATH_ONLY=1
export COMPS="${COMPS:-aime/aime_2024_I aime/aime_2024_II aime/aime_2025 hmmt/hmmt_feb_2025 hmmt/hmmt_feb_2026 custom/amc custom/beyondaime}"
for item in "$@"; do
  IFS=: read TAG MODEL MT MML <<< "$item"
  for i in $(seq 1 60); do m=$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits); [ "$m" -lt 3000 ] && break; sleep 15; done
  echo "LANE_START $TAG gpu=$G $(date -Is)"
  MAX_TOKENS=$MT MAX_MODEL_LEN=${MML:-40960} bash $S/teacher_eval.sh $TAG "$MODEL" $G $P > $L/teval_$TAG.log 2>&1
  echo "EVAL_DONE $TAG rc=$? $(date -Is)"
  ( cd /home/sya225 && python3 ma_score.py $L/v5_$TAG.log )
done
echo "LANE_DONE gpu=$G $(date -Is)"
