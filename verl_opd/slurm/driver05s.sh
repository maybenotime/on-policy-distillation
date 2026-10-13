#!/bin/bash
# q3i17_tpStu_flat: student's own preferences (A = Qwen3-1.7B base, i.e. no preference signal) with a
# CONSTANT target entropy equal to Math-Step500's sequence-mean entropy, recomputed every step.
# Paired with q3i17_tpStu_shapeM500 (-0.1): on this pair the two differ ONLY in the per-position
# entropy profile, because M500's mean entropy already matches the student's (0.25 vs 0.245).
set -u
source /home/sya225/opd/env.sh none
S=$OPD_HOME/scripts; L=$SCRATCH/logs; H=/home/sya225/logs
TAG=q3i17_tpStu_flat
STU=Qwen/Qwen3-1.7B
M5=Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500

echo "DRIVER05S_START $(date -Is)"
SAVE_FREQ=50 bash $S/chain19_tp.sh $TAG flat $STU $M5 0,1,2 /tmp/ray_q3flat 1.0 100
echo "TRAIN_RC_Q3FLAT=$? $(date -Is)"

export MAX_TOKENS=8192 CONCURRENCY=100 GPU_UTIL=0.9 MATH_ONLY=1
ev () { local T=$1 G=$2 P=$3; shift 3
  COMPS="$*" bash $S/teacher_eval.sh $T $HF $G $P $STU > $L/teval_$T.log 2>&1
  echo "EVAL_DONE $T rc=$? $(date -Is)"; }
for ST in 100 50; do
  CK=$SCRATCH/checkpoints/opd_$TAG/global_step_$ST/actor
  HF=$SCRATCH/checkpoints/hf_${TAG}_step$ST
  [ -d $CK ] || { echo "NO CHECKPOINT $CK — skipping s$ST"; continue; }
  if [ ! -s $HF/config.json ]; then
    CUDA_VISIBLE_DEVICES=0 "$VERL_VENV/bin/python" -m verl.model_merger merge \
        --backend fsdp --local_dir $CK --target_dir $HF 2>&1 | tail -1
    "$CODE_EVAL_ENV/bin/python" -c "from transformers import AutoTokenizer as T; T.from_pretrained('$STU').save_pretrained('$HF')"
  fi
  echo "MERGE_DONE q3flat s$ST $(date -Is)"
  ( ev q3TP_Stu_flat-s${ST}-16k  0 8220 custom/amc hmmt/hmmt_feb_2026 ) > $H/ev_q3flat${ST}_a.out 2>&1 &
  ( ev q3TP_Stu_flat-s${ST}b-16k 1 8221 custom/beyondaime ) > $H/ev_q3flat${ST}_b.out 2>&1 &
  ( ev q3TP_Stu_flat-s${ST}c-16k 2 8222 aime/aime_2024_I aime/aime_2024_II aime/aime_2025 hmmt/hmmt_feb_2025 ) > $H/ev_q3flat${ST}_c.out 2>&1 &
  wait
done
echo "DRIVER05S_DONE $(date -Is)"
