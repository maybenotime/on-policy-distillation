#!/bin/bash
# q3i17_tpStu_shapeM500: the missing "student preferences" row of the Qwen3 grid.
# A = Qwen/Qwen3-1.7B (the student's own base, so NO preference information) and the target entropy is
# Math-Step500's per-position entropy.  Qwen3 analogue of the DeepSeek tpStu_shapeJustRL run (+2.4 mean-6).
set -u
source /home/sya225/opd/env.sh none
S=$OPD_HOME/scripts; L=$SCRATCH/logs; H=/home/sya225/logs
TAG=q3i17_tpStu_shapeM500
STU=Qwen/Qwen3-1.7B
M5=Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500

echo "DRIVER05Q_START $(date -Is)"
SAVE_FREQ=50 bash $S/chain19_tp.sh $TAG shape $STU $M5 0,1,2 /tmp/ray_q3stu 1.0 200
echo "TRAIN_RC_Q3STU=$? $(date -Is)"

export MAX_TOKENS=8192 CONCURRENCY=100 GPU_UTIL=0.9 MATH_ONLY=1
ev () { local T=$1 G=$2 P=$3; shift 3
  COMPS="$*" bash $S/teacher_eval.sh $T $HF $G $P $STU > $L/teval_$T.log 2>&1
  echo "EVAL_DONE $T rc=$? $(date -Is)"; }
for ST in 200 50; do
  CK=$SCRATCH/checkpoints/opd_$TAG/global_step_$ST/actor
  HF=$SCRATCH/checkpoints/hf_${TAG}_step$ST
  [ -d $CK ] || { echo "NO CHECKPOINT $CK — skipping s$ST"; continue; }
  if [ ! -s $HF/config.json ]; then
    CUDA_VISIBLE_DEVICES=0 "$VERL_VENV/bin/python" -m verl.model_merger merge \
        --backend fsdp --local_dir $CK --target_dir $HF 2>&1 | tail -1
    "$CODE_EVAL_ENV/bin/python" -c "from transformers import AutoTokenizer as T; T.from_pretrained('$STU').save_pretrained('$HF')"
  fi
  echo "MERGE_DONE q3stu s$ST $(date -Is)"
  ( ev q3TP_Stu_shapeM500-s${ST}-16k  0 8160 custom/amc hmmt/hmmt_feb_2026 ) > $H/ev_q3stu${ST}_a.out 2>&1 &
  ( ev q3TP_Stu_shapeM500-s${ST}b-16k 1 8161 custom/beyondaime ) > $H/ev_q3stu${ST}_b.out 2>&1 &
  ( ev q3TP_Stu_shapeM500-s${ST}c-16k 2 8162 aime/aime_2024_I aime/aime_2024_II aime/aime_2025 hmmt/hmmt_feb_2025 ) > $H/ev_q3stu${ST}_c.out 2>&1 &
  wait
done
echo "DRIVER05Q_DONE $(date -Is)"
