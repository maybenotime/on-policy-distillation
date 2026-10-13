#!/bin/bash
# Uniform-sharpening control for the DeepSeek pair: the student's OWN logits, flattened
# by ONE global temperature T*=1.2869 so the mean entropy matches JustRL's.  Its only
# difference from tpStu_shapeJustRL is per-position vs single T, which isolates the shape.
set -u
source /home/sya225/opd/env.sh none
S=$OPD_HOME/scripts; L=$SCRATCH/logs; H=/home/sya225/logs
TAG=ds15b_tpStu_uniform
BASE=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
T=$(cat /home/sya225/slurm/tp_uniform_T_ds_stu.txt)

echo "DRIVER19M_START T*=$T $(date -Is)"
SAVE_FREQ=50 bash $S/chain05_tp.sh $TAG uniform $BASE - 0,1,2 /tmp/ray_uni $T 200
echo "TRAIN_RC=$? $(date -Is)"

export MAX_TOKENS=16384 CONCURRENCY=100 GPU_UTIL=0.9 MATH_ONLY=1
for ST in 200 50; do
  CK=$SCRATCH/checkpoints/opd_$TAG/global_step_$ST/actor
  HF=$SCRATCH/checkpoints/hf_${TAG}_step$ST
  [ -d $CK ] || { echo "NO CHECKPOINT at $CK — skipping s$ST"; continue; }
  if [ ! -s $HF/config.json ]; then
    CUDA_VISIBLE_DEVICES=0 "$VERL_VENV/bin/python" -m verl.model_merger merge \
        --backend fsdp --local_dir $CK --target_dir $HF 2>&1 | tail -1
    "$CODE_EVAL_ENV/bin/python" -c "from transformers import AutoTokenizer as T; T.from_pretrained('$BASE').save_pretrained('$HF')"
  fi
  echo "MERGE_DONE s$ST $(date -Is)"
  ev () { local suf=$1 g=$2 p=$3; shift 3
    COMPS="$*" bash $S/teacher_eval.sh dsTP_Stu_uniform-s${ST}${suf}-16k $HF $g $p $BASE \
        > $L/teval_uni${ST}${suf}.log 2>&1; echo "EVAL_DONE s${ST}${suf} rc=$? $(date -Is)"; }
  ( ev ''  0 8100 custom/amc hmmt/hmmt_feb_2026 ) > $H/ev_uni${ST}_a.out 2>&1 &
  ( ev b   1 8101 custom/beyondaime ) > $H/ev_uni${ST}_b.out 2>&1 &
  ( ev c   2 8102 aime/aime_2024_I aime/aime_2024_II aime/aime_2025 hmmt/hmmt_feb_2025 ) > $H/ev_uni${ST}_c.out 2>&1 &
  wait
done
echo "DRIVER19M_DONE $(date -Is)"
