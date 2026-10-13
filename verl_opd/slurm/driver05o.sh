#!/bin/bash
# tpJustRL_shapeStu: JustRL's preferences + the student's OWN entropy everywhere
# (zero shape gap).  200 steps, eval at s50 and s200.
set -u
source /home/sya225/opd/env.sh none
S=$OPD_HOME/scripts; L=$SCRATCH/logs; H=/home/sya225/logs
TAG=ds15b_tpJustRL_shapeStu
JR=hbx/JustRL-DeepSeek-1.5B
BASE=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

echo "DRIVER05O_START $(date -Is)"
SAVE_FREQ=50 bash $S/chain05_tp.sh $TAG student $JR - 3,4,5 /tmp/ray_jrstu 1.0 200
echo "TRAIN_RC=$? $(date -Is)"

export MAX_TOKENS=16384 CONCURRENCY=100 GPU_UTIL=0.9 MATH_ONLY=1
for ST in 200 50; do
  CK=$SCRATCH/checkpoints/opd_$TAG/global_step_$ST/actor
  HF=$SCRATCH/checkpoints/hf_${TAG}_step$ST
  [ -d $CK ] || { echo "NO CHECKPOINT at $CK — skipping s$ST"; continue; }
  if [ ! -s $HF/config.json ]; then
    CUDA_VISIBLE_DEVICES=3 "$VERL_VENV/bin/python" -m verl.model_merger merge \
        --backend fsdp --local_dir $CK --target_dir $HF 2>&1 | tail -1
    "$CODE_EVAL_ENV/bin/python" -c "from transformers import AutoTokenizer as T; T.from_pretrained('$BASE').save_pretrained('$HF')"
  fi
  echo "MERGE_DONE s$ST $(date -Is)"
  ev () { local suf=$1 g=$2 p=$3; shift 3
    COMPS="$*" bash $S/teacher_eval.sh dsTP_JustRL_shapeStu-s${ST}${suf}-16k $HF $g $p $BASE \
        > $L/teval_jrstu${ST}${suf}.log 2>&1; echo "EVAL_DONE s${ST}${suf} rc=$? $(date -Is)"; }
  ( ev ''  3 8090 custom/amc hmmt/hmmt_feb_2026 ) > $H/ev_jrstu${ST}_a.out 2>&1 &
  ( ev b   4 8091 custom/beyondaime ) > $H/ev_jrstu${ST}_b.out 2>&1 &
  ( ev c   5 8092 aime/aime_2024_I aime/aime_2024_II aime/aime_2025 hmmt/hmmt_feb_2025 ) > $H/ev_jrstu${ST}_c.out 2>&1 &
  wait
done
echo "DRIVER05O_DONE $(date -Is)"
