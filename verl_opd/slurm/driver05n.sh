#!/bin/bash
# tpStu_shapeJustRL: resume from step 50 to 200 (verl resume_mode=auto picks up
# latest_checkpointed_iteration.txt), then eval s200.  s50 is already scored (28.2).
set -u
source /home/sya225/opd/env.sh none
S=$OPD_HOME/scripts; L=$SCRATCH/logs; H=/home/sya225/logs
TAG=ds15b_tpStu_shapeJustRL
JR=hbx/JustRL-DeepSeek-1.5B
BASE=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

echo "DRIVER05N_START $(date -Is)"
SAVE_FREQ=50 bash $S/chain05_tp.sh $TAG shape $BASE $JR 0,1,2 /tmp/ray_stu200 1.0 200
echo "TRAIN_RC=$? $(date -Is)"

ST=200
CK=$SCRATCH/checkpoints/opd_$TAG/global_step_$ST/actor
HF=$SCRATCH/checkpoints/hf_${TAG}_step$ST
if [ ! -s $HF/config.json ]; then
  [ -d $CK ] || { echo "NO CHECKPOINT at $CK"; exit 1; }
  CUDA_VISIBLE_DEVICES=0 "$VERL_VENV/bin/python" -m verl.model_merger merge \
      --backend fsdp --local_dir $CK --target_dir $HF 2>&1 | tail -1
  "$CODE_EVAL_ENV/bin/python" -c "from transformers import AutoTokenizer as T; T.from_pretrained('$BASE').save_pretrained('$HF')"
fi
echo "MERGE_DONE s$ST $(date -Is)"

export MAX_TOKENS=16384 CONCURRENCY=100 GPU_UTIL=0.9 MATH_ONLY=1
ev () { local suf=$1 g=$2 p=$3; shift 3
  COMPS="$*" bash $S/teacher_eval.sh dsTP_Stu_shapeJustRL-s${ST}${suf}-16k $HF $g $p $BASE \
      > $L/teval_stu${ST}${suf}.log 2>&1; echo "EVAL_DONE s${ST}${suf} rc=$? $(date -Is)"; }
( ev ''  0 8080 custom/amc hmmt/hmmt_feb_2026 ) > $H/ev_stu200_a.out 2>&1 &
( ev b   1 8081 custom/beyondaime ) > $H/ev_stu200_b.out 2>&1 &
( ev c   2 8082 aime/aime_2024_I aime/aime_2024_II aime/aime_2025 hmmt/hmmt_feb_2025 ) > $H/ev_stu200_c.out 2>&1 &
wait
echo "DRIVER05N_DONE $(date -Is)"
