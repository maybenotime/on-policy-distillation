#!/bin/bash
# tpJustRL_shapeStu: JustRL's preferences, the student's OWN entropy at every position
# (zero shape gap by construction) -> how much does the shape channel carry?
# DS analogue of the Qwen3 `tpM500_shapeStu` run (+10.4 vs M500's own +14.5).
set -u
source /home/sya225/opd/env.sh none
S=$OPD_HOME/scripts; L=$SCRATCH/logs; H=/home/sya225/logs
TAG=ds15b_tpJustRL_shapeStu
JR=hbx/JustRL-DeepSeek-1.5B
BASE=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

echo "DRIVER05M_START $(date -Is)"
SAVE_FREQ=50 bash $S/chain05_tp.sh $TAG student $JR - 0,1,2 /tmp/ray_jrstu 1.0 50
echo "TRAIN_RC=$? $(date -Is)"

CK=$SCRATCH/checkpoints/opd_$TAG/global_step_50/actor
HF=$SCRATCH/checkpoints/hf_${TAG}_step50
if [ ! -s $HF/config.json ]; then
  [ -d $CK ] || { echo "NO CHECKPOINT at $CK — aborting eval"; exit 1; }
  CUDA_VISIBLE_DEVICES=0 "$VERL_VENV/bin/python" -m verl.model_merger merge \
      --backend fsdp --local_dir $CK --target_dir $HF 2>&1 | tail -1
  "$CODE_EVAL_ENV/bin/python" -c "from transformers import AutoTokenizer as T; T.from_pretrained('$BASE').save_pretrained('$HF')"
fi
echo "MERGE_DONE $(date -Is)"

export MAX_TOKENS=16384 CONCURRENCY=100 GPU_UTIL=0.9 MATH_ONLY=1
ev () {  # suffix gpu port comps...
  local suf=$1 g=$2 p=$3; shift 3
  COMPS="$*" bash $S/teacher_eval.sh dsTP_JustRL_shapeStu-s50${suf}-16k $HF $g $p $BASE \
      > $L/teval_jrstu${suf}.log 2>&1
  echo "EVAL_DONE $suf rc=$? $(date -Is)"
}
( ev ''  0 8070 custom/amc hmmt/hmmt_feb_2026 ) > $H/ev_jrstu_a.out 2>&1 &
( ev b   1 8071 custom/beyondaime ) > $H/ev_jrstu_b.out 2>&1 &
( ev c   2 8072 aime/aime_2024_I aime/aime_2024_II aime/aime_2025 hmmt/hmmt_feb_2025 ) > $H/ev_jrstu_c.out 2>&1 &
wait
echo "DRIVER05M_DONE $(date -Is)"
