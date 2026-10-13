#!/bin/bash
# JustRL teaching directly, step 100 — the midpoint that was never evaluated, so the
# DeepSeek grid lines up at s50 / s100 / s200.  Runs on node 19 where the ckpt lives.
set -u
source /home/sya225/opd/env.sh none
S=$OPD_HOME/scripts; L=$SCRATCH/logs; H=/home/sya225/logs
BASE=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
CK=$SCRATCH/checkpoints/opd_ds15b_fromJustRL_dapo_16k/global_step_100/actor
HF=$SCRATCH/checkpoints/hf_ds15b_fromJustRL_step100
if [ ! -s $HF/config.json ]; then
  [ -d $CK ] || { echo "NO CHECKPOINT $CK"; exit 1; }
  CUDA_VISIBLE_DEVICES=0 "$VERL_VENV/bin/python" -m verl.model_merger merge \
      --backend fsdp --local_dir $CK --target_dir $HF 2>&1 | tail -1
  "$CODE_EVAL_ENV/bin/python" -c "from transformers import AutoTokenizer as T; T.from_pretrained('$BASE').save_pretrained('$HF')"
fi
echo "MERGE_DONE JustRL s100 $(date -Is)"
export MAX_TOKENS=16384 CONCURRENCY=100 GPU_UTIL=0.9 MATH_ONLY=1
ev () { local T=$1 G=$2 P=$3; shift 3
  COMPS="$*" bash $S/teacher_eval.sh $T $HF $G $P $BASE > $L/teval_$T.log 2>&1
  echo "EVAL_DONE $T rc=$? $(date -Is)"; }
( ev ds15b-JustRL-s100-16k  0 8140 custom/amc hmmt/hmmt_feb_2026 ) > $H/ev_jr100_a.out 2>&1 &
( ev ds15b-JustRL-s100b-16k 1 8141 custom/beyondaime ) > $H/ev_jr100_b.out 2>&1 &
( ev ds15b-JustRL-s100c-16k 2 8142 aime/aime_2024_I aime/aime_2024_II aime/aime_2025 hmmt/hmmt_feb_2025 ) > $H/ev_jr100_c.out 2>&1 &
wait
echo "JR100_ALL_DONE $(date -Is)"
