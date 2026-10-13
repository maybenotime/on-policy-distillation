#!/bin/bash
# Fill the missing s50 point for ds15b_tpJustRL_shapeStu (prefs-only), using the three GPUs the
# Qwen3 run just freed.  The resume driver would only get to s50 after s200 finishes, which may
# not fit before this node's job expires.
set -u
source /home/sya225/opd/env.sh none
S=$OPD_HOME/scripts; L=$SCRATCH/logs; H=/home/sya225/logs
TAG=ds15b_tpJustRL_shapeStu; BASE=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
CK=$SCRATCH/checkpoints/opd_$TAG/global_step_50/actor
HF=$SCRATCH/checkpoints/hf_${TAG}_step50
[ -d $CK ] || { echo "NO CHECKPOINT $CK"; exit 1; }
if [ ! -s $HF/config.json ]; then
  CUDA_VISIBLE_DEVICES=0 "$VERL_VENV/bin/python" -m verl.model_merger merge \
      --backend fsdp --local_dir $CK --target_dir $HF 2>&1 | tail -1
  "$CODE_EVAL_ENV/bin/python" -c "from transformers import AutoTokenizer as T; T.from_pretrained('$BASE').save_pretrained('$HF')"
fi
echo "MERGE_DONE jrstu s50 $(date -Is)"
export MAX_TOKENS=16384 CONCURRENCY=100 GPU_UTIL=0.9 MATH_ONLY=1
ev () { local T=$1 G=$2 P=$3; shift 3
  COMPS="$*" bash $S/teacher_eval.sh $T $HF $G $P $BASE > $L/teval_$T.log 2>&1
  echo "EVAL_DONE $T rc=$? $(date -Is)"; }
( ev dsTP_JustRL_shapeStu-s50-16k  0 8210 custom/amc hmmt/hmmt_feb_2026 ) > $H/ev_jr50b_a.out 2>&1 &
( ev dsTP_JustRL_shapeStu-s50b-16k 1 8211 custom/beyondaime ) > $H/ev_jr50b_b.out 2>&1 &
( ev dsTP_JustRL_shapeStu-s50c-16k 2 8212 aime/aime_2024_I aime/aime_2024_II aime/aime_2025 hmmt/hmmt_feb_2025 ) > $H/ev_jr50b_c.out 2>&1 &
wait
echo "JRSTU50_ALL_DONE $(date -Is)"
