#!/bin/bash
# Resume ds15b_tpJustRL_shapeStu from its (complete) s150 checkpoint to 200, then evaluate s200.
# Runs on GPUs 3,4,5; 0,1,2 are busy with q3i17_tpM500_flat.
set -u
source /home/sya225/opd/env.sh none
S=$OPD_HOME/scripts; L=$SCRATCH/logs; H=/home/sya225/logs
TAG=ds15b_tpJustRL_shapeStu
JR=hbx/JustRL-DeepSeek-1.5B
BASE=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

echo "DRIVER05V_START $(date -Is) resuming from $(cat $SCRATCH/checkpoints/opd_$TAG/latest_checkpointed_iteration.txt 2>/dev/null)"
SAVE_FREQ=50 bash $S/chain05_tp.sh $TAG student $JR - 3,4,5 /tmp/ray_jrstu4 1.0 200
echo "TRAIN_RC_JRSTU200=$? $(date -Is)"

ST=200
CK=$SCRATCH/checkpoints/opd_$TAG/global_step_$ST/actor
HF=$SCRATCH/checkpoints/hf_${TAG}_step$ST
[ -d $CK ] || { echo "NO CHECKPOINT $CK"; exit 1; }
if [ ! -s $HF/config.json ]; then
  CUDA_VISIBLE_DEVICES=3 "$VERL_VENV/bin/python" -m verl.model_merger merge \
      --backend fsdp --local_dir $CK --target_dir $HF 2>&1 | tail -1
  "$CODE_EVAL_ENV/bin/python" -c "from transformers import AutoTokenizer as T; T.from_pretrained('$BASE').save_pretrained('$HF')"
fi
echo "MERGE_DONE jrstu s200 $(date -Is)  (free: $(df -h /localscratch | tail -1 | awk '{print $4}'))"
export MAX_TOKENS=16384 CONCURRENCY=100 GPU_UTIL=0.9 MATH_ONLY=1
ev () { local T=$1 G=$2 P=$3; shift 3
  COMPS="$*" bash $S/teacher_eval.sh $T $HF $G $P $BASE > $L/teval_$T.log 2>&1
  echo "EVAL_DONE $T rc=$? $(date -Is)"; }
( ev dsTP_JustRL_shapeStu-s200-16k  3 8250 custom/amc hmmt/hmmt_feb_2026 ) > $H/ev_jr200_a.out 2>&1 &
( ev dsTP_JustRL_shapeStu-s200b-16k 4 8251 custom/beyondaime ) > $H/ev_jr200_b.out 2>&1 &
( ev dsTP_JustRL_shapeStu-s200c-16k 5 8252 aime/aime_2024_I aime/aime_2024_II aime/aime_2025 hmmt/hmmt_feb_2025 ) > $H/ev_jr200_c.out 2>&1 &
wait
echo "DRIVER05V_DONE $(date -Is)"
