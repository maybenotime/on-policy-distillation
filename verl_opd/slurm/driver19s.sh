#!/bin/bash
# Eval queue on cs-venus-19 GPU 3: merges + MathArena (7 comps, one GPU) for each checkpoint as it appears.
#   (mismatchW stopped at s71 by the user, not evaluated) q3i17_tpM500_shapeStuFix s100 (8k) [user: s100 only; the DS rerun moved to node 05 / driver05x]
set -u
source /home/sya225/opd/env.sh none
S=$OPD_HOME/scripts; L=$SCRATCH/logs; H=/home/sya225/logs
Q3=Qwen/Qwen3-1.7B; DS=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
G=3; PORT=8270
export CONCURRENCY=100 GPU_UTIL=0.9 MATH_ONLY=1
COMPS_ALL="custom/amc hmmt/hmmt_feb_2026 custom/beyondaime aime/aime_2024_I aime/aime_2024_II aime/aime_2025 hmmt/hmmt_feb_2025"
QUEUE="
q3i17_tpM500_shapeStuFix:100:$Q3:8192:q3TP_M500_shapeStuFix
"
echo "DRIVER19S_START $(date -Is)"
for q in $QUEUE; do
  IFS=: read TAG ST BASE MT NAME <<< "$q"
  CK=$SCRATCH/checkpoints/opd_$TAG/global_step_$ST/actor; HF=$SCRATCH/checkpoints/hf_${TAG}_step$ST
  LAT=$SCRATCH/checkpoints/opd_$TAG/latest_checkpointed_iteration.txt
  # wait until verl has finished writing this checkpoint (latest marker >= step), unless the HF copy already exists
  until [ -s $HF/config.json ] || [ "$(cat $LAT 2>/dev/null || echo 0)" -ge "$ST" ]; do sleep 60; done
  sleep 60
  if [ -d $HF ] && ! ls $HF/*.safetensors >/dev/null 2>&1; then rm -rf $HF; fi
  if [ ! -s $HF/config.json ]; then
    [ -d $CK ] || { echo "NO CHECKPOINT $CK"; continue; }
    CUDA_VISIBLE_DEVICES=$G "$VERL_VENV/bin/python" -m verl.model_merger merge --backend fsdp --local_dir $CK --target_dir $HF > $H/merge_${TAG}_$ST.out 2>&1 || { echo "MERGE_FAILED $TAG s$ST"; continue; }
    "$CODE_EVAL_ENV/bin/python" -c "from transformers import AutoTokenizer as T; T.from_pretrained('$BASE').save_pretrained('$HF')" >> $H/merge_${TAG}_$ST.out 2>&1
  fi
  echo "MERGE_DONE $TAG s$ST $(date -Is)"
  for i in $(seq 1 40); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G); [ "$m" -lt 3000 ] && break; sleep 15; done
  T=${NAME}-s${ST}-16k
  MAX_TOKENS=$MT COMPS="$COMPS_ALL" bash $S/teacher_eval.sh $T $HF $G $PORT $BASE > $L/teval_$T.log 2>&1
  echo "EVAL_DONE $T rc=$? $(date -Is)"
done
echo "DRIVER19S_DONE $(date -Is)"
