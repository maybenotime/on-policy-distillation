#!/bin/bash
# GPU 3 sidecar for chain19_ds.sh: for step 0 (base student) and every saved ckpt, merge -> 1000 fixed DAPO prompts
# rollout -> student/teacher top-k -> overlap@16/32 (same metric as the tables). Appends to fork/overlap_curve_$TAG.txt.
# Deletes each FSDP ckpt after measuring, except the newest one (kept for resume).
set -u; source /home/USER/opd/env.sh none; S=/home/USER/opd/scripts; F=$OPD_HOME/fork
TAG=ds15b_fromJustRL_dapo_16k; CK=$SCRATCH/checkpoints/opd_$TAG; GPU=3
STUDENT0=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B; TEACHER=hbx/JustRL-DeepSeek-1.5B
DATA=/home/USER/opd/verl_data/dapo_math17k/train.parquet; OUT=$F/overlap_curve_$TAG.txt; DONE=$F/.done_$TAG
mkdir -p $F; touch $DONE
measure () {  # step model_path
  local n=$1 m=$2
  grep -qx "$n" $DONE && return
  echo "== step $n  $(date -Is)"
  ROLL_DATA=$DATA bash $S/post_topk.sh ${TAG}_s$n "$m" $TEACHER $GPU curve_${TAG}_s$n 0.85 > $SCRATCH/logs/curve_${TAG}_s$n.log 2>&1
  grep -a "^post " $SCRATCH/logs/curve_${TAG}_s$n.log | sed "s/^/step=$n /" | tee -a $OUT
  grep -q "POST_TOPK_DONE" $SCRATCH/logs/curve_${TAG}_s$n.log && echo "$n" >> $DONE
}
measure 0 $STUDENT0
while true; do
  latest=$(cat $CK/latest_checkpointed_iteration.txt 2>/dev/null || echo 0)
  for d in $(ls -d $CK/global_step_* 2>/dev/null | sed 's|.*global_step_||' | sort -n); do
    [ "$d" -le "$latest" ] || continue
    hf=$SCRATCH/checkpoints/hf_${TAG}_step$d
    if [ ! -s $hf/config.json ]; then
      CUDA_VISIBLE_DEVICES=$GPU "$VERL_VENV/bin/python" -m verl.model_merger merge --backend fsdp --local_dir $CK/global_step_$d/actor --target_dir $hf > $SCRATCH/logs/merge_${TAG}_s$d.log 2>&1
      "$CODE_EVAL_ENV/bin/python" -c "from transformers import AutoTokenizer as T; T.from_pretrained('$STUDENT0').save_pretrained('$hf')"
    fi
    measure $d $hf
    if grep -qx "$d" $DONE && [ "$d" -lt "$latest" ]; then rm -rf $CK/global_step_$d; echo "removed fsdp ckpt $d"; fi
  done
  if grep -q "######## END $TAG" /home/USER/logs/ds19.log 2>/dev/null; then
    pending=$(ls -d $CK/global_step_* 2>/dev/null | sed 's|.*global_step_||' | while read d; do grep -qx "$d" $DONE || echo $d; done | wc -l)
    [ "$pending" -eq 0 ] && break
  fi
  sleep 180
done
echo "SIDECAR_DONE $TAG $(date -Is)"
