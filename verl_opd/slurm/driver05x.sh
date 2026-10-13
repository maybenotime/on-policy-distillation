#!/bin/bash
# cs-venus-05, job 270922 (GPUs 0,1,2 / 48 CPUs): ds15b_tpJustRL_shapeStuFix = JustRL preferences + FROZEN base-student
# (DS-R1-Distill-1.5B) entropy profile (TRANSPLANT_MODE=shape, B = base student), 100 steps, then merge + 3-way eval of s100.
# Moved here from the node-19 queue (driver19r) so the two fixed-shape reruns train in parallel.
set -u
source /home/sya225/opd/env.sh none
S=$OPD_HOME/scripts; L=$SCRATCH/logs; H=/home/sya225/logs
DS=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B; JR=hbx/JustRL-DeepSeek-1.5B
TAG=ds15b_tpJustRL_shapeStuFix; ST=100
echo "DRIVER05X_START $(date -Is)"
SAVE_FREQ=50 bash $S/chain05_tp.sh $TAG shape $JR $DS 0,1,2 /tmp/ray_dsfix05 1.0 100
echo "TRAIN_RC_DSFIX=$? $(date -Is)"
for i in $(seq 1 60); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1); [ "$m" -lt 3000 ] && break; sleep 15; done
CK=$SCRATCH/checkpoints/opd_$TAG/global_step_$ST/actor; HF=$SCRATCH/checkpoints/hf_${TAG}_step$ST
[ -d $CK ] || { echo "NO CHECKPOINT $CK"; exit 1; }
if [ ! -s $HF/config.json ]; then
  CUDA_VISIBLE_DEVICES=0 "$VERL_VENV/bin/python" -m verl.model_merger merge --backend fsdp --local_dir $CK --target_dir $HF > $H/merge_dsfix100.out 2>&1 || { echo "MERGE_FAILED"; exit 1; }
  "$CODE_EVAL_ENV/bin/python" -c "from transformers import AutoTokenizer as T; T.from_pretrained('$DS').save_pretrained('$HF')" >> $H/merge_dsfix100.out 2>&1
fi
echo "MERGE_DONE dsfix s$ST $(date -Is)"
export MAX_TOKENS=16384 CONCURRENCY=100 GPU_UTIL=0.9 MATH_ONLY=1
ev () { local T=$1 G=$2 P=$3; shift 3
  COMPS="$*" bash $S/teacher_eval.sh $T $HF $G $P $DS > $L/teval_$T.log 2>&1
  echo "EVAL_DONE $T rc=$? $(date -Is)"; }
( ev dsTP_JustRL_shapeStuFix-s100-16k  0 8250 custom/amc hmmt/hmmt_feb_2026 ) > $H/ev_dsfix100_a.out 2>&1 &
( ev dsTP_JustRL_shapeStuFix-s100b-16k 1 8251 custom/beyondaime )            > $H/ev_dsfix100_b.out 2>&1 &
( ev dsTP_JustRL_shapeStuFix-s100c-16k 2 8252 aime/aime_2024_I aime/aime_2024_II aime/aime_2025 hmmt/hmmt_feb_2025 ) > $H/ev_dsfix100_c.out 2>&1 &
wait
cat $H/ev_dsfix100_?.out
echo "DSFIX05_DONE $(date -Is)"
