#!/bin/bash
# Replaces the eval half of driver05u for q3i17_tpM500_flat s100: the node expires 00:37, so split the 7 comps over
# all 6 GPUs (the driver05u subshell was killed; its training child chain19_tp.sh keeps running to step 100).
set -u
source /home/sya225/opd/env.sh none
S=$OPD_HOME/scripts; L=$SCRATCH/logs; H=/home/sya225/logs
STU=Qwen/Qwen3-1.7B; TAG=q3i17_tpM500_flat; ST=100
TL=$L/train_$TAG.log
echo "DRIVER05W_START $(date -Is)"
LAT=$SCRATCH/checkpoints/opd_$TAG/latest_checkpointed_iteration.txt
until [ "$(cat $LAT 2>/dev/null)" = "$ST" ]; do sleep 30; done; sleep 90
echo "TRAIN_ENDED $(date -Is) step=$(grep -o 'training/global_step:[0-9]*' $TL | tail -1)"
# wait for the trainer/vLLM to release GPUs 0-2
for i in $(seq 1 80); do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0,1,2 | sort -n | tail -1); [ "$m" -lt 3000 ] && break; sleep 15
done
CK=$SCRATCH/checkpoints/opd_$TAG/global_step_$ST/actor; HF=$SCRATCH/checkpoints/hf_${TAG}_step$ST
[ -d $CK ] || { echo "NO CHECKPOINT $CK"; exit 1; }
if [ ! -s $HF/config.json ]; then
  CUDA_VISIBLE_DEVICES=0 "$VERL_VENV/bin/python" -m verl.model_merger merge --backend fsdp --local_dir $CK --target_dir $HF > $H/merge_mf100.out 2>&1 || { echo "MERGE_FAILED"; exit 1; }
  "$CODE_EVAL_ENV/bin/python" -c "from transformers import AutoTokenizer as T; T.from_pretrained('$STU').save_pretrained('$HF')" >> $H/merge_mf100.out 2>&1
fi
echo "MERGE_DONE q3m500flat s$ST $(date -Is)"
export MAX_TOKENS=8192 CONCURRENCY=100 GPU_UTIL=0.9 MATH_ONLY=1
ev () { local T=$1 G=$2 P=$3; shift 3
  COMPS="$*" bash $S/teacher_eval.sh $T $HF $G $P $STU > $L/teval_$T.log 2>&1
  echo "EVAL_DONE $T rc=$? $(date -Is)"; }
( ev q3TP_M500_flat-s100-16k  0 8240 custom/amc )           > $H/ev_mf100_a.out 2>&1 &
( ev q3TP_M500_flat-s100b-16k 1 8241 custom/beyondaime )    > $H/ev_mf100_b.out 2>&1 &
( ev q3TP_M500_flat-s100c-16k 2 8242 aime/aime_2024_I aime/aime_2024_II ) > $H/ev_mf100_c.out 2>&1 &
( ev q3TP_M500_flat-s100d-16k 3 8243 aime/aime_2025 )       > $H/ev_mf100_d.out 2>&1 &
( ev q3TP_M500_flat-s100e-16k 4 8244 hmmt/hmmt_feb_2025 )   > $H/ev_mf100_e.out 2>&1 &
( ev q3TP_M500_flat-s100f-16k 5 8245 hmmt/hmmt_feb_2026 )   > $H/ev_mf100_f.out 2>&1 &
wait
cat $H/ev_mf100_?.out
echo "Q3M500FLAT_DONE $(date -Is)"
