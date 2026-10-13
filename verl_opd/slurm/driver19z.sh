#!/bin/bash
# cs-venus-19 job 271952 (user 9-25): (1) finish the six benchmarks for M500-direct s100/s150 and JustRL-direct s150;
# (2) Base student <- 4B-Base-GRPO (k1 run) s50/100/150/200 on all seven comps (s50/s150 already have the AIME/AMC/Beyond
# five, only HMMT-25/26 are added; s100/s200 are merged here and get all seven).  Base evals: 8k, MAX_MODEL_LEN 32768,
# tokenizer files copied from the existing step-50 HF dir so all four checkpoints share exactly the same template.
set -u
source /home/sya225/opd/env.sh none
S=$OPD_HOME/scripts; L=$SCRATCH/logs; H=/home/sya225/logs; C=$SCRATCH/checkpoints
Q3=Qwen/Qwen3-1.7B; DS=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B; QB=Qwen/Qwen3-1.7B-Base
export CONCURRENCY=100 GPU_UTIL=0.9 MATH_ONLY=1
ALL="aime/aime_2024_I aime/aime_2024_II aime/aime_2025 hmmt/hmmt_feb_2025 hmmt/hmmt_feb_2026 custom/amc custom/beyondaime"
REST="custom/beyondaime hmmt/hmmt_feb_2025 hmmt/hmmt_feb_2026"
HMMT="hmmt/hmmt_feb_2025 hmmt/hmmt_feb_2026"
BRUN=opd_q3b17_fromQ3B4Bgrpo_dapo_16k; BTOK=$C/hf_q3b17_BaseGRPO_k1_step50
merge_base () {  # step gpu
  local ST=$1 G=$2 CK=$C/$BRUN/global_step_$1/actor HF=$C/hf_q3b17_BaseGRPO_k1_step$1
  [ -d $CK ] || { echo "NO CHECKPOINT $CK"; return 1; }
  if [ ! -s $HF/config.json ]; then
    CUDA_VISIBLE_DEVICES=$G "$VERL_VENV/bin/python" -m verl.model_merger merge --backend fsdp --local_dir $CK --target_dir $HF > $H/merge_base_$ST.out 2>&1 || { echo "MERGE_FAILED $HF"; return 1; }
    cp $BTOK/tokenizer.json $BTOK/tokenizer_config.json $BTOK/chat_template.jinja $HF/ 2>/dev/null; [ -f $BTOK/vocab.json ] && cp $BTOK/vocab.json $BTOK/merges.txt $HF/
  fi
  echo "MERGE_DONE $HF $(date -Is)"
}
ev () { local T=$1 HF=$2 G=$3 P=$4 BASE=$5 MT=$6 MML=$7; shift 7
  for i in $(seq 1 40); do m=$(nvidia-smi -i $G --query-gpu=memory.used --format=csv,noheader,nounits); [ "$m" -lt 3000 ] && break; sleep 15; done
  MAX_TOKENS=$MT MAX_MODEL_LEN=$MML COMPS="$*" bash $S/teacher_eval.sh $T $HF $G $P $BASE > $L/teval_$T.log 2>&1
  echo "EVAL_DONE $T rc=$? $(date -Is)"; }
echo "DRIVER19Z_START $(date -Is)"
( ev q3i17_fromM500-s100-16k $C/hf_q3i17_fromM500_step100 0 8330 $Q3 8192 40960 $REST
  merge_base 100 0 && ev q3b17-BaseGRPO-k1-s100-8k $C/hf_q3b17_BaseGRPO_k1_step100 0 8330 $QB 8192 32768 $ALL ) > $H/z19_0.out 2>&1 &
( ev q3i17_fromM500-s150-16k $C/hf_q3i17_fromM500_step150 1 8331 $Q3 8192 40960 $REST
  merge_base 200 1 && ev q3b17-BaseGRPO-k1-s200-8k $C/hf_q3b17_BaseGRPO_k1_step200 1 8331 $QB 8192 32768 $ALL ) > $H/z19_1.out 2>&1 &
( ev ds15b_fromJustRL-s150b-16k $C/hf_ds15b_fromJustRL_step150 2 8332 $DS 16384 40960 custom/beyondaime
  ev q3b17-BaseGRPO-k1-s50-8k $BTOK 2 8332 $QB 8192 32768 $HMMT ) > $H/z19_2.out 2>&1 &
( ev q3b17-BaseGRPO-k1-s150-8k $C/hf_q3b17_BaseGRPO_k1_step150 3 8333 $QB 8192 32768 $HMMT ) > $H/z19_3.out 2>&1 &
wait
cat $H/z19_?.out
echo "DRIVER19Z_DONE $(date -Is)"
