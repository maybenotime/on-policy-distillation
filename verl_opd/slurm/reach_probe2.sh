#!/bin/bash
# Reachability probe v2, N-limited (N=50 for the pipeline test, 1000 for the real run).
#   teacher rollouts on the DAPO pre-probe prompts (reuse existing 1000-files by slicing; generate otherwise, with the
#   vLLM-0.27 teardown-hang workaround from post_topk.sh), then student / teacher scoring.
# usage: bash reach_probe2.sh <N>
source /home/sya225/opd/env.sh none; PY=$CODE_EVAL_ENV/bin/python; S=$OPD_HOME/scripts; F=$OPD_HOME/fork; L=$SCRATCH/logs
export PATH="$CODE_EVAL_ENV/bin:$PATH" TOKENIZERS_PARALLELISM=false VLLM_LOGGING_LEVEL=WARNING
D=/home/sya225/opd/verl_data/dapo_math17k/train.parquet; STU=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
N=${1:-50}; O=$SCRATCH/reach; mkdir -p $O
echo "REACH_START N=$N $(date -Is)"
roll () {  # name model gpu
  local R=$F/roll_T$1-dapo8k.jsonl RN=$F/roll_T$1-dapo8k-n$N.jsonl
  if [ -s $R ] && [ "$(wc -l < $R)" -ge $N ]; then head -n $N $R > $RN; echo "[roll] $1: sliced $N from existing"; return; fi
  [ -s $RN ] && [ "$(wc -l < $RN)" -ge $N ] && { echo "[roll] $1: n$N exists"; return; }
  CUDA_VISIBLE_DEVICES=$3 $PY $S/reach_rollout.py --model $2 --family ds --data $D --n $N --max_tokens 8192 --out $RN > $L/reach_roll_$1_n$N.log 2>&1 &
  local PID=$!
  while kill -0 $PID 2>/dev/null; do sleep 20; if [ -s $RN ] && [ "$(wc -l < $RN)" -ge $N ]; then sleep 30; kill -0 $PID 2>/dev/null && { echo "[roll] $1: file complete, killing hung pid $PID"; kill -9 $PID; }; fi; done
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do c=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-120); echo "$c" | grep -qE "EngineCore" && [ "$(ps -o ppid= -p $p | tr -d ' ')" = "1" ] && { echo "kill orphan $p"; kill -9 $p; }; done
  grep "\[reach\]" $L/reach_roll_$1_n$N.log
}
score () { CUDA_VISIBLE_DEVICES=$4 $PY $S/reach_score.py --model $3 --rollouts $1 --out $O/$2_n$N.npz --label "$2 (n=$N)" 2>&1 | grep "\[score\] " ; }
( roll JustRL hbx/JustRL-DeepSeek-1.5B 0; roll DS7B deepseek-ai/DeepSeek-R1-Distill-Qwen-7B 0 ) &
( roll Sky7B Skywork/Skywork-OR1-Math-7B 1 ) &
wait; echo "ROLLOUTS_DONE $(date -Is)"
head -n $N $F/roll_ds15b-dapo.jsonl > $F/roll_ds15b-dapo-n$N.jsonl
score $F/roll_ds15b-dapo-n$N.jsonl     student_on_student $STU 0 &
score $F/roll_TJustRL-dapo8k-n$N.jsonl student_on_JustRL  $STU 1 &
wait
score $F/roll_TDS7B-dapo8k-n$N.jsonl   student_on_DS7B    $STU 0 &
score $F/roll_TSky7B-dapo8k-n$N.jsonl  student_on_Sky7B   $STU 1 &
wait
score $F/roll_TJustRL-dapo8k-n$N.jsonl JustRL_on_JustRL hbx/JustRL-DeepSeek-1.5B 0 &
score $F/roll_TDS7B-dapo8k-n$N.jsonl   DS7B_on_DS7B deepseek-ai/DeepSeek-R1-Distill-Qwen-7B 1 &
wait
score $F/roll_TSky7B-dapo8k-n$N.jsonl  Sky7B_on_Sky7B Skywork/Skywork-OR1-Math-7B 0
echo "REACH_DONE N=$N $(date -Is)"
