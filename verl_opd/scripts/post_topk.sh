#!/bin/bash
# Post-OPD overlap for one student/teacher pair:
#   bash post_topk.sh <tag> <student_hf> <teacher> <gpu> <out_prefix> [rollout_gpu_mem_util]
# 1) 1000 rollouts from the post-OPD student (vLLM; killed by PID once the file is complete,
#    because vLLM 0.27 can hang in EngineCore teardown), 2) top-k dumps of student and teacher
#    on those rollouts, 3) overlap metrics.
set -u; source /home/sya225/opd/scripts/topk_common.sh
TAG=$1; STUDENT=$2; TEACHER=$3; GPU=$4; P=$5; MU=${6:-0.85}
R=$F/roll_$TAG.jsonl
if [ ! -s "$R" ] || [ "$(wc -l < $R)" -lt 1000 ]; then
  echo "######## [$TAG] rollout gpu=$GPU util=$MU $(date -Is) ########"
  CUDA_VISIBLE_DEVICES=$GPU VLLM_LOGGING_LEVEL=WARNING "$E/bin/python" $OPD_HOME/scripts/fork_rollout.py \
      --model "$STUDENT" --data ${ROLL_DATA:-/home/sya225/opd/verl_data/ot3_code30k/train.parquet} --n 1000 --seed 0 --max_tokens 4096 --gpu_mem_util $MU --out "$R" > $L/roll_$TAG.log 2>&1 &
  RPID=$!
  while kill -0 $RPID 2>/dev/null; do
    sleep 30
    if [ -s "$R" ] && [ "$(wc -l < $R)" -ge 1000 ]; then sleep 60; kill -0 $RPID 2>/dev/null && { echo "[$TAG] rollout file complete, killing hung pid $RPID"; kill -9 $RPID; }; fi
  done
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
    c=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-120)
    if echo "$c" | grep -qE "EngineCore" && [ "$(ps -o ppid= -p $p | tr -d ' ')" = "1" ]; then echo "kill orphan $p: $c"; kill -9 $p; fi
  done
  sleep 10
fi
echo "[$TAG] rollouts: $(wc -l < $R) lines"
dump $GPU $TAG "$STUDENT" ${P}_student
dump $GPU $TAG "$TEACHER" ${P}_teacher
met ${P}_student ${P}_teacher "post $TAG"
echo "POST_TOPK_DONE $TAG $(date -Is)"
