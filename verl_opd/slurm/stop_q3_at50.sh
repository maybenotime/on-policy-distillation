#!/bin/bash
# Stop q3i17_tpStu_shapeM500 once its step-50 checkpoint is on disk, then let driver05q's own
# eval phase take over: it tries s200 first (no checkpoint -> skipped) and then evaluates s50.
set -u
source /home/sya225/opd/env.sh none
TAG=q3i17_tpStu_shapeM500
CKDIR=$SCRATCH/checkpoints/opd_$TAG
echo "STOPWATCH_START $(date -Is)"
for _ in $(seq 1 360); do                     # up to 6 h
  if [ -d "$CKDIR/global_step_50" ] && [ "$(cat $CKDIR/latest_checkpointed_iteration.txt 2>/dev/null)" = "50" ]; then
    echo "CKPT50_PRESENT $(date -Is)"
    sleep 60                                  # let the save finish flushing
    break
  fi
  sleep 60
done
[ -d "$CKDIR/global_step_50" ] || { echo "CKPT50_NEVER_APPEARED $(date -Is)"; exit 1; }
echo "KILLING_TRAINING $(date -Is)"
for p in $(pgrep -f ray_q3stu); do kill -9 $p 2>/dev/null; done
sleep 5
for i in 0 1 2; do
  for p in $(nvidia-smi -i $i --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done
done
sleep 10
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -3
echo "STOPWATCH_DONE — driver05q will now skip s200 and evaluate s50 $(date -Is)"
