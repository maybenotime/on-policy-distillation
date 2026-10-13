#!/bin/bash
# Stop flat once its step-100 checkpoint is on disk.  driver19p is already watching for the driver's
# TRAIN_RC_FLAT line, which the chain script prints when training exits, so the hand-off to the
# queued M500-prefs+4B-shape resume happens automatically after this.
set -u
source /home/sya225/opd/env.sh none
TAG=ds15b_tpStu_flat
CKDIR=$SCRATCH/checkpoints/opd_$TAG
echo "FLATSTOP_WATCH_START $(date -Is)"
for _ in $(seq 1 360); do
  if [ -d "$CKDIR/global_step_100" ] && [ "$(cat $CKDIR/latest_checkpointed_iteration.txt 2>/dev/null)" = "100" ]; then
    echo "CKPT100_PRESENT $(date -Is)"; sleep 60; break
  fi
  sleep 60
done
[ -d "$CKDIR/global_step_100" ] || { echo "CKPT100_NEVER_APPEARED $(date -Is)"; exit 1; }
echo "KILLING_FLAT_TRAINING $(date -Is)"
for p in $(pgrep -f ray_flat); do kill -9 $p 2>/dev/null; done
sleep 5
for i in 0 1 2; do for p in $(nvidia-smi -i $i --query-compute-apps=pid --format=csv,noheader); do kill -9 $p 2>/dev/null; done; done
sleep 10
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; echo
echo "FLATSTOP_DONE $(date -Is)"
