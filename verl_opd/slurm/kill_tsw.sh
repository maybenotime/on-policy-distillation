#!/bin/bash
# stop q3i17_ts_mismatchW training + its evals on cs-venus-19 (only my job runs here); patterns live in this file so
# the caller's command line never matches them
for pat in driver19s.sh q3TS_mismatchW hf_q3i17_ts_mismatchW "chain19_ts.sh q3i17_ts_mismatchW" run_opd_code.sh verl.trainer "ray::" raylet gcs_server "vllm serve" "vllm.entrypoints"; do
  pkill -f -- "$pat" 2>/dev/null
done
sleep 20
for pat in "ray::" q3TS_mismatchW hf_q3i17_ts_mismatchW verl.trainer raylet; do pkill -9 -f -- "$pat" 2>/dev/null; done
sleep 5
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "remaining: $(pgrep -f -c 'ray::|raylet|verl.trainer|vllm' )"
ps -eo pid,cmd --user $USER | grep -E 'ray::|raylet|verl|vllm|teacher_eval|driver19' | grep -v grep | head
