#!/bin/bash
# fresh 4000-prompt DS base-student rollouts (seed 1, 4k, T0.7) + the content-word fork scan
source /home/sya225/opd/env.sh none; E=$CODE_EVAL_ENV; S=$OPD_HOME/scripts; R=$SCRATCH/rho; mkdir -p $R
export PATH="$E/bin:$PATH" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_LOGGING_LEVEL=WARNING
echo "BIG_START $(date -Is)"
( setsid $E/bin/python $S/fork_rollout.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --data /home/sya225/opd/verl_data/dapo_math17k/train.parquet --n 4000 --seed 1 --max_tokens 4096 --out $R/roll_ds15b-dapo-big.jsonl > /home/sya225/logs/big_roll.out 2>&1 & pid=$!
  for _ in $(seq 1 720); do grep -q '^\[fork\] wrote' /home/sya225/logs/big_roll.out 2>/dev/null && break; kill -0 $pid 2>/dev/null || break; sleep 10; done
  sleep 5; kill -9 -$pid 2>/dev/null; kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null
  for _ in $(seq 1 30); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 3000 ] 2>/dev/null && break; sleep 5; done )
echo "ROLL_DONE $(date -Is)"; grep '^\[fork\] wrote' /home/sya225/logs/big_roll.out
$E/bin/python /home/sya225/teaser_scan_big.py $R/roll_ds15b-dapo-big.jsonl /home/sya225/teaser_forks_big.json > /home/sya225/logs/big_scan.out 2>&1
echo "SCAN_RC=$? $(date -Is)"; tail -2 /home/sya225/logs/big_scan.out
echo "BIG_DONE $(date -Is)"
