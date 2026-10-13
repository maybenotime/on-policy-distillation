#!/bin/bash
# Teacher code evals on cs-venus-19: 14B / RLCR / Code-300 in parallel on GPUs 0,2,3;
# Code-1200 on GPU1 once the fork worker there is done. Base-model _fix rescoring first (CPU).
set -u
source /home/sya225/opd/env.sh none
L=$SCRATCH/logs; T=$OPD_HOME/scripts/teacher_code_eval.sh
for b in Qwen3-1.7B Qwen3-4B Qwen3-8B; do
  bash $T $b x -1 > $L/tce_$b.log 2>&1      # gen skipped (outputs exist) -> scoring only
done &
bash $T Qwen3-14B Qwen/Qwen3-14B 0 > $L/tce_Qwen3-14B.log 2>&1 &
bash $T RLCR4B resistz/RLCR-Calibrated-Qwen3-4B-DAPO-Math14k-3Epoch-330Step 2 > $L/tce_RLCR4B.log 2>&1 &
bash $T K4Bcode300 Keven16/Qwen3-4B-Non-Thinking-RL-Code-Step300 3 > $L/tce_K4Bcode300.log 2>&1 &
( until grep -q "fromRLCR4B\] DONE" /home/sya225/logs/fork_rlcr.log 2>/dev/null; do sleep 60; done
  bash $T K4Bcode1200 Keven16/Qwen3-4B-Non-Thinking-RL-Code-Step1200 1 > $L/tce_K4Bcode1200.log 2>&1 ) &
wait
echo "TEACHERS19_DONE $(date -Is)"
