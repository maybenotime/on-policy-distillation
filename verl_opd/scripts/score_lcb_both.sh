#!/bin/bash
# Score LCB under both extractors. The three base models need scoring once: their
# two extractions are byte-identical (0 differing samples), so official == fixed.
E=/home/USER/opd/envs/code_eval_env
export PATH="$E/bin:$PATH" HF_HOME=/localscratch/USER/hf_cache
export PYTHONPATH=/home/USER/opd/LiveCodeBench
O=/home/USER/opd/bench/official
cd /home/USER/opd/LiveCodeBench
rm -f "$O"/lcb_v6_*_codegeneration_output_eval*.json "$O"/lcb_v6_*_codegeneration_output.json

score () {   # file-stem
  "$E/bin/python" -m lcb_runner.runner.custom_evaluator \
      --custom_output_file "$O/lcb_v6_$1.json" --release_version release_v6 \
      --start_date 2025-01-01 --scenario codegeneration \
      --num_process_evaluate 24 --timeout 8 > /localscratch/USER/logs/lcbs_$1.log 2>&1
  echo "scored $1 $(date -Is)"
}

( score Qwen3-8B;   score opd-4b;            score opd-4b_fix ) &
( score Qwen3-4B;   score opd-1.7b;          score opd-1.7b_fix ) &
( score Qwen3-1.7B; score opd-1.7b-from14b;  score opd-1.7b-from14b_fix ) &
wait
echo "LCB_BOTH_DONE $(date -Is)"
