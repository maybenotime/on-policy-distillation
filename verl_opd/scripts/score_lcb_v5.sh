#!/bin/bash
# LCB v6 scoring. lcb_runner is NOT installed into the venv: it is imported from
# the repo checkout, and it also reads its few-shot data relative to cwd, so both
# PYTHONPATH and cd are required (this is what the earlier attempt was missing).
E=/home/USER/opd/envs/code_eval_env
export PATH="$E/bin:$PATH" HF_HOME=/localscratch/USER/hf_cache
export PYTHONPATH=/home/USER/opd/LiveCodeBench
cd /home/USER/opd/LiveCodeBench || exit 1
O=/home/USER/opd/bench/official
rm -f "$O"/*_output_eval.json "$O"/*_codegeneration_output_eval.json
for T in Qwen3-8B Qwen3-4B Qwen3-1.7B opd-4b opd-1.7b; do
  echo "######## LCB v6 : $T ########"
  "$E/bin/python" -m lcb_runner.runner.custom_evaluator \
      --custom_output_file "$O/lcb_v6_${T}.json" \
      --release_version release_v6 --start_date 2025-01-01 \
      --scenario codegeneration --num_process_evaluate 16 --timeout 8 2>&1 | tail -8
done
echo "SCORE_LCB_DONE $(date -Is)"
