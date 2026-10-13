#!/bin/bash
cd /home/USER/opd/LiveCodeBench || exit 1
export HF_HOME=/localscratch/USER/qwen36_openhands/hf_cache
LOG=/home/USER/opd/LiveCodeBench/final_eval.log
echo "=== official custom_evaluator (final) ===" > "$LOG"
.venv/bin/python -m lcb_runner.runner.custom_evaluator \
  --custom_output_file /localscratch/USER/qwen36_openhands/lcb/final_custom_outputs.json \
  --release_version release_v6 --start_date 2025-01-01 \
  --scenario codegeneration --num_process_evaluate 16 --timeout 8 >> "$LOG" 2>&1
echo "EXIT=$?" >> "$LOG"; echo FINAL_EVAL_FINISHED >> "$LOG"
