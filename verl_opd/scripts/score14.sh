#!/bin/bash
E=/home/USER/opd/envs/code_eval_env
export PATH="$E/bin:$PATH" HF_HOME=/localscratch/USER/hf_cache
export PYTHONPATH=/home/USER/opd/LiveCodeBench
O=/home/USER/opd/bench/official
T=opd-1.7b-from14b
# evalplus silently reuses its cache; LCB needs cwd + PYTHONPATH at the repo.
rm -f "$O/humanevalplus_${T}_eval_results.json" "$O/lcb_v6_${T}"*eval*.json
echo "######## HumanEval+ : $T ########"
"$E/bin/python" -m evalplus.evaluate --dataset humaneval --samples "$O/humanevalplus_${T}.jsonl" 2>&1 | tail -5
cd /home/USER/opd/LiveCodeBench
echo "######## LCB v6 : $T ########"
"$E/bin/python" -m lcb_runner.runner.custom_evaluator \
    --custom_output_file "$O/lcb_v6_${T}.json" --release_version release_v6 \
    --start_date 2025-01-01 --scenario codegeneration \
    --num_process_evaluate 32 --timeout 8 > /localscratch/USER/logs/lcb14_raw.log 2>&1
echo "SCORE14_DONE $(date -Is)"
