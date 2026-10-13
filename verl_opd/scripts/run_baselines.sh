#!/bin/bash
# Baselines: Qwen3-8B (teacher) + Qwen3-4B (student), NON-THINKING,
# on HumanEval+ (164) and LiveCodeBench v6 start_date=2025-01-01 (182).
#
# Generation uses the verl_opd venv (py3.12) -- the LiveCodeBench venv is py3.11
# and its vllm dies with "array.array is not subscriptable" at engine startup.
# PYTHONPATH + cd give us lcb_runner's dataset loader; cwd matters because
# lcb_runner reads its few_shot data files relative to the working directory.
# HF_DATASETS_CACHE points at the OLD cache that already holds LCB v6.
set -x
export HF_HOME=/localscratch/USER/hf_cache
export HF_DATASETS_CACHE=/localscratch/USER/hf_cache/datasets
export PYTHONPATH=/home/USER/opd/LiveCodeBench
LCB=/home/USER/opd/LiveCodeBench
OUT=/home/USER/opd/bench/baseline
mkdir -p "$OUT"
cd "$LCB"

for MODEL in Qwen/Qwen3-8B Qwen/Qwen3-4B; do
  TAG=$(basename "$MODEL")
  for BENCH in humanevalplus lcb_v6; do
    echo "######## GEN $TAG / $BENCH  $(date -Is) ########"
    /home/USER/opd/envs/verl_opd/bin/python /home/USER/opd/scripts/gen_code_bench.py \
        --model "$MODEL" --bench "$BENCH" --out_dir "$OUT" --tag "$TAG" \
        --max_tokens 6144 --temperature 0.0 --n 1 --tp 1 --gpu_mem_util 0.85 \
        2>&1 | grep -aE "\[gen\]|Error|Traceback|assert" | tail -12
  done
done

echo "######## SCORING  $(date -Is) ########"
for TAG in Qwen3-8B Qwen3-4B; do
  echo "==== HumanEval+ : $TAG ===="
  /home/USER/opd/envs/evalplus_env/bin/python -m evalplus.evaluate \
      --dataset humaneval --samples "$OUT/humanevalplus_${TAG}.jsonl" 2>&1 | tail -10
  echo "==== LiveCodeBench v6 : $TAG ===="
  cd "$LCB"
  .venv/bin/python -m lcb_runner.runner.custom_evaluator \
      --custom_output_file "$OUT/lcb_v6_${TAG}.json" \
      --release_version release_v6 --start_date 2025-01-01 \
      --scenario codegeneration --num_process_evaluate 16 --timeout 8 2>&1 | tail -10
done
echo "BASELINE_DONE $(date -Is)"
