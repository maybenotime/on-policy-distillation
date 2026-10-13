#!/bin/bash
# Regenerate LCB v6 for all six models, saving raw completions so the official
# extractor and a fixed one can be compared on IDENTICAL generations.
# Training owns GPUs 0,1,2 until it finishes; use 3,4,5 (three streams).
E=/home/USER/opd/envs/code_eval_env
export PATH="$E/bin:$PATH" HF_HOME=/localscratch/USER/hf_cache VLLM_LOGGING_LEVEL=WARNING
export PYTHONPATH=/home/USER/opd/LiveCodeBench
export HF_DATASETS_CACHE=/localscratch/USER/hf_cache/datasets
O=/home/USER/opd/bench/official
C=/localscratch/USER/checkpoints
SAMP="--temperature 0.7 --top_p 0.8 --top_k 20 --min_p 0.0 --n 4 --max_tokens 8192"
cd /home/USER/opd/LiveCodeBench

gen () {  # tag model gpu
  echo "######## [$1] start $(date -Is) ########"
  CUDA_VISIBLE_DEVICES=$3 "$E/bin/python" /home/USER/opd/scripts/gen2.py \
      --model "$2" --bench lcb_v6 --out_dir "$O" --tag "$1" \
      $SAMP --tp 1 --gpu_mem_util 0.85 2>&1 | grep -aE "\[gen\]|Error|Traceback"
  echo "######## [$1] done $(date -Is) ########"
}

( gen Qwen3-8B          Qwen/Qwen3-8B                 3
  gen opd-1.7b          $C/hf_opd17b_step424          3 ) &
( gen Qwen3-4B          Qwen/Qwen3-4B                 4
  gen opd-1.7b-from14b  $C/hf_opd17b_from14b_step424  4 ) &
( gen Qwen3-1.7B        Qwen/Qwen3-1.7B               5
  gen opd-4b            $C/hf_opd30k_step424          5 ) &
wait
echo "REGEN_LCB_DONE $(date -Is)"
