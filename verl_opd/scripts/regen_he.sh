#!/bin/bash
# Regenerate HumanEval+ for ALL models with the fixed extractor so every number
# comes from the same rule. Two GPU streams; the 4B-teacher training owns 0,1,2.
E=/home/sya225/opd/envs/code_eval_env
export PATH="$E/bin:$PATH" HF_HOME=/localscratch/sya225/hf_cache VLLM_LOGGING_LEVEL=WARNING
export PYTHONPATH=/home/sya225/opd/LiveCodeBench
export HF_DATASETS_CACHE=/localscratch/sya225/hf_cache/datasets
O=/home/sya225/opd/bench/official
C=/localscratch/sya225/checkpoints
SAMP="--temperature 0.7 --top_p 0.8 --top_k 20 --min_p 0.0 --n 4 --max_tokens 8192"
cd /home/sya225/opd/LiveCodeBench

gen () {  # tag model gpu
  echo "######## [$1] $(date -Is) ########"
  CUDA_VISIBLE_DEVICES=$3 "$E/bin/python" /home/sya225/opd/scripts/gen2.py \
      --model "$2" --bench humanevalplus --out_dir "$O" --tag "$1" \
      $SAMP --tp 1 --gpu_mem_util 0.85 2>&1 | grep -aE "\[gen\]|Error|Traceback"
}

( gen Qwen3-8B          Qwen/Qwen3-8B                 3
  gen Qwen3-1.7B        Qwen/Qwen3-1.7B               3
  gen opd-1.7b          $C/hf_opd17b_step424          3 ) &
( gen Qwen3-4B          Qwen/Qwen3-4B                 4
  gen opd-4b            $C/hf_opd30k_step424          4
  gen opd-1.7b-from14b  $C/hf_opd17b_from14b_step424  4 ) &
wait
echo "REGEN_DONE $(date -Is)"
