#!/bin/bash
# Download Qwen3-1.7B and take its pre-distillation baseline on both benchmarks,
# using the exact protocol as every other number in this study:
# non-thinking, T=0.6/top_p=0.95/top_k=20, n=4, avg pass@1.
set -x
E=/home/USER/opd/envs/code_eval_env
export PATH="$E/bin:$PATH" HF_HOME=/localscratch/USER/hf_cache
export HF_DATASETS_CACHE=/localscratch/USER/hf_cache/datasets
export PYTHONPATH=/home/USER/opd/LiveCodeBench

echo "=== download Qwen3-1.7B $(date -Is) ==="
"$E/bin/python" -c "
from huggingface_hub import snapshot_download
p = snapshot_download('Qwen/Qwen3-1.7B',
        allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja'], max_workers=8)
print('OK', p)"
du -sh /localscratch/USER/hf_cache/hub/models--Qwen--Qwen3-1.7B

OUT=/home/USER/opd/bench/sampled
cd /home/USER/opd/LiveCodeBench
for BENCH in humanevalplus lcb_v6; do
  echo "######## GEN Qwen3-1.7B / $BENCH $(date -Is) ########"
  "$E/bin/python" /home/USER/opd/scripts/gen2.py --model Qwen/Qwen3-1.7B --bench "$BENCH" \
      --out_dir "$OUT" --tag Qwen3-1.7B --max_tokens 6144 \
      --temperature 0.6 --top_p 0.95 --top_k 20 --n 4 --tp 1 --gpu_mem_util 0.85 \
      2>&1 | grep -aE "\[gen\]|rror|Traceback|unrecognized" | tail -6
done

echo "==== HumanEval+ : Qwen3-1.7B ===="
rm -f "$OUT"/humanevalplus_Qwen3-1.7B_eval_results.json   # evalplus silently reuses this
"$E/bin/python" -m evalplus.evaluate --dataset humaneval \
    --samples "$OUT/humanevalplus_Qwen3-1.7B.jsonl" 2>&1 | tail -5
echo "==== LiveCodeBench v6 : Qwen3-1.7B ===="
"$E/bin/python" -m lcb_runner.runner.custom_evaluator \
    --custom_output_file "$OUT/lcb_v6_Qwen3-1.7B.json" \
    --release_version release_v6 --start_date 2025-01-01 \
    --scenario codegeneration --num_process_evaluate 16 --timeout 8 2>&1 | tail -4
echo "PREP17B_DONE $(date -Is)"
