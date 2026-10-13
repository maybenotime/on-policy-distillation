#!/bin/bash
# Wait for the 1.7B run to exit, then merge + evaluate the final checkpoint.
# Same protocol as every other number in this study: non-thinking,
# T=0.6/top_p=0.95/top_k=20, n=4, avg pass@1.
set -x
L=/localscratch/sya225/logs/train_17b.log
while ! grep -aq TRAIN_EXIT "$L"; do sleep 120; done
grep -a TRAIN_EXIT "$L"

E=/home/sya225/opd/envs/code_eval_env
V=/home/sya225/opd/envs/verl_opd
export PATH="$E/bin:$PATH" HF_HOME=/localscratch/sya225/hf_cache
export HF_DATASETS_CACHE=/localscratch/sya225/hf_cache/datasets
export PYTHONPATH=/home/sya225/opd/LiveCodeBench

C=/localscratch/sya225/checkpoints/opd_code_17b
N=$(ls -d "$C"/global_step_* | sed "s|.*/global_step_||" | sort -n | tail -1)
S="$C/global_step_$N"; H=/localscratch/sya225/checkpoints/hf_opd17b_step${N}
echo "[eval] checkpoint: $S"
"$V/bin/python" -m verl.model_merger merge --backend fsdp --local_dir "$S/actor" --target_dir "$H" 2>&1|tail -3
"$E/bin/python" -c "
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B').save_pretrained('$H'); print('tokenizer saved')"

OUT=/home/sya225/opd/bench/sampled
T=opd17b-step${N}
cd /home/sya225/opd/LiveCodeBench
for BENCH in humanevalplus lcb_v6; do
  echo "######## GEN $T / $BENCH $(date -Is) ########"
  "$E/bin/python" /home/sya225/opd/scripts/gen2.py --model "$H" --bench "$BENCH" \
      --out_dir "$OUT" --tag "$T" --max_tokens 6144 \
      --temperature 0.6 --top_p 0.95 --top_k 20 --n 4 --tp 1 --gpu_mem_util 0.85 \
      2>&1 | grep -aE "\[gen\]|rror|Traceback" | tail -6
done
echo "==== HumanEval+ : $T ===="
rm -f "$OUT/humanevalplus_${T}_eval_results.json"   # evalplus silently reuses this
"$E/bin/python" -m evalplus.evaluate --dataset humaneval --samples "$OUT/humanevalplus_${T}.jsonl" 2>&1|tail -5
echo "==== LiveCodeBench v6 : $T ===="
"$E/bin/python" -m lcb_runner.runner.custom_evaluator --custom_output_file "$OUT/lcb_v6_${T}.json" \
    --release_version release_v6 --start_date 2025-01-01 \
    --scenario codegeneration --num_process_evaluate 16 --timeout 8 2>&1|tail -4

echo "=== archiving to /project ==="
D=/project/3dlg-hcvc/hsp/opd_code_qwen3
cp -r "$H" "$D"/ 2>&1|tail -1
cp -r "$OUT" "$D"/bench/ 2>&1|tail -1
cp "$L" "$D"/logs/ 2>/dev/null
echo "EVAL17B_DONE $(date -Is)"
