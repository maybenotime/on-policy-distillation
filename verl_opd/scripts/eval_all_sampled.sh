#!/bin/bash
# Re-evaluate ALL THREE models under the models' own recommended sampling.
#
# The earlier numbers used temperature=0 (greedy). Both Qwen3 models ship
# generation_config.json with do_sample=true / temperature=0.6 / top_p=0.95 /
# top_k=20, and the Qwen3 card warns greedy causes endless repetition. Measured:
# greedy truncated 13.7% of LCB v6 (25/182, every one a decoding loop repeating a
# single comment line); at T=0.6 that drops to 3.3% (6/182). So the greedy LCB
# numbers understate every model and must not be mixed with these.
#
# n=4 with a fixed seed, reported as avg pass@1 -- single-sample scores are too
# noisy once decoding is stochastic.
set -x
E=/home/sya225/opd/envs/code_eval_env
V=/home/sya225/opd/envs/verl_opd
export PATH="$E/bin:$PATH"
export HF_HOME=/localscratch/sya225/hf_cache
export HF_DATASETS_CACHE=/localscratch/sya225/hf_cache/datasets
export PYTHONPATH=/home/sya225/opd/LiveCodeBench

CKPT_ROOT=/localscratch/sya225/checkpoints/opd_code_30k
STEP_NUM=$(ls -d "$CKPT_ROOT"/global_step_* 2>/dev/null | sed "s|.*/global_step_||" | sort -n | tail -1)
STEP_DIR="$CKPT_ROOT/global_step_$STEP_NUM"
HF_OUT=/localscratch/sya225/checkpoints/hf_opd30k_step${STEP_NUM}
echo "[eval] merging $STEP_DIR -> $HF_OUT"
"$V/bin/python" -m verl.model_merger merge --backend fsdp \
    --local_dir "$STEP_DIR/actor" --target_dir "$HF_OUT" 2>&1 | tail -4
"$E/bin/python" -c "
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained('Qwen/Qwen3-4B').save_pretrained('$HF_OUT')
print('tokenizer saved')"

OUT=/home/sya225/opd/bench/sampled
mkdir -p "$OUT"
cd /home/sya225/opd/LiveCodeBench

run() {  # $1=model  $2=tag
  for BENCH in humanevalplus lcb_v6; do
    echo "######## GEN $2 / $BENCH  $(date -Is) ########"
    "$E/bin/python" /home/sya225/opd/scripts/gen2.py --model "$1" --bench "$BENCH" \
        --out_dir "$OUT" --tag "$2" --max_tokens 6144 \
        --temperature 0.6 --top_p 0.95 --top_k 20 --n 4 --tp 1 --gpu_mem_util 0.85 \
        2>&1 | grep -aE "\[gen\]|Error|Traceback" | tail -8
  done
}
run Qwen/Qwen3-8B          Qwen3-8B
run Qwen/Qwen3-4B          Qwen3-4B
run "$HF_OUT"              opd30k-step${STEP_NUM}

echo "######## SCORING $(date -Is) ########"
for TAG in Qwen3-8B Qwen3-4B opd30k-step${STEP_NUM}; do
  echo "==== HumanEval+ : $TAG ===="
  "$E/bin/python" -m evalplus.evaluate --dataset humaneval \
      --samples "$OUT/humanevalplus_${TAG}.jsonl" 2>&1 | tail -6
  echo "==== LiveCodeBench v6 : $TAG ===="
  cd /home/sya225/opd/LiveCodeBench
  "$E/bin/python" -m lcb_runner.runner.custom_evaluator \
      --custom_output_file "$OUT/lcb_v6_${TAG}.json" \
      --release_version release_v6 --start_date 2025-01-01 \
      --scenario codegeneration --num_process_evaluate 16 --timeout 8 2>&1 | tail -6
done
echo "EVAL_ALL_DONE $(date -Is)"
