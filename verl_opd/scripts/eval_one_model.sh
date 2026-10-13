#!/bin/bash
# Evaluate ONE model on both code benchmarks and all eight math competitions.
# Pinned to a single GPU; five of these run concurrently, one per card.
#
# Code first (gen2.py uses vLLM's offline LLM class, which owns the GPU and then
# exits), then the math half (MathArena needs an OpenAI-compatible HTTP server,
# so we serve the model and query it). Doing them the other way round would keep
# two vLLM instances alive on one card.
#
# Sampling throughout is Qwen3's official NON-thinking recommendation:
# T=0.7, top_p=0.8, top_k=20, min_p=0. max_tokens 32768 (not the card's 38912
# competition figure) because Qwen3's context is 40960 and 38912+prompt overflows it.
set -x
CFG=$1; MODEL=$2; TAG=$3; PORT=$4; GPU=$5; TP=${6:-1}
export CUDA_VISIBLE_DEVICES=$GPU
E=/home/sya225/opd/envs/code_eval_env
R=/home/sya225/opd/matharena
export PATH="$E/bin:$PATH"
export HF_HOME=/localscratch/sya225/hf_cache
export VLLM_API_KEY=dummy
export PYTHONPATH=/home/sya225/opd/LiveCodeBench
OUT=/home/sya225/opd/bench/official
mkdir -p "$OUT"

SAMP="--temperature 0.7 --top_p 0.8 --top_k 20 --min_p 0.0 --n 4 --max_tokens 32768"

# ---------- code ----------
export HF_DATASETS_CACHE=/localscratch/sya225/hf_cache/datasets
cd /home/sya225/opd/LiveCodeBench
for BENCH in humanevalplus lcb_v6; do
  echo "######## [$TAG] CODE $BENCH $(date -Is) ########"
  "$E/bin/python" /home/sya225/opd/scripts/gen2.py --model "$MODEL" --bench "$BENCH" \
      --out_dir "$OUT" --tag "$TAG" $SAMP --tp "$TP" --gpu_mem_util 0.85 \
      2>&1 | grep -aE "\[gen\]|rror|Traceback" | tail -6
done

# ---------- math ----------
export HF_DATASETS_CACHE=/localscratch/sya225/hf_cache/datasets
echo "######## [$TAG] serving on :$PORT $(date -Is) ########"
"$E/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name "$MODEL" \
    --port "$PORT" --tensor-parallel-size "$TP" --dtype bfloat16 --gpu-memory-utilization 0.85 \
    --max-model-len 40960 --enforce-eager --api-key dummy \
    > /localscratch/sya225/logs/vllm_${CFG}.log 2>&1 &
PID=$!
for i in $(seq 1 120); do
    curl -s -H "Authorization: Bearer dummy" "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && break
    sleep 10
done
curl -s -H "Authorization: Bearer dummy" "http://localhost:$PORT/v1/models" | head -c 120; echo

cd "$R"
echo "######## [$TAG] MATH $(date -Is) ########"
"$R/.venv/bin/python" scripts/run.py \
    --comp aime/aime_2024_I aime/aime_2024_II aime/aime_2025 \
           hmmt/hmmt_feb_2025 hmmt/hmmt_feb_2026 \
           custom/amc custom/math500 custom/beyondaime \
    --models "local/$CFG" --n 4 2>&1 | tail -20
kill $PID 2>/dev/null; wait $PID 2>/dev/null
echo "MODEL_DONE $TAG $(date -Is)"
