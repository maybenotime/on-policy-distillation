#!/bin/bash
# One model: both code benchmarks + all eight math competitions.
#
# v4: keep ALL output. v3 piped generation through `grep '[gen]|rror'`, which
# swallowed vLLM's progress bars and throughput lines and left multi-hour runs
# unobservable. Nothing is filtered now; each model writes its own full log.
#
# v3 changes vs v2, both measured rather than assumed:
#  * PIECEWISE cudagraphs instead of enforce_eager. Only FULL capture hangs on
#    sm_120 (stuck at 14/35); PIECEWISE finishes 51/51 in 2s and throughput goes
#    3127 -> 6186 tok/s on Qwen3-1.7B. Disabling graphs wholesale cost ~2x.
#  * MathArena concurrent_requests 64 -> 256 (config side). At 64 the server sat
#    at Running=64/Waiting=0 with GPU KV cache at 26-30%: the harness was
#    starving the GPU.
#
# Sampling: Qwen3's official NON-thinking recommendation, T=0.7 top_p=0.8
# top_k=20 min_p=0. max_tokens 32768 (not 38912) because context is 40960.
set -x
CFG=$1; MODEL=$2; TAG=$3; PORT=$4; GPUS=$5; TP=${6:-1}
export CUDA_VISIBLE_DEVICES=$GPUS
E=/home/sya225/opd/envs/code_eval_env
R=/home/sya225/opd/matharena
export PATH="$E/bin:$PATH" HF_HOME=/localscratch/sya225/hf_cache VLLM_API_KEY=dummy
export PYTHONPATH=/home/sya225/opd/LiveCodeBench
OUT=/home/sya225/opd/bench/official
mkdir -p "$OUT"
SAMP="--temperature 0.7 --top_p 0.8 --top_k 20 --min_p 0.0 --n 4 --max_tokens 32768"

export HF_DATASETS_CACHE=/localscratch/sya225/hf_cache/datasets
cd /home/sya225/opd/LiveCodeBench
for BENCH in humanevalplus lcb_v6; do
  echo "######## [$TAG] CODE $BENCH $(date -Is) ########"
  "$E/bin/python" /home/sya225/opd/scripts/gen2.py --model "$MODEL" --bench "$BENCH" \
      --out_dir "$OUT" --tag "$TAG" $SAMP --tp "$TP" --gpu_mem_util 0.85 2>&1 \
      
done

export HF_DATASETS_CACHE=/localscratch/sya225/hf_cache/datasets
echo "######## [$TAG] serving :$PORT tp=$TP $(date -Is) ########"
"$E/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name "$MODEL" --port "$PORT" \
    --tensor-parallel-size "$TP" --dtype bfloat16 --gpu-memory-utilization 0.85 \
    --max-model-len 40960 --api-key dummy \
    --no-enforce-eager --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
    > /localscratch/sya225/logs/vllm_${CFG}.log 2>&1 &
PID=$!
for i in $(seq 1 150); do
    curl -s -H "Authorization: Bearer dummy" "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && break
    sleep 10
done
curl -s -H "Authorization: Bearer dummy" "http://localhost:$PORT/v1/models" | head -c 100; echo
cd "$R"
echo "######## [$TAG] MATH $(date -Is) ########"
"$R/.venv/bin/python" scripts/run.py \
    --comp aime/aime_2024_I aime/aime_2024_II aime/aime_2025 \
           hmmt/hmmt_feb_2025 hmmt/hmmt_feb_2026 \
           custom/amc custom/math500 custom/beyondaime \
    --models "local/$CFG" --n 4 2>&1 
kill $PID 2>/dev/null; wait $PID 2>/dev/null
echo "MODEL_DONE $TAG $(date -Is)"
