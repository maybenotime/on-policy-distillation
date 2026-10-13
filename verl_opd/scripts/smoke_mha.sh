#!/bin/bash
# Smoke test the whole MathArena path on the smallest model and the three
# competitions we added, before committing hours to the full 5-model sweep.
set -x
E=/home/USER/opd/envs/code_eval_env
R=/home/USER/opd/matharena
export HF_HOME=/localscratch/USER/hf_cache VLLM_API_KEY=dummy
export HF_DATASETS_CACHE=/localscratch/USER/hf_cache/datasets
export PATH="$E/bin:$PATH"

"$E/bin/python" -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-1.7B --served-model-name Qwen/Qwen3-1.7B \
    --port 8012 --dtype bfloat16 --gpu-memory-utilization 0.85 \
    --max-model-len 40960 --enforce-eager --api-key dummy \
    > /localscratch/USER/logs/vllm_smoke.log 2>&1 &
PID=$!
for i in $(seq 1 90); do
    curl -s -H "Authorization: Bearer dummy" http://localhost:8012/v1/models >/dev/null 2>&1 && break
    sleep 10
done
echo "server up"
cd "$R"
# aime_2024_I (official, 15 problems) + our three additions, n=1, first 2 problems each
"$R/.venv/bin/python" scripts/run.py \
    --comp aime/aime_2024_I custom/amc custom/math500 custom/beyondaime \
    --models local/qwen3_1p7b --n 1 --problems 1 2 2>&1 | tail -30
echo "SMOKE_RC=$?"
kill $PID 2>/dev/null
echo "SMOKE_MHA_DONE $(date -Is)"
