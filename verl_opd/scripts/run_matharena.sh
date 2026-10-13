#!/bin/bash
# Math evals through the official MathArena harness.
#
# One model at a time: serve it with vLLM's OpenAI server, point MathArena's
# `api: custom` config at that port, run all 8 competitions, then tear the
# server down. Serial rather than parallel because we hold only 2 GPUs and a
# 38912-token budget on 806 problems needs the KV cache.
#
# Sampling is Qwen3's official NON-thinking recommendation (T=0.7, top_p=0.8,
# top_k=20, min_p=0) — deliberately NOT the generation_config.json values
# (T=0.6/top_p=0.95), which are the thinking-mode recommendation.
set -x
E=/home/USER/opd/envs/code_eval_env      # has vllm 0.27.1
R=/home/USER/opd/matharena
export HF_HOME=/localscratch/USER/hf_cache
export HF_DATASETS_CACHE=/localscratch/USER/hf_cache/datasets
export VLLM_API_KEY=dummy
export PATH="$E/bin:$PATH"

COMPS="aime/aime_2024_I aime/aime_2024_II aime/aime_2025 hmmt/hmmt_feb_2025 hmmt/hmmt_feb_2026 custom/amc custom/math500 custom/beyondaime"
N=${N:-4}

run_model () {   # $1=config name  $2=model path  $3=port  $4=tp
  local cfg=$1 path=$2 port=$3 tp=${4:-1}
  echo "######## $cfg  ($path)  port=$port  $(date -Is) ########"
  "$E/bin/python" -m vllm.entrypoints.openai.api_server \
      --model "$path" --served-model-name "$path" \
      --port "$port" --tensor-parallel-size "$tp" --dtype bfloat16 \
      --gpu-memory-utilization 0.90 --max-model-len 40960 --enforce-eager \
      --api-key dummy > /localscratch/USER/logs/vllm_${cfg}.log 2>&1 &
  local pid=$!
  for i in $(seq 1 90); do
      curl -s -H "Authorization: Bearer dummy" "http://localhost:$port/v1/models" >/dev/null 2>&1 && break
      sleep 10
  done
  curl -s -H "Authorization: Bearer dummy" "http://localhost:$port/v1/models" | head -c 200; echo

  cd "$R"
  "$R/.venv/bin/python" scripts/run.py --comp $COMPS --models "local/$cfg" --n "$N" 2>&1 | tail -25
  echo "RUN_RC=$?"
  kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 20
}

run_model qwen3_8b   Qwen/Qwen3-8B    8010 2
run_model qwen3_4b   Qwen/Qwen3-4B    8011 1
run_model qwen3_1p7b Qwen/Qwen3-1.7B  8012 1
run_model opd_4b     /localscratch/USER/checkpoints/hf_opd30k_step424 8013 1
if [ -n "${OPD17B:-}" ]; then run_model opd_1p7b "$OPD17B" 8014 1; fi
echo "MATHARENA_ALL_DONE $(date -Is)"
