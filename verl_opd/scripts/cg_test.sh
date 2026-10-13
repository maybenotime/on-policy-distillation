#!/bin/bash
# Does PIECEWISE cudagraph capture survive on sm_120, and what does it buy?
# During training, FULL capture hung at 14/35 while PIECEWISE completed 51/51,
# so we disabled graphs wholesale with enforce_eager. That was blunt: this
# measures the middle ground on the standalone server path.
set -x
E=/home/USER/opd/envs/code_eval_env
export PATH="$E/bin:$PATH" HF_HOME=/localscratch/USER/hf_cache
M=Qwen/Qwen3-1.7B

bench () {  # $1=label $2=gpu $3=port  $4...=extra flags
  local label=$1 gpu=$2 port=$3; shift 3
  CUDA_VISIBLE_DEVICES=$gpu "$E/bin/python" -m vllm.entrypoints.openai.api_server \
      --model $M --served-model-name $M --port $port --dtype bfloat16 \
      --gpu-memory-utilization 0.85 --max-model-len 40960 --api-key dummy "$@" \
      > /localscratch/USER/logs/cg_${label}.log 2>&1 &
  local pid=$! t0=$(date +%s)
  for i in $(seq 1 90); do
      curl -s -H "Authorization: Bearer dummy" http://localhost:$port/v1/models >/dev/null 2>&1 && break
      sleep 5
  done
  local t1=$(date +%s)
  if ! curl -s -H "Authorization: Bearer dummy" http://localhost:$port/v1/models >/dev/null 2>&1; then
      echo "RESULT $label: SERVER NEVER CAME UP (likely capture hang)"; kill -9 $pid 2>/dev/null; return
  fi
  echo "RESULT $label: startup=$((t1-t0))s"
  # 128 concurrent short generations, measure wall time
  "$E/bin/python" - "$port" "$label" <<'PY'
import sys, time, asyncio, aiohttp, json
port, label = sys.argv[1], sys.argv[2]
PROMPT = "Compute 17*23 step by step, then give the final answer in \\boxed{}."
async def one(sess):
    async with sess.post(f"http://localhost:{port}/v1/chat/completions",
        headers={"Authorization":"Bearer dummy"},
        json={"model":"Qwen/Qwen3-1.7B","messages":[{"role":"user","content":PROMPT}],
              "max_tokens":512,"temperature":0.7,"top_p":0.8,
              "chat_template_kwargs":{"enable_thinking":False}}) as r:
        d = await r.json()
        return d["usage"]["completion_tokens"]
async def main():
    async with aiohttp.ClientSession() as s:
        t0=time.time(); res = await asyncio.gather(*[one(s) for _ in range(128)]); dt=time.time()-t0
        print(f"RESULT {label}: {sum(res)} tokens in {dt:.1f}s = {sum(res)/dt:.0f} tok/s")
asyncio.run(main())
PY
  kill -9 $pid 2>/dev/null; sleep 15
}

bench eager     0 8020 --enforce-eager
bench piecewise 0 8021 --no-enforce-eager -O.cudagraph_mode=PIECEWISE
echo "CG_TEST_DONE $(date -Is)"
