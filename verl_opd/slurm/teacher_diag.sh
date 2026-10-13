L=/localscratch/sya225/logs/train_q3i17_from8B_dapo_16k.log; L2=/localscratch/sya225/logs/train_ds15b_fromDS7B_dapo_16k.log
for f in $L $L2; do echo "== $(basename $f): teacher args with values"; grep -a "vLLMHttpServer" $f | grep -a -A1 -E "'--(logprobs_mode|override_generation_config|max_logprobs|dtype|hf_overrides|max_model_len|gpu_memory_utilization|seed)'" | grep -av "^--$" | sed -E 's/^\(vLLMHttpServer pid=[0-9]+\) *//' | head -n 40 | tr "\n" " " | cut -c1-1200; echo; done
echo "== teacher server on :41019"
curl -s -m 10 http://localhost:41019/v1/models | head -c 400; echo
curl -s -m 60 http://localhost:41019/v1/completions -H 'Content-Type: application/json' -d '{"model":"Qwen/Qwen3-8B","prompt":"The capital of France is Paris. 2+2=4.","max_tokens":1,"prompt_logprobs":3,"temperature":1.0}' | head -c 1500; echo
