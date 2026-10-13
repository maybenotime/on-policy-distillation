"""Reproduce verl's teacher request offline: vLLM prompt_logprobs=64 with logprobs_mode=<arg2>, enforce_eager, bf16."""
import sys, math
from vllm import LLM, SamplingParams
m, mode, util = sys.argv[1], sys.argv[2], float(sys.argv[3])
llm = LLM(model=m, dtype="bfloat16", enforce_eager=True, logprobs_mode=mode, max_logprobs=64, max_model_len=2048,
          gpu_memory_utilization=util, enable_prefix_caching=False, seed=42)
text = "<|im_start|>user\nCompute 17*23.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nTo compute 17 times 23, we"
out = llm.generate([text], SamplingParams(max_tokens=1, temperature=1.0, top_k=-1, top_p=1.0, prompt_logprobs=64))[0]
pl = out.prompt_logprobs
print("mode", mode, "positions", len(pl))
for pos in [3, 10, len(pl)-2]:
    d = pl[pos]
    if d is None: print(pos, None); continue
    items = sorted(d.items(), key=lambda kv: kv[1].rank)[:5]
    lps = [round(v.logprob, 3) for _, v in items]
    mass = sum(math.exp(v.logprob) for v in d.values() if v.logprob > -1e30)
    ninf = sum(1 for v in d.values() if v.logprob == float("-inf") or v.logprob < -1e30)
    print(f"pos {pos}: n={len(d)} top5 lp={lps} mass64={mass:.4f} n(-inf)={ninf}")
