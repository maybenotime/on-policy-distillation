"""Teacher rolls out on the SAME 1000 DAPO prompts as the student pre-probe (same seed/idx as fork_rollout.py),
with its own recommended sampling.  Output = fork_rollout format (prompt_ids / resp_ids) so scorers can reuse it.
usage: reach_rollout.py --model M --family ds|qwen3nt --data train.parquet --n 1000 --seed 0 --max_tokens 8192 --out roll.jsonl
"""
import argparse, json, numpy as np, pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--family", choices=["ds", "qwen3nt", "t1", "q3off"], default="ds")
ap.add_argument("--data", required=True); ap.add_argument("--n", type=int, default=1000); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max_tokens", type=int, default=8192); ap.add_argument("--gpu_mem_util", type=float, default=0.85)
ap.add_argument("--out", required=True)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
df = pd.read_parquet(a.data)
idx = np.random.default_rng(a.seed).choice(len(df), a.n, replace=False)     # identical to fork_rollout.py
prompts = []
for i in idx:
    msgs = [{"role": m["role"], "content": m["content"]} for m in df.iloc[i]["prompt"]]
    kw = {"enable_thinking": False} if a.family == "qwen3nt" else {}
    prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw))
llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=a.gpu_mem_util, max_model_len=a.max_tokens + 2048, seed=a.seed)
sp = {"ds": SamplingParams(temperature=0.6, top_p=0.95, max_tokens=a.max_tokens, seed=a.seed),
      "qwen3nt": SamplingParams(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, max_tokens=a.max_tokens, seed=a.seed),
      "t1": SamplingParams(temperature=1.0, top_p=1.0, max_tokens=a.max_tokens, seed=a.seed),
      "q3off": SamplingParams(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, max_tokens=a.max_tokens, seed=a.seed)}[a.family]
outs = llm.generate(prompts, sp)
n_tr = 0
with open(a.out, "w") as f:
    for j, o in enumerate(outs):
        c = o.outputs[0]; n_tr += c.finish_reason == "length"
        f.write(json.dumps({"idx": int(idx[j]), "prompt_ids": list(o.prompt_token_ids), "resp_ids": list(c.token_ids)}) + "\n")
lens = [len(o.outputs[0].token_ids) for o in outs]
print(f"[reach] wrote {a.out} len p50={int(np.median(lens))} mean={np.mean(lens):.0f} truncated={n_tr}/{len(outs)}", flush=True)
