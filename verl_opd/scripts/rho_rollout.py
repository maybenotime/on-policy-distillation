"""Sharded student rollout over a FULL training subset (7000 prompts).

Same as reach_rollout.py but takes --offset so the 7000 prompts can be swept in
shards; the permutation is deterministic, so shards are disjoint and their union
is the whole file.
"""
import argparse, json, numpy as np, pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--family", choices=["ds", "qwen3nt", "t1"], default="ds")
ap.add_argument("--data", required=True)
ap.add_argument("--offset", type=int, default=0)
ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max_tokens", type=int, default=8192)
ap.add_argument("--gpu_mem_util", type=float, default=0.85)
ap.add_argument("--out", required=True)
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
df = pd.read_parquet(a.data)
perm = np.random.default_rng(a.seed).permutation(len(df))
idx = perm[a.offset: a.offset + a.n]
if len(idx) == 0:
    raise SystemExit(f"[rho] empty shard offset={a.offset} of {len(df)}")

prompts = []
for i in idx:
    msgs = [{"role": m["role"], "content": m["content"]} for m in df.iloc[i]["prompt"]]
    kw = {"enable_thinking": False} if a.family == "qwen3nt" else {}
    prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw))

llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=a.gpu_mem_util,
          max_model_len=a.max_tokens + 2048, seed=a.seed)
sp = {"ds":      SamplingParams(temperature=0.6, top_p=0.95, max_tokens=a.max_tokens, seed=a.seed),
      "qwen3nt": SamplingParams(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0,
                                max_tokens=a.max_tokens, seed=a.seed),
      "t1":      SamplingParams(temperature=1.0, top_p=1.0, max_tokens=a.max_tokens, seed=a.seed)}[a.family]
outs = llm.generate(prompts, sp)

n_tr = 0
with open(a.out, "w") as f:
    for j, o in enumerate(outs):
        c = o.outputs[0]; n_tr += c.finish_reason == "length"
        f.write(json.dumps({"idx": int(idx[j]), "prompt_ids": list(o.prompt_token_ids),
                            "resp_ids": list(c.token_ids)}) + "\n")
lens = [len(o.outputs[0].token_ids) for o in outs]
print(f"[rho] wrote {a.out} n={len(outs)} len p50={int(np.median(lens))} mean={np.mean(lens):.0f} "
      f"truncated={n_tr}/{len(outs)}", flush=True)
