"""Length probe: sample a model on n math prompts, report length stats, truncation, repetition, and print tails."""
import argparse, json, re, collections, pandas as pd, numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--n", type=int, default=48)
ap.add_argument("--max_tokens", type=int, default=16384); ap.add_argument("--util", type=float, default=0.3); ap.add_argument("--out", required=True)
a = ap.parse_args()
df = pd.read_parquet("/home/sya225/opd/verl_data/ot3_math30k/train.parquet").sample(n=a.n, random_state=1)
tok = AutoTokenizer.from_pretrained(a.model)
prompts = [tok.apply_chat_template([dict(m) for m in r], tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in df["prompt"]]
llm = LLM(model=a.model, gpu_memory_utilization=a.util, max_model_len=a.max_tokens + 1024, enable_prefix_caching=False)
outs = llm.generate(prompts, SamplingParams(temperature=0.7, top_p=0.8, top_k=20, max_tokens=a.max_tokens, seed=0))
L, trunc, rep, boxed = [], 0, 0, 0
with open(a.out, "w") as f:
    for p, o in zip(prompts, outs):
        t = o.outputs[0].text; n = len(o.outputs[0].token_ids); L.append(n)
        trunc += o.outputs[0].finish_reason == "length"; boxed += "\\boxed" in t
        # repetition: fraction of 50-token windows that repeat an earlier window
        ids = list(o.outputs[0].token_ids); w = [tuple(ids[i:i+50]) for i in range(0, max(1, len(ids)-50), 50)]
        r = 1 - len(set(w)) / max(1, len(w)); rep += r > 0.2
        f.write(json.dumps({"len": n, "finish": o.outputs[0].finish_reason, "rep": round(r, 3), "head": t[:300], "tail": t[-600:]}) + "\n")
L = np.array(L)
print(f"[probe] {a.model} n={a.n} max_tokens={a.max_tokens}  len p50={np.median(L):.0f} mean={L.mean():.0f} p90={np.percentile(L,90):.0f}  truncated={trunc}/{a.n}  repetitive(>20% dup windows)={rep}/{a.n}  boxed={boxed}/{a.n}", flush=True)
