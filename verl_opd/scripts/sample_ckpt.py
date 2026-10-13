"""Sample a few DAPO train prompts from a merged HF checkpoint and report length / truncation / repetition.
usage: sample_ckpt.py <hf_dir> <train.parquet> <n> <max_tokens> <gpu_util> <out.jsonl>"""
import sys, json, zlib, pandas as pd
from vllm import LLM, SamplingParams
hf, pq, n, mt, util, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
df = pd.read_parquet(pq).sample(n=n, random_state=0)
def prompt_of(row):
    p = row["prompt"]
    if isinstance(p, str): return [{"role": "user", "content": p}]
    return [{"role": m["role"], "content": m["content"]} for m in p]
msgs = [prompt_of(r) for _, r in df.iterrows()]
llm = LLM(model=hf, gpu_memory_utilization=util, max_model_len=mt + 2048, enable_prefix_caching=False, seed=0)
tok = llm.get_tokenizer()
texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False) for m in msgs]
sp = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, max_tokens=mt, seed=0)
outs = llm.generate(texts, sp)
rows = []
for m, o in zip(msgs, outs):
    t = o.outputs[0].text; nt = len(o.outputs[0].token_ids); fr = o.outputs[0].finish_reason
    tail = t[-6000:].encode(); comp = len(zlib.compress(tail)) / max(1, len(tail))
    rows.append({"ntok": nt, "finish": fr, "tail_compress_ratio": round(comp, 3), "has_boxed": "\\boxed" in t, "prompt": m[-1]["content"][:200], "response": t})
json.dump(rows, open(out, "w"))
import statistics as st
print("n", len(rows), "mean ntok", st.mean(r["ntok"] for r in rows), "median", st.median(r["ntok"] for r in rows),
      "truncated", sum(r["finish"] == "length" for r in rows), "boxed", sum(r["has_boxed"] for r in rows))
print("tail compress ratio (low = repetitive): truncated", [r["tail_compress_ratio"] for r in rows if r["finish"] == "length"],
      "| finished", [r["tail_compress_ratio"] for r in rows if r["finish"] != "length"])
for r in [x for x in rows if x["finish"] == "length"][:3]:
    print("---- TRUNCATED sample, last 700 chars:"); print(r["response"][-700:].replace("\n", " | "))
