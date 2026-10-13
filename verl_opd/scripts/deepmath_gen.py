"""One rollout per problem with a DeepSeek-R1-Distill-family model, recommended sampling
(T=0.6, top_p=0.95, max 32768 new tokens, no system prompt, boxed instruction appended).
Sharded and chunked; resumable (skips src_idx already present in --out).
usage: deepmath_gen.py --model M --data sample.parquet --shard 0 --nshards 2 --out gen_M_shard0.jsonl
"""
import argparse, json, os, time
import numpy as np, pandas as pd
from vllm import LLM, SamplingParams

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--data", required=True)
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--out", required=True)
ap.add_argument("--max_tokens", type=int, default=32768); ap.add_argument("--chunk", type=int, default=1500)
ap.add_argument("--gpu_mem_util", type=float, default=0.90); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--stop_file", default="")
ap.add_argument("--done_from", default="", help="glob of sibling output files whose src_idx also count as done")
ap.add_argument("--family", choices=["ds", "qwen3nt"], default="ds",
                help="ds: DeepSeek-R1-Distill recommended (T0.6/top_p0.95); qwen3nt: Qwen3 non-thinking official (T0.7/top_p0.8/top_k20, enable_thinking=False)")
a = ap.parse_args()

SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
df = pd.read_parquet(a.data)
df = df.iloc[a.shard::a.nshards]
if a.limit: df = df.iloc[:a.limit]
import glob
done = set()
for fn in ([a.out] if os.path.exists(a.out) else []) + (glob.glob(a.done_from) if a.done_from else []):
    with open(fn) as f:
        for line in f:
            try: done.add(json.loads(line)["src_idx"])
            except Exception: pass
df = df[~df.src_idx.isin(done)]
print(f"[gen] model={a.model} shard {a.shard}/{a.nshards}: {len(df)} to do ({len(done)} already done)", flush=True)
if len(df) == 0: raise SystemExit

llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=a.gpu_mem_util,
          max_model_len=a.max_tokens + 2048, enable_prefix_caching=True, seed=a.seed)
if a.family == "ds":
    sp = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=a.max_tokens, n=1, seed=a.seed)
    chat_kw = {}
else:
    sp = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, max_tokens=a.max_tokens, n=1, seed=a.seed)
    chat_kw = {"chat_template_kwargs": {"enable_thinking": False}}
print(f"[gen] family={a.family} sampling={sp}", flush=True)

rows = df.to_dict("records")
t0 = time.time(); ntok = 0; ntr = 0
for c0 in range(0, len(rows), a.chunk):
    if a.stop_file and os.path.exists(a.stop_file):
        print(f"[gen] stop file {a.stop_file} found, exiting after {c0} rows", flush=True); break
    chunk = rows[c0:c0 + a.chunk]
    msgs = [[{"role": "user", "content": r["question"] + SUFFIX}] for r in chunk]
    outs = llm.chat(msgs, sp, use_tqdm=False, **chat_kw)
    with open(a.out, "a") as f:
        for r, o in zip(chunk, outs):
            c = o.outputs[0]; ntok += len(c.token_ids); ntr += c.finish_reason == "length"
            f.write(json.dumps({"src_idx": int(r["src_idx"]), "response": c.text,
                                "finish_reason": c.finish_reason, "ntok": len(c.token_ids)}) + "\n")
    el = time.time() - t0
    print(f"[gen] {c0 + len(chunk)}/{len(rows)}  {ntok/el:.0f} tok/s  elapsed {el/3600:.2f} h  "
          f"truncated so far: {ntr}", flush=True)
print(f"[gen] DONE shard {a.shard} {time.strftime('%FT%T')}", flush=True)
