"""Stage 1 — sample N training prompts and roll the student out on-policy.

Only the student's own trajectory is used: both entropy profiles are later measured
at the *same* positions of that trajectory, which is what makes rho / ForkOverlap
comparable between the two models.
"""
import argparse, json
import numpy as np, pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data", default="/home/USER/opd/verl_data/ot3_code30k/train.parquet")
ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max_tokens", type=int, default=4096)
ap.add_argument("--gpu_mem_util", type=float, default=0.85)
ap.add_argument("--out", required=True)
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
df = pd.read_parquet(a.data)
idx = np.random.default_rng(a.seed).choice(len(df), a.n, replace=False)

prompts = []
for i in idx:
    msgs = [{"role": m["role"], "content": m["content"]} for m in df.iloc[i]["prompt"]]
    prompts.append(tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))
print(f"[fork] {len(prompts)} prompts, seed={a.seed}", flush=True)

llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=a.gpu_mem_util,
          max_model_len=min(a.max_tokens + 4096, 40960), enforce_eager=False,
          compilation_config={"cudagraph_mode": "PIECEWISE"})
# Qwen3 official NON-thinking sampling parameters, same as the benchmark runs.
outs = llm.generate(prompts, SamplingParams(temperature=0.7, top_p=0.8, top_k=20,
                                            min_p=0.0, max_tokens=a.max_tokens, n=1,
                                            seed=a.seed))

n_trunc = 0
with open(a.out, "w") as f:
    for j, o in enumerate(outs):
        c = o.outputs[0]
        n_trunc += c.finish_reason == "length"
        f.write(json.dumps({"idx": int(idx[j]),
                            "prompt_ids": list(o.prompt_token_ids),
                            "resp_ids": list(c.token_ids)}) + "\n")
lens = [len(o.outputs[0].token_ids) for o in outs]
print(f"[fork] wrote {a.out}  len p50={int(np.median(lens))} mean={np.mean(lens):.0f} "
      f"max={max(lens)}  truncated={n_trunc}/{len(outs)}", flush=True)
