"""Own-rollout length of a teacher on ot3_code30k prompts (non-thinking, T=1).
Also flags degenerate loops: fraction of the last 2k tokens covered by a repeated
32-gram, and whether a finished response ends with a code block / final answer."""
import sys, json, re, collections, pandas as pd, numpy as np
from vllm import LLM, SamplingParams
model, n, max_tokens = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
out_path = sys.argv[4] if len(sys.argv) > 4 else None
df = pd.read_parquet("/home/sya225/opd/verl_data/ot3_code30k/test.parquet").head(n)
llm = LLM(model=model, dtype="bfloat16", max_model_len=1024 + max_tokens, gpu_memory_utilization=0.85, enforce_eager=True)
tok = llm.get_tokenizer()
prompts = [tok.apply_chat_template(list(p), tokenize=False, add_generation_prompt=True, enable_thinking=False) for p in df["prompt"]]
outs = llm.generate(prompts, SamplingParams(temperature=1.0, top_p=1.0, max_tokens=max_tokens))

def rep_frac(ids, k=32, tail=2048):
    t = list(ids[-tail:])
    if len(t) < 2 * k: return 0.0
    c = collections.Counter(tuple(t[i:i+k]) for i in range(len(t) - k + 1))
    top = c.most_common(1)[0][1]
    return top * k / len(t) if top > 1 else 0.0

recs = []
for o, p in zip(outs, df["prompt"]):
    ids, txt = o.outputs[0].token_ids, o.outputs[0].text
    recs.append(dict(len=len(ids), finish=o.outputs[0].finish_reason, rep=rep_frac(ids),
                     n_code_blocks=txt.count("```") // 2, text=txt))
L = np.array([r["len"] for r in recs]); fin = np.array([r["finish"] for r in recs]); rep = np.array([r["rep"] for r in recs])
clip = fin == "length"
print(f"RESULT {model} max_tokens={max_tokens}: n={n} mean_len={L.mean():.0f} median={np.median(L):.0f} "
      f"clip={clip.mean():.3f} | clipped: rep>0.5={ (rep[clip]>0.5).mean() if clip.any() else 0:.2f} "
      f"| finished: mean_len={L[~clip].mean() if (~clip).any() else 0:.0f} code_blocks={np.mean([r['n_code_blocks'] for r,c in zip(recs,clip) if not c]) if (~clip).any() else 0:.1f}")
if out_path:
    with open(out_path, "w") as f:
        for r in recs: f.write(json.dumps(r) + "\n")
