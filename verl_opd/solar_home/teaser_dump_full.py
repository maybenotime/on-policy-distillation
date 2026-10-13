"""Dump full-rollout per-token entropies + top-16 of both models (teacher-forced on the student's tokens) for given seqs.
usage: python teaser_dump_full.py <rollouts.jsonl> <out.json> <seq> [<seq> ...]"""
import json, sys, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
ROLL, OUT = sys.argv[1], sys.argv[2]; SEQS = [int(s) for s in sys.argv[3:]]
STU = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"; TEA = "hbx/JustRL-DeepSeek-1.5B"
recs = [json.loads(l) for l in open(ROLL)]; tok = AutoTokenizer.from_pretrained(STU)
def disp(i):
    s = tok.decode([i]); s = s.replace("\n", "\\n"); return s.strip() if s.strip() else ("␣" if s == " " else s)
dev = "cuda"
ms = AutoModelForCausalLM.from_pretrained(STU, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
mt = AutoModelForCausalLM.from_pretrained(TEA, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
out = {}
with torch.inference_mode():
    for c in SEQS:
        r = recs[c]; p, sids = r["prompt_ids"], r["resp_ids"]; n = min(len(sids), 1500)
        x = torch.tensor([p + sids[:n]], device=dev); res = {}
        for name, m in (("S", ms), ("T", mt)):
            lp = torch.log_softmax(m(x, use_cache=False).logits[0][len(p) - 1:len(p) + n - 1].float(), -1)
            H = (-(lp.exp() * lp).sum(-1)).cpu().numpy(); tk = torch.topk(lp, 16, dim=-1)
            res[name] = (H, tk.indices.cpu().numpy(), tk.values.exp().cpu().numpy())
        Hs, Ht = res["S"][0], res["T"][0]
        out[str(c)] = dict(prompt_idx=r["idx"], n=int(n), toks=[disp(i) for i in sids[:n]], Hs=Hs.tolist(), Ht=Ht.tolist(),
            rho_all=float(np.corrcoef(Hs, Ht)[0, 1]),
            ov16_all=float(np.mean([len(set(res["S"][1][k]) & set(res["T"][1][k])) / 16 for k in range(n)])),
            topk={str(k): dict(S=[(disp(int(i)), float(v)) for i, v in zip(res["S"][1][k], res["S"][2][k])],
                               T=[(disp(int(i)), float(v)) for i, v in zip(res["T"][1][k], res["T"][2][k])]) for k in range(n)})
        print("done", c, n, flush=True)
json.dump(out, open(OUT, "w")); print("WROTE", OUT)
