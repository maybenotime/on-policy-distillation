"""For a few shortlisted rollouts: decoded tokens for the whole response, full-vocab entropies of both models, and top-16
candidates/probs of both models at every position in [lo, hi) — so the window and the fork can be chosen offline."""
import json, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
F="/home/sya225/opd/fork"; STU="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"; TEA="hbx/JustRL-DeepSeek-1.5B"
SEL={353:(0,80),754:(60,160),728:(430,520),240:(80,170),275:(70,150),648:(590,660)}
recs=[json.loads(l) for l in open(f"{F}/roll_ds15b-dapo.jsonl")]
tok=AutoTokenizer.from_pretrained(STU)
def disp(i):
    s=tok.decode([i]); s=s.replace("\n","\\n"); return s.strip() if s.strip() else ("␣" if s==" " else s)
dev="cuda"; ms=AutoModelForCausalLM.from_pretrained(STU,dtype=torch.bfloat16,attn_implementation="sdpa").to(dev).eval()
mt=AutoModelForCausalLM.from_pretrained(TEA,dtype=torch.bfloat16,attn_implementation="sdpa").to(dev).eval()
out={}
with torch.inference_mode():
    for seq,(lo,hi) in SEL.items():
        r=recs[seq]; p,sids=r["prompt_ids"],r["resp_ids"]; n=len(sids); hi=min(hi,n)
        x=torch.tensor([p+sids],device=dev); res={}
        for name,m in (("S",ms),("T",mt)):
            lp=torch.log_softmax(m(x,use_cache=False).logits[0][len(p)-1:len(p)+n-1].float(),-1)
            H=(-(lp.exp()*lp).sum(-1)).cpu().numpy(); tk=torch.topk(lp,16,dim=-1)
            res[name]=(H,tk.indices.cpu().numpy(),tk.values.exp().cpu().numpy())
        ov16=float(np.mean([len(set(res["S"][1][j])&set(res["T"][1][j]))/16 for j in range(n)]))
        rho=float(np.corrcoef(res["S"][0],res["T"][0])[0,1])
        out[str(seq)]=dict(prompt_idx=r["idx"],n=n,rho_all=rho,ov16_all=ov16,toks=[disp(i) for i in sids],
            Hs=res["S"][0].tolist(),Ht=res["T"][0].tolist(),lo=lo,hi=hi,
            topk={str(j):dict(S=[(disp(int(i)),float(v)) for i,v in zip(res["S"][1][j],res["S"][2][j])],
                              T=[(disp(int(i)),float(v)) for i,v in zip(res["T"][1][j],res["T"][2][j])]) for j in range(lo,hi)})
        print("done",seq,r["idx"],n,round(rho,3),round(ov16,3))
json.dump(out,open("/home/sya225/teaser_full.json","w")); print("WROTE")
