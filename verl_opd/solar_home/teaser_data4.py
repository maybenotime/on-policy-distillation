"""Sparse-fork windows early in the rollout: teacher fork (H>1.2) where the student is committed (H<0.4, top-1>0.8), the window
has <=4 teacher forks, natural text; after the forward pass keep hits where the TEACHER's top-4 contains an explicit rethink/branch
word (Wait/But/Hmm/...) that the student does not consider."""
import json, re, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
F="/home/sya225/opd/fork"; STU="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"; TEA="hbx/JustRL-DeepSeek-1.5B"
recs=[json.loads(l) for l in open(f"{F}/roll_ds15b-dapo.jsonl")]
ES=np.load(f"{F}/ent_student_ds15b-dapo.npz",allow_pickle=True); ET=np.load(f"{F}/ent_TJustRL_ds15b-dapo.npz",allow_pickle=True)
tok=AutoTokenizer.from_pretrained(STU)
def disp(i):
    s=tok.decode([i]); s=s.replace("\n","\\n"); return s.strip() if s.strip() else ("␣" if s==" " else s)
BR={"wait","but","hmm","alternatively","however","actually","let","maybe","perhaps","so","then","okay","now","therefore","thus","well","hold","instead","or"}
NAT=re.compile(r"^[A-Za-z0-9,.:;!?'()+=\-/*]+$"); PRE,POST=10,16
pre=[]
for i,r in enumerate(recs):
    s=np.asarray(ES[str(i)],np.float32); t=np.asarray(ET[str(i)],np.float32); n=min(len(s),len(t),len(r["resp_ids"]),900)
    if n<80: continue
    ids=r["resp_ids"][:n]; wt_all=None
    for j in range(PRE,n-POST):
        if t[j]<1.2 or s[j]>0.4: continue
        a,b=j-PRE,j+POST
        if (t[a:b]>1.0).sum()>4: continue
        if wt_all is None: wt_all=[disp(x) for x in ids]
        wt=wt_all[a:b]
        if np.mean([bool(NAT.match(x)) for x in wt])<0.9 or any(("\\" in x or "{" in x or "$" in x) for x in wt): continue
        if np.mean([x.isdigit() for x in wt])>0.2: continue
        rho_w=float(np.corrcoef(s[a:b],t[a:b])[0,1]) if s[a:b].std()>1e-6 and t[a:b].std()>1e-6 else 1.0
        pre.append(dict(seq=i,prompt_idx=r["idx"],pos=j,a=a,b=b,Ht=float(t[j]),Hs=float(s[j]),rho_w=rho_w,gap=float(t[j]-s[j]),toks=wt))
pre.sort(key=lambda h:-h["gap"]); print("prefilter hits",len(pre))
byseq={}
for h in pre:
    byseq.setdefault(h["seq"],[]).append(h)
seqs=list(byseq)[:70]; print("rollouts to forward",len(seqs))
dev="cuda"; ms=AutoModelForCausalLM.from_pretrained(STU,dtype=torch.bfloat16,attn_implementation="sdpa").to(dev).eval()
mt=AutoModelForCausalLM.from_pretrained(TEA,dtype=torch.bfloat16,attn_implementation="sdpa").to(dev).eval()
out=[]
with torch.inference_mode():
    for seq in seqs:
        r=recs[seq]; p,sids=r["prompt_ids"],r["resp_ids"]; n=len(sids); x=torch.tensor([p+sids],device=dev); res={}
        for name,m in (("S",ms),("T",mt)):
            lp=torch.log_softmax(m(x,use_cache=False).logits[0][len(p)-1:len(p)+n-1].float(),-1)
            H=(-(lp.exp()*lp).sum(-1)).cpu().numpy(); tk=torch.topk(lp,16,dim=-1); res[name]=(H,tk.indices.cpu().numpy(),tk.values.exp().cpu().numpy())
        ov16=float(np.mean([len(set(res["S"][1][j])&set(res["T"][1][j]))/16 for j in range(n)])); rho=float(np.corrcoef(res["S"][0],res["T"][0])[0,1])
        for h in byseq[seq][:3]:
            j=h["pos"]; tT=[(disp(int(i)),float(v)) for i,v in zip(res["T"][1][j],res["T"][2][j])]; tS=[(disp(int(i)),float(v)) for i,v in zip(res["S"][1][j],res["S"][2][j])]
            t_alts=[w for w,v in tT[:4] if w.lower().strip(",.") in BR and v>0.05]; s_top=tS[0]
            s_alts=[w for w,v in tS[:3] if w.lower().strip(",.") in BR and v>0.05]
            tok_here=disp(sids[j])
            score=h["gap"]+1.5*len([w for w in t_alts if w.lower() not in [q.lower() for q in s_alts]])+(1.0 if tok_here.lower().strip(",.") in BR else 0)+(0.8 if s_top[1]>0.8 else 0)
            lo,hi=h["a"]-4,h["b"]+4
            out.append(dict(h,tok=tok_here,n=n,rho_all=rho,ov16_all=ov16,score=float(score),t_alts=t_alts,s_alts=s_alts,
                lo=lo,hi=hi,toks_all=[disp(i) for i in sids[lo:hi]],Hs_win=res["S"][0][lo:hi].tolist(),Ht_win=res["T"][0][lo:hi].tolist(),
                topk={str(k):dict(S=[(disp(int(i)),float(v)) for i,v in zip(res["S"][1][k],res["S"][2][k])],T=[(disp(int(i)),float(v)) for i,v in zip(res["T"][1][k],res["T"][2][k])]) for k in range(lo,hi)}))
out.sort(key=lambda h:-h["score"])
for h in out[:25]: print(h["seq"],h["prompt_idx"],h["pos"],repr(h["tok"]),f"Ht={h['Ht']:.2f} Hs={h['Hs']:.2f} rho_w={h['rho_w']:.2f} score={h['score']:.2f}","T4:",[(w,round(v,2)) for w,v in h["topk"][str(h["pos"])]["T"][:4]],"S3:",[(w,round(v,2)) for w,v in h["topk"][str(h["pos"])]["S"][:3]]," | "," ".join(h["toks"]))
json.dump(out[:40],open("/home/sya225/teaser_forks2.json","w")); print("WROTE")
