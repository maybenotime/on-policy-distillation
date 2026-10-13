"""Positions where the STUDENT commits (top-1 > 0.8) to a NON-discourse token while the TEACHER is uncertain and spreads mass
over several discourse/fork tokens (Wait/But/However/Alternatively/...)."""
import json, re, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
F="/home/sya225/opd/fork"; STU="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"; TEA="hbx/JustRL-DeepSeek-1.5B"
recs=[json.loads(l) for l in open(f"{F}/roll_ds15b-dapo.jsonl")]
ES=np.load(f"{F}/ent_student_ds15b-dapo.npz",allow_pickle=True); ET=np.load(f"{F}/ent_TJustRL_ds15b-dapo.npz",allow_pickle=True)
tok=AutoTokenizer.from_pretrained(STU)
def disp(i):
    s=tok.decode([i]); s=s.replace("\n","\\n"); return s.strip() if s.strip() else ("␣" if s==" " else s)
BR={"wait","but","hmm","however","alternatively","actually","hold","maybe","perhaps","let","suppose","although","though","instead","or","yet","okay","hmmm","no"}
DISC=BR|{"so","then","thus","therefore","now","also","since","because","hence","first","next","well"}
NAT=re.compile(r"^[A-Za-z0-9,.:;!?'()+=\-/*]+$"); PRE,POST=12,14
pre=[]
for i,r in enumerate(recs):
    s=np.asarray(ES[str(i)],np.float32); t=np.asarray(ET[str(i)],np.float32); n=min(len(s),len(t),len(r["resp_ids"]),1200)
    if n<60: continue
    ids=r["resp_ids"][:n]; wt_all=None
    for j in range(PRE,n-POST):
        if t[j]<1.3 or s[j]>0.35: continue
        if wt_all is None: wt_all=[disp(x) for x in ids]
        if wt_all[j].lower().strip(",.") in DISC or not NAT.match(wt_all[j]): continue
        a,b=j-PRE,j+POST; wt=wt_all[a:b]
        if (t[a:b]>1.0).sum()>5: continue
        if np.mean([bool(NAT.match(x)) for x in wt])<0.85 or any(("\\" in x or "{" in x or "$" in x) for x in wt): continue
        if np.mean([x.isdigit() for x in wt])>0.25: continue
        pre.append(dict(seq=i,prompt_idx=r["idx"],pos=j,a=a,b=b,Ht0=float(t[j]),Hs0=float(s[j]),gap=float(t[j]-s[j]),toks=wt))
pre.sort(key=lambda h:-h["gap"]); print("prefilter",len(pre))
byseq={}
for h in pre: byseq.setdefault(h["seq"],[]).append(h)
seqs=list(byseq)[:160]; print("rollouts",len(seqs))
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
        for h in byseq[seq][:4]:
            j=h["pos"]; tT=[(disp(int(i)),float(v)) for i,v in zip(res["T"][1][j],res["T"][2][j])]; tS=[(disp(int(i)),float(v)) for i,v in zip(res["S"][1][j],res["S"][2][j])]
            t_br=[(w,v) for w,v in tT[:5] if w.lower().strip(",.") in BR]; s1=tS[0]
            if s1[1]<0.8 or s1[0].lower().strip(",.") in DISC or len(t_br)<2: continue
            score=sum(v for _,v in t_br)+0.5*len(t_br)+0.3*h["gap"]
            lo,hi=h["a"]-4,h["b"]+4
            out.append(dict(h,tok=disp(sids[j]),n=n,rho_all=rho,ov16_all=ov16,score=float(score),t_br=t_br,lo=lo,hi=hi,toks_all=[disp(i) for i in sids[lo:hi]],
                Hs_win=res["S"][0][lo:hi].tolist(),Ht_win=res["T"][0][lo:hi].tolist(),
                topk={str(k):dict(S=[(disp(int(i)),float(v)) for i,v in zip(res["S"][1][k],res["S"][2][k])],T=[(disp(int(i)),float(v)) for i,v in zip(res["T"][1][k],res["T"][2][k])]) for k in range(lo,hi)}))
out.sort(key=lambda h:-h["score"]); print("hits",len(out))
for h in out[:30]: print(h["seq"],h["prompt_idx"],h["pos"],repr(h["tok"]),f"Ht={h['Ht0']:.2f} Hs={h['Hs0']:.2f} score={h['score']:.2f}","T5:",[(w,round(v,2)) for w,v in h["topk"][str(h["pos"])]["T"][:5]],"S2:",[(w,round(v,2)) for w,v in h["topk"][str(h["pos"])]["S"][:2]]," | "," ".join(h["toks"]))
json.dump(out[:40],open("/home/sya225/teaser_forks3.json","w")); print("WROTE")
