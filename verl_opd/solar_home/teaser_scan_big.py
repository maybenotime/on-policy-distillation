"""Scan freshly sampled base-student rollouts (no precomputed entropies): forward both models on every rollout, keep positions
where the student commits (>=0.8) to a content word (alphabetic, len>=3, not function/discourse) while the teacher puts >=0.2 mass
on fork words (>=0.12 on one of them), in a natural-text window."""
import json, re, sys, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
ROLL, OUT = sys.argv[1], sys.argv[2]
STU="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"; TEA="hbx/JustRL-DeepSeek-1.5B"
recs=[json.loads(l) for l in open(ROLL)]
tok=AutoTokenizer.from_pretrained(STU)
def disp(i):
    s=tok.decode([i]); s=s.replace("\n","\\n"); return s.strip() if s.strip() else ("␣" if s==" " else s)
BR={"wait","but","hmm","however","alternatively","actually","hold","maybe","perhaps","let","suppose","although","though","instead","or","yet","okay","no","hmmm","unless"}
STOP=BR|{"so","then","thus","therefore","now","also","since","because","hence","first","next","well","the","a","an","and","of","to","in","on","at","for","with","that","this","which","where","when","is","are","was","were","be","it","its","as","by","from","if","we","i","you","they","he","she","there","here","not","can","will","would","should","could","must","have","has","had","do","does","did","than","each","all","any","some","both","into","over","up","out","about","one","two","three","get","got","like","just","only","very","more","most","such","same","other","s","t","me","my","our","their","what","how","who","them","these","those","being","been"}
NAT=re.compile(r"^[A-Za-z0-9,.:;!?'()+=\-/*]+$"); PRE,POST=12,14
dev="cuda"; ms=AutoModelForCausalLM.from_pretrained(STU,dtype=torch.bfloat16,attn_implementation="sdpa").to(dev).eval()
mt=AutoModelForCausalLM.from_pretrained(TEA,dtype=torch.bfloat16,attn_implementation="sdpa").to(dev).eval()
out=[]
with torch.inference_mode():
    for c,r in enumerate(recs):
        p,sids=r["prompt_ids"],r["resp_ids"]; n=min(len(sids),1500)
        if n<60: continue
        x=torch.tensor([p+sids[:n]],device=dev); res={}
        for name,m in (("S",ms),("T",mt)):
            lp=torch.log_softmax(m(x,use_cache=False).logits[0][len(p)-1:len(p)+n-1].float(),-1)
            H=(-(lp.exp()*lp).sum(-1)).cpu().numpy(); tk=torch.topk(lp,16,dim=-1); res[name]=(H,tk.indices.cpu().numpy(),tk.values.exp().cpu().numpy())
        Hs,Ht=res["S"][0],res["T"][0]
        cand=np.where((Ht>1.2)&(Hs<0.4))[0]; cand=[j for j in cand if PRE<=j<n-POST]
        if not cand: continue
        wt_all=[disp(i) for i in sids[:n]]; rho=None; ov16=None
        for j in cand:
            w=wt_all[j]
            if not (w.isalpha() and len(w)>=3 and w.lower() not in STOP): continue
            sI,sP=res["S"][1][j],res["S"][2][j]
            if sP[0]<0.8 or disp(int(sI[0]))!=w: continue
            tT=[(disp(int(i)),float(v)) for i,v in zip(res["T"][1][j],res["T"][2][j])]
            t_br=[(ww,v) for ww,v in tT[:6] if ww.lower().strip(",.") in BR]
            if not t_br or sum(v for _,v in t_br)<0.2 or max(v for _,v in t_br)<0.12: continue
            a,b=int(j-PRE),int(j+POST); wt=wt_all[a:b]
            if np.mean([bool(NAT.match(x_)) for x_ in wt])<0.85 or any(("\\" in x_ or "{" in x_ or "$" in x_) for x_ in wt): continue
            if np.mean([x_.isdigit() for x_ in wt])>0.2 or (Ht[a:b]>1.0).sum()>6: continue
            if rho is None:
                rho=float(np.corrcoef(Hs,Ht)[0,1]); ov16=float(np.mean([len(set(res["S"][1][k])&set(res["T"][1][k]))/16 for k in range(n)]))
            lo,hi=a-4,b+4
            out.append(dict(seq=c,prompt_idx=r["idx"],pos=int(j),a=a,b=b,tok=w,Ht0=float(Ht[j]),Hs0=float(Hs[j]),gap=float(Ht[j]-Hs[j]),toks=wt,n=n,rho_all=rho,ov16_all=ov16,
                score=float(sum(v for _,v in t_br)+0.3*len(t_br)+0.2*(Ht[j]-Hs[j])),t_br=t_br,lo=lo,hi=hi,toks_all=wt_all[lo:hi],
                Hs_win=Hs[lo:hi].tolist(),Ht_win=Ht[lo:hi].tolist(),
                topk={str(k):dict(S=[(disp(int(i)),float(v)) for i,v in zip(res["S"][1][k],res["S"][2][k])],T=[(disp(int(i)),float(v)) for i,v in zip(res["T"][1][k],res["T"][2][k])]) for k in range(lo,hi)}))
        if c%200==0: print("fwd",c,"hits",len(out),flush=True)
out.sort(key=lambda h:-h["score"]); print("hits",len(out))
for h in out[:40]: print(h["seq"],h["prompt_idx"],h["pos"],repr(h["tok"]),f"Ht={h['Ht0']:.2f} Hs={h['Hs0']:.2f} score={h['score']:.2f}","T6:",[(w,round(v,2)) for w,v in h["topk"][str(h["pos"])]["T"][:6]],"S2:",[(w,round(v,2)) for w,v in h["topk"][str(h["pos"])]["S"][:2]]," | "," ".join(h["toks"]))
json.dump(out[:80],open(OUT,"w")); print("WROTE")
