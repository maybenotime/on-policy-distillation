"""Teaser-figure data: find natural-text windows with clear teacher forks on the base student's DAPO rollouts
(DS-R1-Distill-1.5B student, JustRL-1.5B teacher), then teacher-force both models on the chosen rollouts to get
top-16 candidates/probs at every window position and Overlap@16 over the whole rollout.  Writes a JSON of candidates."""
import json, re, sys, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
F="/home/sya225/opd/fork"; STU="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"; TEA="hbx/JustRL-DeepSeek-1.5B"
recs=[json.loads(l) for l in open(f"{F}/roll_ds15b-dapo.jsonl")]
ES=np.load(f"{F}/ent_student_ds15b-dapo.npz",allow_pickle=True); ET=np.load(f"{F}/ent_TJustRL_ds15b-dapo.npz",allow_pickle=True)
tok=AutoTokenizer.from_pretrained(STU)
def disp(i):
    s=tok.decode([i]); s=s.replace("\n","\\n"); return s.strip() if s.strip() else ("␣" if s==" " else s)
NAT=re.compile(r"^[A-Za-z0-9,.:;!?'()+=\-/*]+$")
FORKW={"wait","but","let","alternatively","hmm","however","so","actually","now","then","maybe","perhaps","first","let's"}
W=26; cands=[]
for i,r in enumerate(recs):
    s=np.asarray(ES[str(i)],np.float32); t=np.asarray(ET[str(i)],np.float32); n=min(len(s),len(t),len(r["resp_ids"]))
    if n<60: continue
    s,t=s[:n],t[:n]; rho_all=float(np.corrcoef(s,t)[0,1]) if s.std()>1e-6 and t.std()>1e-6 else np.nan
    ids=r["resp_ids"][:n]; toks=[disp(x) for x in ids]
    for a in range(0,n-W,4):
        b=a+W; ts,ss=t[a:b],s[a:b]; wt=toks[a:b]
        nat=np.mean([bool(NAT.match(x)) for x in wt])
        if nat<0.9 or any(("\\" in x or "{" in x or "$" in x) for x in wt): continue
        forks=[j for j in range(W) if ts[j]>1.0]
        if not (2<=len(forks)<=3): continue
        if any(abs(forks[j]-forks[j+1])<3 for j in range(len(forks)-1)): continue
        rho_w=float(np.corrcoef(ss,ts)[0,1]) if ss.std()>1e-6 and ts.std()>1e-6 else 1.0
        speaks=[j for j in range(W) if ss[j]>0.6]
        mism=sum(1 for j in speaks if ts[j]<0.5)+sum(1 for j in forks if ss[j]<0.5*ts[j])
        bonus=sum(1 for j in forks if wt[j].lower().strip(",.") in FORKW)
        score=2*bonus+mism+(1.0-abs(rho_w-0.45))*2+nat
        cands.append(dict(seq=i,prompt_idx=r["idx"],a=a,b=b,rho_all=rho_all,rho_w=rho_w,forks=forks,bonus=bonus,mism=mism,score=float(score),toks=wt,Ht=ts.tolist(),Hs=ss.tolist()))
cands.sort(key=lambda c:-c["score"]); print("windows",len(cands))
top=[]; seen=set()
for c in cands:
    if c["seq"] in seen: continue
    seen.add(c["seq"]); top.append(c)
    if len(top)>=12: break
for c in top: print(c["seq"],c["prompt_idx"],c["a"],f"rho_w={c['rho_w']:.2f} rho_all={c['rho_all']:.2f} bonus={c['bonus']} mism={c['mism']}"," ".join(c["toks"]))
# forward passes on the top candidates' rollouts: top-16 ids/probs at every window position, Overlap@16 over the rollout
dev="cuda"; out=[]
ms=AutoModelForCausalLM.from_pretrained(STU,dtype=torch.bfloat16,attn_implementation="sdpa").to(dev).eval()
mt=AutoModelForCausalLM.from_pretrained(TEA,dtype=torch.bfloat16,attn_implementation="sdpa").to(dev).eval()
with torch.inference_mode():
    for c in top:
        r=recs[c["seq"]]; p,sids=r["prompt_ids"],r["resp_ids"]; n=len(sids)
        x=torch.tensor([p+sids],device=dev)
        res={}
        for name,m in (("S",ms),("T",mt)):
            lg=m(x,use_cache=False).logits[0][len(p)-1:len(p)+n-1].float()
            lp=torch.log_softmax(lg,-1); tk=torch.topk(lp,16,dim=-1)
            res[name]=(tk.indices.cpu().numpy(),tk.values.exp().cpu().numpy(),(-(lp.exp()*lp).sum(-1)).cpu().numpy())
        ov16=float(np.mean([len(set(res["S"][0][j])&set(res["T"][0][j]))/16 for j in range(n)]))
        rho_chk=float(np.corrcoef(res["S"][2],res["T"][2])[0,1])
        win=[]
        for j in range(c["a"],c["b"]):
            sI,sP=res["S"][0][j],res["S"][1][j]; tI,tP=res["T"][0][j],res["T"][1][j]
            win.append(dict(pos=j,tok=disp(sids[j]),Hs=float(res["S"][2][j]),Ht=float(res["T"][2][j]),
                            S=[(disp(int(i)),float(v)) for i,v in zip(sI,sP)],T=[(disp(int(i)),float(v)) for i,v in zip(tI,tP)],
                            ov3=len(set(sI[:3])&set(tI[:3]))/3,ov5=len(set(sI[:5])&set(tI[:5]))/5))
        c2=dict(c); c2.update(overlap16_rollout=ov16,rho_all_check=rho_chk,window=win); out.append(c2)
        print("done",c["seq"],"ov16",round(ov16,3),"rho_chk",round(rho_chk,3))
json.dump(out,open("/home/sya225/teaser_cands.json","w"))
print("WROTE")
