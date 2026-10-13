"""Find positions where the TEACHER forks at an explicit discourse token (Wait/But/Hmm/...) while the STUDENT is low-entropy,
inside a natural-text window; then teacher-force both models for the top hits and dump tokens/entropies/top-16."""
import json, re, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
F="/home/sya225/opd/fork"; STU="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"; TEA="hbx/JustRL-DeepSeek-1.5B"
recs=[json.loads(l) for l in open(f"{F}/roll_ds15b-dapo.jsonl")]
ES=np.load(f"{F}/ent_student_ds15b-dapo.npz",allow_pickle=True); ET=np.load(f"{F}/ent_TJustRL_ds15b-dapo.npz",allow_pickle=True)
tok=AutoTokenizer.from_pretrained(STU)
def disp(i):
    s=tok.decode([i]); s=s.replace("\n","\\n"); return s.strip() if s.strip() else ("␣" if s==" " else s)
FORKW={"wait","but","hmm","alternatively","however","actually","let","maybe","perhaps","so","then","okay","now","therefore","thus","hold","also"}
NAT=re.compile(r"^[A-Za-z0-9,.:;!?'()+=\-/*]+$")
PRE,POST=10,16; hits=[]
for i,r in enumerate(recs):
    s=np.asarray(ES[str(i)],np.float32); t=np.asarray(ET[str(i)],np.float32); n=min(len(s),len(t),len(r["resp_ids"]))
    if n<80: continue
    ids=r["resp_ids"][:n]
    for j in range(PRE,n-POST):
        if t[j]<1.3 or s[j]>0.45: continue
        w=disp(ids[j]).lower().strip(",.")
        if w not in FORKW: continue
        a,b=j-PRE,j+POST; wt=[disp(x) for x in ids[a:b]]
        nat=np.mean([bool(NAT.match(x)) for x in wt]); 
        if nat<0.88 or any(("\\" in x or "{" in x or "$" in x) for x in wt): continue
        digits=np.mean([x.isdigit() for x in wt])
        if digits>0.25: continue
        other=[k for k in range(a,b) if k!=j and t[k]>1.0]
        rho_w=float(np.corrcoef(s[a:b],t[a:b])[0,1]) if s[a:b].std()>1e-6 and t[a:b].std()>1e-6 else 1.0
        hits.append(dict(seq=i,prompt_idx=r["idx"],pos=j,word=w,Ht=float(t[j]),Hs=float(s[j]),a=a,b=b,n_other=len(other),rho_w=rho_w,
                         score=float(t[j]-s[j])+0.5*min(len(other),2)-0.3*digits*4,toks=wt))
hits.sort(key=lambda h:-h["score"]); print("hits",len(hits))
top=[]; seen=set()
for h in hits:
    if h["seq"] in seen: continue
    seen.add(h["seq"]); top.append(h)
    if len(top)>=16: break
for h in top: print(h["seq"],h["prompt_idx"],h["pos"],h["word"],f"Ht={h['Ht']:.2f} Hs={h['Hs']:.2f} other={h['n_other']} rho_w={h['rho_w']:.2f}"," ".join(h["toks"]))
dev="cuda"; ms=AutoModelForCausalLM.from_pretrained(STU,dtype=torch.bfloat16,attn_implementation="sdpa").to(dev).eval()
mt=AutoModelForCausalLM.from_pretrained(TEA,dtype=torch.bfloat16,attn_implementation="sdpa").to(dev).eval()
out={}
with torch.inference_mode():
    for h in top:
        r=recs[h["seq"]]; p,sids=r["prompt_ids"],r["resp_ids"]; n=len(sids); lo,hi=max(0,h["pos"]-30),min(n,h["pos"]+30)
        x=torch.tensor([p+sids],device=dev); res={}
        for name,m in (("S",ms),("T",mt)):
            lp=torch.log_softmax(m(x,use_cache=False).logits[0][len(p)-1:len(p)+n-1].float(),-1)
            H=(-(lp.exp()*lp).sum(-1)).cpu().numpy(); tk=torch.topk(lp,16,dim=-1)
            res[name]=(H,tk.indices.cpu().numpy(),tk.values.exp().cpu().numpy())
        ov16=float(np.mean([len(set(res["S"][1][j])&set(res["T"][1][j]))/16 for j in range(n)])); rho=float(np.corrcoef(res["S"][0],res["T"][0])[0,1])
        out[str(h["seq"])]=dict(h,n=n,rho_all=rho,ov16_all=ov16,lo=lo,hi=hi,toks_all=[disp(i) for i in sids[lo:hi]],Hs=res["S"][0][lo:hi].tolist(),Ht=res["T"][0][lo:hi].tolist(),
            topk={str(j):dict(S=[(disp(int(i)),float(v)) for i,v in zip(res["S"][1][j],res["S"][2][j])],T=[(disp(int(i)),float(v)) for i,v in zip(res["T"][1][j],res["T"][2][j])]) for j in range(lo,hi)})
        tk=out[str(h["seq"])]["topk"][str(h["pos"])]
        print("done",h["seq"],h["pos"],h["word"],"T:",[(t,round(v,2)) for t,v in tk["T"][:5]],"S:",[(t,round(v,2)) for t,v in tk["S"][:4]],"rho",round(rho,3),"ov16",round(ov16,3))
json.dump(out,open("/home/sya225/teaser_forks.json","w")); print("WROTE")
