"""Student commits (>=0.85) to a CONTENT word (alphabetic, len>=4, not a function/discourse word) while the teacher spreads
mass over >=2 fork words (Wait/But/Or/Maybe/However/...).  Scan all rollouts (entropy + token prefilter), forward the rest."""
import json, re, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
PAIR=sys.argv[1]
F="/home/sya225/opd/fork"
if PAIR=="ds": STU,TEA,ROLL,ENTS,ENTT="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B","hbx/JustRL-DeepSeek-1.5B","roll_ds15b-dapo.jsonl","ent_student_ds15b-dapo.npz","ent_TJustRL_ds15b-dapo.npz"
else: STU,TEA,ROLL,ENTS,ENTT="Qwen/Qwen3-1.7B","Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500","roll_q3i17-dapo.jsonl","ent_student_q3i17-dapo.npz","ent_TK4Bmath500_q3i17-dapo.npz"
recs=[json.loads(l) for l in open(f"{F}/{ROLL}")]
ES=np.load(f"{F}/{ENTS}",allow_pickle=True); ET=np.load(f"{F}/{ENTT}",allow_pickle=True)
tok=AutoTokenizer.from_pretrained(STU)
def disp(i):
    s=tok.decode([i]); s=s.replace("\n","\\n"); return s.strip() if s.strip() else ("␣" if s==" " else s)
BR={"wait","but","hmm","however","alternatively","actually","hold","maybe","perhaps","let","suppose","although","though","instead","or","yet","okay","no","hmmm","unless"}
STOP=BR|{"so","then","thus","therefore","now","also","since","because","hence","first","next","well","the","a","an","and","of","to","in","on","at","for","with","that","this","which","where","when","is","are","was","were","be","it","its","as","by","from","if","we","i","you","they","he","she","there","here","not","can","will","would","should","could","must","have","has","had","do","does","did","than","each","all","any","some","both","into","over","up","out","about","one","two","three","get","got","like","just","only","very","more","most","such","same","other","same","s","t","me","my","our","their","what","how","who","them","these","those","being","been","also"}
NAT=re.compile(r"^[A-Za-z0-9,.:;!?'()+=\-/*]+$"); PRE,POST=12,14
byseq={}
for i,r in enumerate(recs):
    s=np.asarray(ES[str(i)],np.float32); t=np.asarray(ET[str(i)],np.float32); n=min(len(s),len(t),len(r["resp_ids"]),1500)
    if n<60: continue
    ids=r["resp_ids"][:n]; wt_all=None
    for j in range(PRE,n-POST):
        if t[j]<1.2 or s[j]>0.4: continue
        if wt_all is None: wt_all=[disp(x) for x in ids]
        w=wt_all[j]
        if not (w.isalpha() and len(w)>=3 and w.lower() not in STOP): continue
        a,b=j-PRE,j+POST; wt=wt_all[a:b]
        if np.mean([bool(NAT.match(x)) for x in wt])<0.85 or any(("\\" in x or "{" in x or "$" in x) for x in wt): continue
        if np.mean([x.isdigit() for x in wt])>0.2 or (t[a:b]>1.0).sum()>6: continue
        byseq.setdefault(i,[]).append(dict(seq=i,prompt_idx=r["idx"],pos=j,a=a,b=b,Ht0=float(t[j]),Hs0=float(s[j]),gap=float(t[j]-s[j]),tok=w,toks=wt))
print("rollouts with candidates",len(byseq),"positions",sum(len(v) for v in byseq.values()))
dev="cuda"; ms=AutoModelForCausalLM.from_pretrained(STU,dtype=torch.bfloat16,attn_implementation="sdpa").to(dev).eval()
mt=AutoModelForCausalLM.from_pretrained(TEA,dtype=torch.bfloat16,attn_implementation="sdpa").to(dev).eval()
out=[]
with torch.inference_mode():
    for c,(seq,hs) in enumerate(byseq.items()):
        r=recs[seq]; p,sids=r["prompt_ids"],r["resp_ids"]; n=len(sids); x=torch.tensor([p+sids],device=dev); res={}
        for name,m in (("S",ms),("T",mt)):
            lp=torch.log_softmax(m(x,use_cache=False).logits[0][len(p)-1:len(p)+n-1].float(),-1)
            H=(-(lp.exp()*lp).sum(-1)).cpu().numpy(); tk=torch.topk(lp,16,dim=-1); res[name]=(H,tk.indices.cpu().numpy(),tk.values.exp().cpu().numpy())
        ov16=None; rho=None
        for h in hs:
            j=h["pos"]; tT=[(disp(int(i)),float(v)) for i,v in zip(res["T"][1][j],res["T"][2][j])]; tS=[(disp(int(i)),float(v)) for i,v in zip(res["S"][1][j],res["S"][2][j])]
            s1=tS[0]; t_br=[(w,v) for w,v in tT[:6] if w.lower().strip(",.") in BR]
            if s1[1]<0.8 or s1[0]!=h["tok"] or len(t_br)<1 or sum(v for _,v in t_br)<0.2 or max(v for _,v in t_br)<0.12: continue
            if ov16 is None:
                ov16=float(np.mean([len(set(res["S"][1][k])&set(res["T"][1][k]))/16 for k in range(n)])); rho=float(np.corrcoef(res["S"][0],res["T"][0])[0,1])
            lo,hi=h["a"]-4,h["b"]+4
            out.append(dict(h,n=n,rho_all=rho,ov16_all=ov16,score=float(sum(v for _,v in t_br)+0.3*len(t_br)+0.2*h["gap"]),t_br=t_br,lo=lo,hi=hi,toks_all=[disp(i) for i in sids[lo:hi]],
                Hs_win=res["S"][0][lo:hi].tolist(),Ht_win=res["T"][0][lo:hi].tolist(),
                topk={str(k):dict(S=[(disp(int(i)),float(v)) for i,v in zip(res["S"][1][k],res["S"][2][k])],T=[(disp(int(i)),float(v)) for i,v in zip(res["T"][1][k],res["T"][2][k])]) for k in range(lo,hi)}))
        if c%50==0: print("fwd",c,len(out),flush=True)
out.sort(key=lambda h:-h["score"]); print("hits",len(out))
for h in out[:40]: print(h["seq"],h["prompt_idx"],h["pos"],repr(h["tok"]),f"Ht={h['Ht0']:.2f} Hs={h['Hs0']:.2f} score={h['score']:.2f}","T6:",[(w,round(v,2)) for w,v in h["topk"][str(h["pos"])]["T"][:6]],"S2:",[(w,round(v,2)) for w,v in h["topk"][str(h["pos"])]["S"][:2]]," | "," ".join(h["toks"]))
json.dump(out[:60],open(f"/home/sya225/teaser_forks5_{PAIR}.json","w")); print("WROTE")
