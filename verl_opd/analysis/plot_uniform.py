"""The DeepSeek uniform-T* control run: it does not hold the manipulation — training a model
against a flattened copy of its OWN logits is a runaway, and the student's entropy explodes."""
import csv, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":8,"pdf.fonttype":42})
def load(p):
    d={}
    for r in csv.DictReader(open(p)):
        for k,v in r.items():
            try: d.setdefault(k,[]).append(float(v))
            except (ValueError,TypeError): d.setdefault(k,[]).append(np.nan)
    return {k:np.array(v) for k,v in d.items()}
def ma(y,w=5):
    y=np.asarray(y,float); k=np.ones(w)/w
    return np.concatenate([np.full(w-1,np.nan), np.convolve(np.nan_to_num(y,nan=np.nanmean(y)),k,mode="valid")])
U=load("ser_tpUni.csv"); S=load("ser_tpStu_200.csv"); D=load("ser_JustRL.csv")
RUNS=[("uniform T*=1.287 (student's own logits, flattened)", U, "#e6550d"),
      ("per-position shape(JustRL)",                          S, "#2ca02c"),
      ("JustRL direct (reference)",                           D, "#1f77b4")]
fig,ax=plt.subplots(2,3,figsize=(10.5,5.2))
def P(a,key,title,log=False,ref=None):
    for lab,d,c in RUNS:
        if key not in d or np.all(np.isnan(d[key])): continue
        m=d["step"]<=200
        a.plot(d["step"][m],ma(d[key])[m],c=c,lw=1.3,label=lab)
    if ref is not None: a.axhline(ref,ls="--",c="gray",lw=.8)
    a.set_title(title,fontsize=9); a.set_xlabel("training step"); a.grid(alpha=.25)
    a.spines[["top","right"]].set_visible(False)
    if log: a.set_yscale("log")
    for s in (50,200): a.axvline(s,ls=":",c="gray",lw=.7)
P(ax[0,0],"actor/entropy","student entropy (nats)  — JustRL's level = 0.657",ref=0.657)
P(ax[0,1],"actor/probe/sampled_in_topk","sampled token inside teacher top-64")
P(ax[0,2],"response_length/mean","mean response length (tokens)")
P(ax[1,0],"actor/tp/overlap16_sA","overlap@16 (student vs its own preference source)")
P(ax[1,1],"actor/tp/abs_k1_A","|k1| against the training target",log=True)
P(ax[1,2],"actor/grad_norm","grad norm")
ax[0,0].legend(fontsize=6.5,frameon=False,loc="upper left")
fig.suptitle("DeepSeek-R1-Distill-1.5B: the uniform-temperature control degenerates\n"
             "dotted lines = the two evaluated checkpoints (s50 +1.6, s200 +1.3)",fontsize=10)
fig.tight_layout(); fig.savefig("ds_uniform_curves.png",dpi=140,bbox_inches="tight"); fig.savefig("ds_uniform_curves.pdf",bbox_inches="tight")
print("wrote ds_uniform_curves.png")
