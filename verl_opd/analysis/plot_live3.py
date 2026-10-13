"""The three runs in flight (2026-09-20 evening), each against its own reference.
rho column differs per run: JustRL/M500 is the PREFERENCE source in some runs and the SHAPE source in others."""
import csv, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":8,"pdf.fonttype":42})
def load(p):
    d={}
    for r in csv.DictReader(open(p)):
        for k,v in r.items():
            try: d.setdefault(k,[]).append(float(v))
            except (ValueError,TypeError): d.setdefault(k,[]).append(np.nan)
    d={k:np.array(v) for k,v in d.items()}
    for new,old in {"response_length/mean":"response_length_mean","actor/entropy":"actor_entropy",
                    "actor/grad_norm":"actor_grad_norm","actor/distillation/abs_loss":"actor_distillation_abs_loss",
                    "actor/probe/rho":"rho","actor/probe/overlap16":"overlap16"}.items():
        if new not in d and old in d: d[new]=d[old]
    return d
def ma(y,w=5):
    y=np.asarray(y,float); k=np.ones(w)/w
    return np.concatenate([np.full(w-1,np.nan), np.convolve(np.nan_to_num(y,nan=np.nanmean(y)),k,mode="valid")])
# label, file, rho-key toward the reference teacher, overlap-key, colour
LIVE=[("DS: prefs only (JustRL prefs + student shape)","ser_live_jrstu.csv","actor/tp/rho_s_A","actor/tp/overlap16_sA","#d62728"),
      ("DS: level only (flat, JustRL mean entropy)",    "ser_live_flat.csv", None,               "actor/tp/overlap16_sB","#e6550d"),
      ("Q3: shape only (student prefs + M500 shape)",   "ser_live_q3stu.csv","actor/tp/rho_s_B","actor/tp/overlap16_sB","#2ca02c")]
REF=[("DS: JustRL direct (ref)","ser_JustRL.csv","actor/probe/rho","actor/probe/overlap16","#1f77b4","--"),
     ("Q3: M500 direct (ref)","probe_q3i17_fromK4Bmath500.csv","actor/probe/rho","actor/probe/overlap16","#7f7f7f","--")]
fig,ax=plt.subplots(2,3,figsize=(10.5,5.2))
def P(a,key_idx,title,log=False):
    for lab,f,rk,ok,c in LIVE:
        d=load(f); x=d["step"]; k=[rk,ok,"response_length/mean","actor/entropy",None,"actor/grad_norm"][key_idx]
        if k is None and key_idx==0: continue      # rho vs B is not logged in flat mode
        if k is None:
            k=("actor/tp/abs_k1_A" if "actor/tp/abs_k1_A" in d and not np.all(np.isnan(d["actor/tp/abs_k1_A"])) else "actor/distillation/abs_loss")
        if k not in d or np.all(np.isnan(d[k])): continue
        a.plot(x,ma(d[k]),c=c,lw=1.4,label=lab)
    for lab,f,rk,ok,c,ls in REF:
        d=load(f); x=d["step"]; m=x<=200; k=[rk,ok,"response_length/mean","actor/entropy",None,"actor/grad_norm"][key_idx]
        if k is None: k="actor/distillation/abs_loss"
        if k not in d or np.all(np.isnan(d[k])): continue
        a.plot(x[m],ma(d[k])[m],c=c,lw=1.0,ls=ls,alpha=.75,label=lab)
    a.set_title(title,fontsize=8.5); a.set_xlabel("training step"); a.grid(alpha=.25)
    a.spines[["top","right"]].set_visible(False)
    if log: a.set_yscale("log")
P(ax[0,0],0,r"$\rho$(student, reference teacher)   [flat: not logged]")
P(ax[0,1],1,"overlap@16 vs reference teacher")
P(ax[0,2],2,"mean response length (tokens)")
P(ax[1,0],3,"student entropy (nats)")
P(ax[1,1],4,"|k1| against the training target",log=True)
P(ax[1,2],5,"grad norm")
ax[0,0].legend(fontsize=6,frameon=False,loc="lower right")
fig.suptitle("Three runs in flight, 2026-09-20 19:40 — dashed = the matching direct-teaching reference",fontsize=9.5)
fig.tight_layout(); fig.savefig("live3_curves.png",dpi=140,bbox_inches="tight")
print("wrote live3_curves.png")
for lab,f,rk,ok,c in LIVE:
    d=load(f); x=d["step"]; m=x>=x.max()-9
    r=np.nanmean(d[rk][m]) if rk in d and not np.all(np.isnan(d[rk])) else float('nan')
    print(f"{lab:<48} step {int(x.max()):>3}  rho={r:.4f}  H={np.nanmean(d['actor/entropy'][m]):.3f}  len={np.nanmean(d['response_length/mean'][m]):.0f}")
