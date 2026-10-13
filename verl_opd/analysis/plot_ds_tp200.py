"""The two node-05 DeepSeek transplant runs (crashed at s145 / s111) vs JustRL teaching directly.
For each run the meaningful similarity is rho(student, JustRL): it is tp/rho_s_B when JustRL is the
SHAPE source and tp/rho_s_A when JustRL is the PREFERENCE source."""
import csv, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":8,"pdf.fonttype":42})
def load(p):
    d={}
    for r in csv.DictReader(open(p)):
        for k,v in r.items():
            try: d.setdefault(k,[]).append(float(v))
            except (ValueError,TypeError): d.setdefault(k,[]).append(np.nan)
    return {k:np.array(v) for k,v in d.items()}
def ma(y,w=7):
    y=np.asarray(y,float); k=np.ones(w)/w
    out=np.convolve(np.nan_to_num(y,nan=np.nanmean(y)),k,mode="valid")
    return np.concatenate([np.full(w-1,np.nan),out])
S=load("ser_tpStu_200.csv"); J=load("ser_tpJRstu.csv"); D=load("ser_JustRL.csv")
RUNS=[("student prefs + JustRL shape\n(tpStu_shapeJustRL, died s145)", S, "actor/tp/rho_s_B", "actor/tp/overlap16_sB", "#2ca02c"),
      ("JustRL prefs + student shape\n(tpJustRL_shapeStu, died s111)", J, "actor/tp/rho_s_A", "actor/tp/overlap16_sA", "#d62728"),
      ("JustRL direct (reference)",                                    D, "actor/probe/rho",  "actor/probe/overlap16", "#1f77b4")]
fig,ax=plt.subplots(2,3,figsize=(10.5,5.4))
PAN=[("rho","rho(student, JustRL)"),("ov","overlap@16 (student vs JustRL)"),("len","mean response length (tokens)"),
     ("H","student entropy (nats)"),("k1","|k1| against the training target"),("g","grad norm")]
for lab,d,rk,ok,c in RUNS:
    x=d["step"]; m=x<=200
    ax[0,0].plot(x[m],ma(d[rk])[m],c=c,lw=1.3,label=lab)
    ax[0,1].plot(x[m],ma(d[ok])[m],c=c,lw=1.3,label=lab)
    ax[0,2].plot(x[m],ma(d["response_length/mean"])[m],c=c,lw=1.3,label=lab)
    ax[1,0].plot(x[m],ma(d["actor/entropy"])[m],c=c,lw=1.3,label=lab)
    k1=d["actor/tp/abs_k1_A"] if "actor/tp/abs_k1_A" in d and not np.all(np.isnan(d["actor/tp/abs_k1_A"])) else d["actor/distillation/abs_loss"]
    ax[1,1].plot(x[m],ma(k1)[m],c=c,lw=1.3,label=lab)
    ax[1,2].plot(x[m],ma(d["actor/grad_norm"])[m],c=c,lw=1.3,label=lab)
for a,(k,t) in zip(ax.ravel(),PAN):
    a.set_title(t,fontsize=9); a.set_xlabel("training step"); a.grid(alpha=.25)
    a.spines[["top","right"]].set_visible(False)
    for s in (100,145): a.axvline(s,ls=":",c="gray",lw=.7)
ax[1,1].set_yscale("log")
ax[0,0].legend(fontsize=6.5,frameon=False,loc="lower right")
fig.suptitle("DeepSeek-R1-Distill-1.5B: separating the preference channel from the shape channel\n"
             "dotted lines = last saved checkpoint (s100) and the crash point (s145)",fontsize=10)
fig.tight_layout(); fig.savefig("ds_tp200_curves.png",dpi=140,bbox_inches="tight"); fig.savefig("ds_tp200_curves.pdf",bbox_inches="tight")
print("wrote ds_tp200_curves.png")
for lab,d,rk,ok,c in RUNS:
    x=d["step"]
    for lo,hi in [(1,10),(96,111),(136,145)]:
        m=(x>=lo)&(x<=hi)
        if m.sum()<3: continue
        print(f"{lab.splitlines()[0][:34]:<36} {lo:>4}-{hi:<4} rho={np.nanmean(d[rk][m]):.4f} ov={np.nanmean(d[ok][m]):.4f} len={np.nanmean(d['response_length/mean'][m]):.0f}")
