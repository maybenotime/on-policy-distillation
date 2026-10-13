"""Rank-mismatch soft-weighted OPD (q3i17_ts_mismatchW) vs M500 teaching directly (uniform weights)."""
import csv, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":8,"pdf.fonttype":42})
def load(p):
    d={}
    for r in csv.DictReader(open(p)):
        for k,v in r.items():
            try: d.setdefault(k,[]).append(float(v))
            except (ValueError,TypeError): d.setdefault(k,[]).append(np.nan)
    d={k:np.array(v) for k,v in d.items()}
    for new,old in {"response_length/mean":"response_length_mean","actor/entropy":"actor_entropy","actor/grad_norm":"actor_grad_norm",
                    "actor/distillation/abs_loss":"actor_distillation_abs_loss","actor/probe/rho":"rho","actor/probe/overlap16":"overlap16"}.items():
        if new not in d and old in d: d[new]=d[old]
    return d
def ma(y,w=5):
    y=np.asarray(y,float); k=np.ones(w)/w
    return np.concatenate([np.full(w-1,np.nan), np.convolve(np.nan_to_num(y,nan=np.nanmean(y)),k,mode="valid")])
W=load("ser_tsw.csv"); D=load("probe_q3i17_fromK4Bmath500.csv")
last=int(W["step"].max())
fig,ax=plt.subplots(2,3,figsize=(10.5,5.2))
P=[("actor/probe/rho",r"$\rho$(student, M500)"),("actor/probe/overlap16","overlap@16 vs M500"),("response_length/mean","mean response length"),
   ("actor/entropy","student entropy (nats)"),("actor/distillation/abs_loss","|k1| (unweighted mean)"),("actor/grad_norm","grad norm")]
for a,(k,t) in zip(ax.ravel(),P):
    m=D["step"]<=max(60,last+10)
    a.plot(D["step"][m],ma(D[k])[m],c="#d62728",lw=1.2,ls="--",label="M500 direct (uniform weights)")
    a.plot(W["step"],ma(W[k]),c="#2ca02c",lw=1.5,label=f"rank-mismatch weighted (to step {last})")
    a.set_title(t,fontsize=9); a.set_xlabel("training step"); a.grid(alpha=.25); a.spines[["top","right"]].set_visible(False)
ax[0,0].legend(fontsize=7,frameon=False,loc="lower right")
# inset-style panel: the weighting diagnostics
ax2=ax[1,2].twinx(); ax2.plot(W["step"],W["select/abs_k1_weighted"]/W["select/abs_k1_unweighted"],c="#9467bd",lw=1.2,label="|k1| weighted/unweighted")
ax2.plot(W["step"],W["select/eff_frac"],c="#8c564b",lw=1.2,label="effective kept fraction"); ax2.set_ylim(0,1.6); ax2.legend(fontsize=6.5,frameon=False,loc="upper right")
fig.suptitle(f"Qwen3-1.7B <- Math-Step500: rank-mismatch soft weighting vs uniform, first {last} steps",fontsize=10)
fig.tight_layout(); fig.savefig("tsw_curves.png",dpi=140,bbox_inches="tight"); print("wrote tsw_curves.png")
for lo,hi in [(1,10),(11,20),(21,last)]:
    m=(W["step"]>=lo)&(W["step"]<=hi); md=(D["step"]>=lo)&(D["step"]<=hi)
    print(f"steps {lo:>2}-{hi:<3} weighted: rho={np.nanmean(W['actor/probe/rho'][m]):.3f} ov={np.nanmean(W['actor/probe/overlap16'][m]):.3f} H={np.nanmean(W['actor/entropy'][m]):.3f} len={np.nanmean(W['response_length/mean'][m]):.0f} k1ratio={np.nanmean(W['select/abs_k1_weighted'][m]/W['select/abs_k1_unweighted'][m]):.2f} | direct: rho={np.nanmean(D['actor/probe/rho'][md]):.3f} ov={np.nanmean(D['actor/probe/overlap16'][md]):.3f} H={np.nanmean(D['actor/entropy'][md]):.3f} len={np.nanmean(D['response_length/mean'][md]):.0f}")
