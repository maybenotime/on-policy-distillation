"""Qwen3-1.7B transplant grid: A = Qwen3-4B (ineffective, +0.7), B = Math-Step500 (effective, +16.2).
The comparable similarity is always rho(student, Math-Step500): tp/rho_s_B when M500 is the SHAPE source,
tp/rho_s_A when M500 is the PREFERENCE source.  Direct-teaching runs use the online probe."""
import csv, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":8,"pdf.fonttype":42})
def load(p):
    d={}
    for r in csv.DictReader(open(p)):
        for k,v in r.items():
            try: d.setdefault(k,[]).append(float(v))
            except (ValueError,TypeError): d.setdefault(k,[]).append(np.nan)
    d={k:np.array(v) for k,v in d.items()}
    ALIAS={"response_length/mean":"response_length_mean","actor/entropy":"actor_entropy",
           "actor/grad_norm":"actor_grad_norm","actor/distillation/abs_loss":"actor_distillation_abs_loss",
           "actor/probe/rho":"rho","actor/probe/overlap16":"overlap16"}
    for new,old in ALIAS.items():
        if new not in d and old in d: d[new]=d[old]
    return d
def ma(y,w=5):
    y=np.asarray(y,float); k=np.ones(w)/w
    return np.concatenate([np.full(w-1,np.nan), np.convolve(np.nan_to_num(y,nan=np.nanmean(y)),k,mode="valid")])
# label, file, rho-key vs M500, overlap-key vs M500, colour
R=[("B = M500 direct (+14.5 @s50)",        "probe_q3i17_fromK4Bmath500.csv","actor/probe/rho",  "actor/probe/overlap16", "#d62728"),
   ("B prefs + A shape (+12.7)",           "ser_q3_tpM500_shapeA4B.csv",    "actor/tp/rho_s_A", "actor/tp/overlap16_sA", "#ff7f0e"),
   ("B prefs + student shape (+10.4)",     "ser_q3_tpM500_shapeStu.csv",    "actor/tp/rho_s_A", "actor/tp/overlap16_sA", "#8c564b"),
   ("A prefs + B shape (+3.8)",            "ser_q3_tpA4B_shapeM500.csv",    "actor/tp/rho_s_B", "actor/tp/overlap16_sB", "#2ca02c"),
   ("A prefs + uniform T*=0.89 (+0.0)",    "ser_q3_tpA4B_uniform.csv",      None,               None,                    "#9467bd"),
   ("A prefs + student shape (+0.6)",      "ser_q3_tpA4B_shapeStu.csv",     None,               None,                    "#7f7f7f"),
   ("A = Qwen3-4B direct (+0.7)",          "probe_q3i17_from4B.csv",        None,               None,                    "#1f77b4")]
fig,ax=plt.subplots(2,3,figsize=(10.5,5.2))
for lab,f,rk,ok,c in R:
    try: d=load(f)
    except FileNotFoundError: print("missing",f); continue
    x=d["step"]; m=x<=200
    if rk and rk in d and not np.all(np.isnan(d[rk])):
        ax[0,0].plot(x[m],ma(d[rk])[m],c=c,lw=1.3,label=lab)
        ax[0,1].plot(x[m],ma(d[ok])[m],c=c,lw=1.3,label=lab)
    ax[0,2].plot(x[m],ma(d["response_length/mean"])[m],c=c,lw=1.3,label=lab)
    ax[1,0].plot(x[m],ma(d["actor/entropy"])[m],c=c,lw=1.3,label=lab)
    k1=d.get("actor/tp/abs_k1_A"); k1=k1 if k1 is not None and not np.all(np.isnan(k1)) else d["actor/distillation/abs_loss"]
    ax[1,1].plot(x[m],ma(k1)[m],c=c,lw=1.3,label=lab)
    ax[1,2].plot(x[m],ma(d["actor/grad_norm"])[m],c=c,lw=1.3,label=lab)
T=[r"$\rho$(student, Math-Step500)  — only runs where M500 is present","overlap@16 vs Math-Step500","mean response length (tokens)",
   "student entropy (nats)","|k1| against the training target","grad norm"]
for a,t in zip(ax.ravel(),T):
    a.set_title(t,fontsize=8.5); a.set_xlabel("training step"); a.grid(alpha=.25); a.spines[["top","right"]].set_visible(False)
    a.axvline(50,ls=":",c="gray",lw=.7)
ax[1,1].set_yscale("log")
ax[0,2].legend(fontsize=6,frameon=False,loc="upper left")
fig.suptitle("Qwen3-1.7B transplant grid (A = Qwen3-4B, B = Math-Step500); dotted line = step 50, where the grid is compared",fontsize=9.5)
fig.tight_layout(); fig.savefig("q3_tp_curves.png",dpi=140,bbox_inches="tight"); fig.savefig("q3_tp_curves.pdf",bbox_inches="tight")
print("wrote q3_tp_curves.png")
for lab,f,rk,ok,c in R:
    d=load(f); x=d["step"]
    m=(x>=41)&(x<=50)
    r = np.nanmean(d[rk][m]) if rk and rk in d and not np.all(np.isnan(d[rk])) else float('nan')
    print(f"{lab:<34} @s41-50  rho={r:.4f}  H={np.nanmean(d['actor/entropy'][m]):.3f}  len={np.nanmean(d['response_length/mean'][m]):.0f}")
