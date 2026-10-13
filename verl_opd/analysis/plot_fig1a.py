"""Figure 1(a): per-position entropy along a student rollout, before vs after OPD (DS-R1-Distill-1.5B <- JustRL-1.5B).
Pre and post are DIFFERENT rollouts (each model is on-policy), so they are drawn as two strips; a third panel gives the
per-rollout rho distribution over all 1000 rollouts."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})
S0=np.load("fork_npz/ent_student_ds15b-dapo.npz"); T0=np.load("fork_npz/ent_TJustRL_ds15b-dapo.npz")
S1=np.load("fork_npz/ent_student_ds15b-fromJustRL-post.npz"); T1=np.load("fork_npz/ent_teacher_ds15b-fromJustRL-post.npz")
def rhos(S,T):
    out=[]
    for k in S.files:
        a,b=S[k].astype(float),T[k].astype(float); n=min(len(a),len(b))
        out.append(np.corrcoef(a[:n],b[:n])[0,1] if n>=25 and a[:n].std()>0 and b[:n].std()>0 else np.nan)
    return np.array(out)
r0,r1=rhos(S0,T0),rhos(S1,T1); m0,m1=np.nanmedian(r0),np.nanmedian(r1)
L0=np.array([len(S0[k]) for k in S0.files]); L1=np.array([len(S1[k]) for k in S1.files])
def pick(r,L,med,tol):
    c=[i for i in range(len(r)) if 400<=L[i]<=2000 and abs(r[i]-med)<tol]
    return c[0] if c else int(np.nanargmin(np.abs(r-med)))
i0,i1=pick(r0,L0,m0,.03),pick(r1,L1,m1,.02)
print(f"pre idx {i0} rho {r0[i0]:.3f} len {L0[i0]} | post idx {i1} rho {r1[i1]:.3f} len {L1[i1]} | medians {m0:.3f}->{m1:.3f}")
W=360
fig=plt.figure(figsize=(7.2,2.6)); gs=fig.add_gridspec(2,2,width_ratios=[3.2,1],hspace=.15,wspace=.25)
def strip(ax,S,T,i,label,rho,col):
    s,t=S[str(i)].astype(float)[:W],T[str(i)].astype(float)[:W]; n=min(len(s),len(t)); s,t=s[:n],t[:n]; x=np.arange(n)
    k=int(.2*n); forks=np.argsort(-t)[:k]
    for f in forks: ax.axvspan(f-.5,f+.5,color="#fdd0a2",lw=0,alpha=.9,zorder=0)
    ax.plot(x,t,c="k",lw=.9,label="teacher (JustRL-1.5B)")
    ax.plot(x,s,c=col,lw=.9,label=label)
    ax.set_xlim(0,n); ax.set_ylim(0,max(3.2,np.percentile(np.r_[s,t],99.5)+.3)); ax.set_ylabel("entropy (nats)")
    ax.text(.995,.05,rf"$\rho$ = {rho:.2f}",transform=ax.transAxes,ha="right",va="bottom",fontsize=8,bbox=dict(fc="white",ec="none",alpha=.8,pad=1))
    ax.legend(loc="upper left",frameon=False,fontsize=7,ncol=2); ax.spines[["top","right"]].set_visible(False)
a0=fig.add_subplot(gs[0,0]); strip(a0,S0,T0,i0,"student before OPD",r0[i0],"#9ecae1"); a0.set_xticklabels([])
a1=fig.add_subplot(gs[1,0]); strip(a1,S1,T1,i1,"student after OPD (own rollout)",r1[i1],"#08519c"); a1.set_xlabel("response position (tokens); shaded = teacher's top-20% entropy positions (forks)")
ah=fig.add_subplot(gs[:,1]); bins=np.linspace(.3,1,36)
ah.hist(r0[~np.isnan(r0)],bins,color="#9ecae1",alpha=.9,label=f"before (median {m0:.2f})")
ah.hist(r1[~np.isnan(r1)],bins,color="#08519c",alpha=.75,label=f"after (median {m1:.2f})")
ah.set_yscale("log"); ah.set_xlabel(r"per-rollout $\rho$ (1000 rollouts)"); ah.set_ylabel("count (log)"); ah.legend(frameon=False,fontsize=7,loc="upper left"); ah.spines[["top","right"]].set_visible(False)
fig.savefig("paper_figs/fig1a_entropy_shape.pdf",bbox_inches="tight"); fig.savefig("paper_figs/fig1a_entropy_shape.png",dpi=200,bbox_inches="tight"); print("wrote fig1a")
