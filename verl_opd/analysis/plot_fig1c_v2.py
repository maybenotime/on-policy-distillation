"""Pre-OPD shape correlation vs OPD gain, 10 pairings (DeepScaleR dropped, user 9-25).  Same data as the prediction
table (predict_table_noDSR.py): pre-rho from the base student's 1000 DAPO rollouts (4k, T0.7; Qwen3-4B = online step-1),
gain = six-benchmark average change at step 200."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
DS=[("JustRL-1.5B",.853,8.1),("Nemotron-1.5B",.812,11.0),("R1-Distill-7B",.930,0.2),("Skywork-OR1-7B",.864,3.5),("Light-R1-7B",.866,0.7)]
Q3=[("Qwen3-4B-RL-Math",.771,15.3),("Qwen3-4B-Code-1200",.837,8.8),("Qwen3-8B-GRPO-MATH",.865,5.1),("Qwen3-8B",.889,2.3),("Qwen3-4B",.895,0.5)]
def spear(x,y): rx=np.argsort(np.argsort(x)); ry=np.argsort(np.argsort(y)); return np.corrcoef(rx,ry)[0,1]
allx=[r for _,r,_ in DS+Q3]; ally=[g for _,_,g in DS+Q3]
fig,ax=plt.subplots(figsize=(4.2,2.85))
OFF={"JustRL-1.5B":(0.004,0.4,"right"),"Nemotron-1.5B":(0.004,0.3,"right"),"R1-Distill-7B":(-0.003,0.9,"left"),
     "Skywork-OR1-7B":(0.004,0.3,"right"),"Light-R1-7B":(-0.003,0.6,"left"),"Qwen3-4B-RL-Math":(0.004,-0.9,"right"),
     "Qwen3-4B-Code-1200":(-0.004,-1.1,"left"),"Qwen3-8B-GRPO-MATH":(-0.004,0.4,"left"),"Qwen3-8B":(0.004,0.5,"right"),"Qwen3-4B":(0.004,-1.35,"right")}
for grp,mk,c,lab in ((DS,"o","#1f77b4","R1-Distill-1.5B (16k)"),(Q3,"s","#d62728","Qwen3-1.7B (8k)")):
    ax.scatter([r for _,r,_ in grp],[g for _,_,g in grp],marker=mk,c=c,s=40,label=lab,zorder=3,edgecolor="k",lw=.4)
    for n,r,g in grp:
        dx,dy,ha=OFF[n]; ax.annotate(n,(r,g),(r+dx,g+dy),fontsize=6.5,ha=ha)
ax.set_xlabel(r"pre-OPD ESC $\rho$"); ax.set_ylabel(r"$\Delta$ Average Performance (points)")
ax.invert_xaxis(); ax.axhline(0,c="gray",lw=.6,ls=":"); ax.set_ylim(-1.5,17.5)
ax.set_title(f"Spearman = {spear(allx,ally):+.2f} (n={len(allx)})",fontsize=9)
ax.legend(loc="upper left",frameon=False,fontsize=8); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); out="paper_figs/fig1c_prerho_vs_gain_v2"; fig.savefig(out+".png",dpi=200); fig.savefig(out+".pdf")
print("saved",out,"spearman",round(spear(allx,ally),3),"DS",round(spear([r for _,r,_ in DS],[g for _,_,g in DS]),2),"Q3",round(spear([r for _,r,_ in Q3],[g for _,_,g in Q3]),2))
