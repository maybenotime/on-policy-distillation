"""Transplant runs: token-level overlap@16 of the student's top-16 with each side.
Columns of *_series.txt: step, tp/rho_s_B, tp/rho_s_A, tp/rho_tilde_B, probe/rho, probe/overlap16, response_length/mean
(probe/overlap16 = student vs the vLLM teacher = the PREFERENCE source in every mode).
The per-side tp/overlap16_s{A,B} exist only from run 4 on (added 9-19); for run 3/5 only probe/overlap16 is available.
usage: plot_transplant_ov.py out.png"""
import csv, re, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def series(f):
    a = np.array([[float(x) for x in l.split()] for l in open(f) if len(l.split()) == 7])
    return dict(zip(("step","rB","rA","rq","rho","ov","len"), (a[:,i] for i in range(7))))
def ma(x,w=5): return np.convolve(x,np.ones(w)/w,mode="valid") if len(x)>=w else np.asarray(x)
def csvcol(f,c):
    r=list(csv.DictReader(open(f))); return np.array([float(x["step"]) for x in r]), np.array([float(x[c]) for x in r])
R3=series("q3i17_tpA4B_shapeM500_series.txt"); R5=series("q3i17_tpA4B_uniform_series.txt"); R4=series("q3i17_tpM500_shapeA4B_series.txt")
ex=lambda f,k: ([ (int(re.search(r"(?:^|\s)step:(\d+)",l).group(1)), float(m.group(1)))
                  for l in open(f) if (m:=re.search(re.escape(k)+r":(?:np\.float64\()?([0-9.]+)",l)) and re.search(r"(?:^|\s)step:(\d+)",l) ])
fig,ax=plt.subplots(1,2,figsize=(12,4.4))
sM,oM=csvcol("probe_q3i17_fromK4Bmath500.csv","overlap16"); s4,o4=csvcol("probe_q3i17_from4B.csv","overlap16")
for d,c,lab in ((R3,"crimson","run 3: 4B ⊕ shape(M500)"),(R4,"navy","run 4: M500 ⊕ shape(4B)"),(R5,"gray","run 5: 4B / T*")):
    y=ma(d["ov"]); ax[0].plot(d["step"][:len(y)],y,color=c,lw=1.8,label=lab)
y=ma(oM); ax[0].plot(sM[:len(y)],y,color="green",lw=1.6,ls="--",label="M500 direct")
y=ma(o4); ax[0].plot(s4[:len(y)],y,color="purple",lw=1.6,ls="--",label="4B direct")
ax[0].set_title("overlap@16 (student vs its preference source)",fontsize=10.5); ax[0].legend(fontsize=8); ax[0].set_ylim(0.70,0.80)
# run 4 only: both sides + the two teachers against each other
L="/dev/null"
try:
    import subprocess
except Exception: pass
for key,c,lab in (("actor/tp/overlap16_sA","navy","student vs M500 (preferences)"),
                  ("actor/tp/overlap16_sB","orange","student vs 4B (shape source)"),
                  ("actor/tp/overlap16_AB","black","M500 vs 4B")):
    pts=ex("run4_ovraw.txt",key)
    if pts:
        st=np.array([p[0] for p in pts]); v=np.array([p[1] for p in pts]); y=ma(v)
        ax[1].plot(st[:len(y)],y,color=c,lw=1.8,label=lab)
ax[1].set_title("run 4 only: overlap@16 against both sides",fontsize=10.5); ax[1].legend(fontsize=8)
for a in ax: a.grid(alpha=.25,lw=.5); a.set_xlabel("training step",fontsize=9); a.tick_params(labelsize=8.5)
fig.suptitle("Transplant runs — token-level overlap (5-step MA)",fontsize=12)
fig.tight_layout(rect=[0,0,1,0.93]); fig.savefig("transplant_q3i17_overlap.png",dpi=150); print("saved")
