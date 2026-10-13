"""Prediction task on the code-domain pool (student Qwen3-1.7B, OT3-code-30k, 424 steps; log 1.1-1.3).
Pre-probes on the base student's own 1000 code rollouts (4k, T0.7). Gains = LCB v6 pass@1 and HE+ pass@1 over the base
(LCB 14.97, HE+ 60.1). Teacher acc = teacher's own HE+ (Code-300 never scored -> excluded from that predictor)."""
import numpy as np
rng=np.random.default_rng(0)
#        teacher, rho, FO, ov16, dH, teacher HE+, size, dLCB, dHE+
P=[("Qwen3-8B",      .885,77.8,71.1,.125,80.0,8, 2.3, 3.8),
   ("Qwen3-4B",      .884,78.3,72.0,.041,72.1,4, 4.0, 4.1),
   ("Qwen3-14B",     .873,76.4,69.4,.166,85.5,14,3.7, 6.7),
   ("RLCR-4B",       .866,76.4,70.8,.045,75.0,4, 3.6, 3.5),
   ("Code-300",      .850,74.7,70.0,.114,np.nan,4,8.4,14.0),
   ("Code-1200",     .818,71.3,68.1,.171,80.9,4,11.3,14.0)]
name=["pre-ρ (lower)","ForkOverlap (lower)","overlap@16 (lower)","ΔH (higher)","teacher HE+ (higher)","teacher size (higher)"]
sign=[-1,-1,-1,1,1,1]; cols=[1,2,3,4,5,6]
def spear(x,y):
    rx=np.argsort(np.argsort(x)); ry=np.argsort(np.argsort(y)); return np.corrcoef(rx,ry)[0,1]
def ci(x,y,B=5000):
    n=len(x); v=[]
    for _ in range(B):
        i=rng.integers(0,n,n)
        if len(set(x[i]))<3 or len(set(y[i]))<3: continue
        v.append(spear(x[i],y[i]))
    return np.percentile(v,[2.5,97.5])
for gi,gname in ((7,"LCB"),(8,"HE+")):
    g=np.array([p[gi] for p in P])
    print(f"\n### target = Δ{gname}  (n=6)")
    print("| predictor | Spearman | 95% CI | regret | top-1 |"); print("|---|---|---|---|---|")
    for nm,sg,c in zip(name,sign,cols):
        x=np.array([p[c] for p in P],float); ok=~np.isnan(x); xs=x[ok]*sg; gs=g[ok]
        s=spear(xs,gs); lo,hi=ci(xs,gs); pick=np.argmax(xs); reg=gs.max()-gs[pick]
        print(f"| {nm} | {s:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {reg:.1f} | {'yes' if reg==0 else 'no'} |" + ("  (n=5, Code-300 unscored)" if (~ok).any() else ""))
print("\nper-pairing:")
print("| teacher | pre-ρ | FO | ov@16 | ΔH | teacher HE+ | ΔLCB | ΔHE+ |"); print("|---|---|---|---|---|---|---|---|")
for p in P: print(f"| {p[0]} | {p[1]:.3f} | {p[2]:.1f} | {p[3]:.1f} | {p[4]:.3f} | {'–' if np.isnan(p[5]) else p[5]} | {p[7]:+.1f} | {p[8]:+.1f} |")
