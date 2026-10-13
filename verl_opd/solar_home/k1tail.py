import numpy as np, glob
for tag in ("p1w_JustRL","p1c_JustRL"):
    ks=[]
    for p in sorted(glob.glob(f"/localscratch/sya225/rho/k1_{tag}_sh*.npz")):
        z=np.load(p); n=sum(1 for f in z.files if f.startswith("k1_"))
        for i in range(n): ks.append(np.abs(z[f"k1_{i}"].astype(np.float32)))
    a=np.concatenate(ks)
    print(f"{tag}: n={a.size/1e6:.1f}M tokens  mean|k1|={a.mean():.3f}  frac>1={100*(a>1).mean():.2f}%  frac>3={100*(a>3).mean():.2f}%  frac>10={100*(a>10).mean():.3f}%  p99={np.percentile(a,99):.2f}")
