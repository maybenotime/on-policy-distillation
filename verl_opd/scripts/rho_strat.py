"""Length-stratified comparison of mean rho between the both-wrong and both-correct
subsets, so the raw gap isn't just "harder problems produce longer trajectories".

  rho_strat.py --w_stu '...' --w_tea '...' --c_stu '...' --c_tea '...' --label DS
"""
import argparse, glob, numpy as np

ap = argparse.ArgumentParser()
for k in ("w_stu", "w_tea", "c_stu", "c_tea"):
    ap.add_argument("--" + k, required=True)
ap.add_argument("--label", default="")
ap.add_argument("--min_len", type=int, default=25)
a = ap.parse_args()

def collect(sg, tg):
    stem = lambda p: p.rsplit("_sh", 1)[-1]
    tmap = {stem(p): p for p in sorted(glob.glob(tg))}
    out = []
    for ps in sorted(glob.glob(sg)):
        if stem(ps) not in tmap:
            continue
        S, T = np.load(ps), np.load(tmap[stem(ps)])
        for k in S.files:
            if k not in T.files:
                continue
            hs, ht = S[k].astype(np.float64), T[k].astype(np.float64)
            n = min(len(hs), len(ht))
            if n < a.min_len or np.std(hs[:n]) == 0 or np.std(ht[:n]) == 0:
                continue
            out.append((n, np.corrcoef(hs[:n], ht[:n])[0, 1]))
    return np.array(out)

W, C = collect(a.w_stu, a.w_tea), collect(a.c_stu, a.c_tea)
print(f"\n===== {a.label}  (both-wrong n={len(W)}, both-correct n={len(C)}) =====")
print(f"raw          wrong rho={W[:,1].mean():.4f}  correct rho={C[:,1].mean():.4f}  "
      f"diff={W[:,1].mean()-C[:,1].mean():+.4f}")

# quantile bins over the POOLED length distribution
edges = np.quantile(np.concatenate([W[:, 0], C[:, 0]]), np.linspace(0, 1, 7))
edges[0], edges[-1] = -np.inf, np.inf
print(f"{'length bin':>18} {'n_w':>6} {'n_c':>6} {'rho_w':>8} {'rho_c':>8} {'diff':>8}")
num_w = num_c = den = 0.0
for i in range(len(edges) - 1):
    mw = (W[:, 0] > edges[i]) & (W[:, 0] <= edges[i + 1])
    mc = (C[:, 0] > edges[i]) & (C[:, 0] <= edges[i + 1])
    if mw.sum() < 20 or mc.sum() < 20:
        continue
    rw, rc = W[mw, 1].mean(), C[mc, 1].mean()
    lo = "-inf" if np.isneginf(edges[i]) else f"{edges[i]:.0f}"
    hi = "inf" if np.isposinf(edges[i + 1]) else f"{edges[i + 1]:.0f}"
    print(f"{lo+'-'+hi:>18} {mw.sum():>6} {mc.sum():>6} {rw:>8.4f} {rc:>8.4f} {rw-rc:>+8.4f}")
    wgt = mw.sum() + mc.sum()
    num_w += rw * wgt; num_c += rc * wgt; den += wgt
print(f"length-matched (pooled-bin weighted): wrong={num_w/den:.4f}  correct={num_c/den:.4f}  "
      f"diff={(num_w-num_c)/den:+.4f}")

# rho as a function of length, both subsets pooled -> how strong is the confound?
A = np.vstack([W, C])
r = np.corrcoef(A[:, 0], A[:, 1])[0, 1]
print(f"corr(len, rho) over both subsets = {r:+.3f}")
