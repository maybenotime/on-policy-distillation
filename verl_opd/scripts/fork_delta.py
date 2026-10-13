"""Paired pre/post-OPD delta. The two runs use different rollouts (each model is
on-policy) but the SAME 1000 prompts in the same order, so pairing by prompt index
gives a per-prompt delta and hence a real standard error on the change."""
import argparse, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--pre", required=True); ap.add_argument("--post", required=True)
ap.add_argument("--label", default=""); ap.add_argument("--min_len", type=int, default=25)
a = ap.parse_args()

def per_traj(sf, tf):
    S, T = np.load(sf), np.load(tf)
    out = {}
    for k in S.files:
        hs, ht = S[k].astype(np.float64), T[k].astype(np.float64)
        n = min(len(hs), len(ht))
        if n < a.min_len or np.std(hs[:n]) == 0 or np.std(ht[:n]) == 0:
            continue
        hs, ht = hs[:n], ht[:n]
        k20 = int(0.2 * n)
        fs = set(np.argsort(-hs)[:k20].tolist()); ft = set(np.argsort(-ht)[:k20].tolist())
        out[k] = (np.corrcoef(hs, ht)[0, 1], len(fs & ft) / k20,
                  abs(hs.mean() - ht.mean()))
    return out

D = "/home/USER/opd/fork/"
pre = per_traj(f"{D}ent_student_{a.pre}.npz", f"{D}ent_teacher_{a.pre}.npz")
post = per_traj(f"{D}ent_student_{a.post}.npz", f"{D}ent_teacher_{a.post}.npz")
common = sorted(set(pre) & set(post), key=int)
d = np.array([[post[k][i] - pre[k][i] for i in range(3)] for k in common])
n = len(common)
names = ["rho", "ForkOverlap", "dH"]
scale = [1, 100, 1]
print(f"{a.label}  (paired on {n} prompts)")
for i, nm in enumerate(names):
    x = d[:, i] * scale[i]
    se = x.std(ddof=1) / np.sqrt(n)
    lo, hi = x.mean() - 1.96 * se, x.mean() + 1.96 * se
    print(f"   d{nm:<12} {x.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
          f"(improved on {100*(x>0).mean():.0f}% of prompts)")
