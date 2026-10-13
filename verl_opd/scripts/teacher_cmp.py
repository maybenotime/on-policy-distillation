"""Paired teacher comparison against a fixed student rollout set.
All teachers are scored at identical token positions, so this pairs exactly."""
import numpy as np, itertools
D = "/home/sya225/opd/fork/"
S = np.load(D + "ent_student_Qwen3-1.7B.npz")
T = {"4B": np.load(D + "ent_T4B_Qwen3-1.7B.npz"),
     "8B": np.load(D + "ent_teacher_Qwen3-1.7B.npz"),
     "14B": np.load(D + "ent_T14B_Qwen3-1.7B.npz")}

def stats(hs, ht):
    n = min(len(hs), len(ht)); hs, ht = hs[:n], ht[:n]
    k = int(0.2 * n)
    a = set(np.argsort(-hs)[:k].tolist()); b = set(np.argsort(-ht)[:k].tolist())
    return np.corrcoef(hs, ht)[0, 1], len(a & b) / k

per = {k: {} for k in T}
for key in S.files:
    hs = S[key].astype(np.float64)
    if len(hs) < 25 or hs.std() == 0: continue
    ok = True
    vals = {}
    for k, z in T.items():
        ht = z[key].astype(np.float64)
        if len(ht) < 25 or ht.std() == 0: ok = False; break
        vals[k] = stats(hs, ht)
    if ok:
        for k in T: per[k][key] = vals[k]

keys = sorted(per["8B"], key=int)
print(f"paired on {len(keys)} prompts\n")
for a, b in [("4B", "8B"), ("14B", "8B"), ("14B", "4B")]:
    for i, nm in enumerate(["rho", "Fork%"]):
        sc = 100 if i else 1
        d = np.array([(per[a][k][i] - per[b][k][i]) * sc for k in keys])
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"  {a:>3} - {b:<3} d{nm:<6} {d.mean():+.4f}  95% CI "
              f"[{d.mean()-1.96*se:+.4f}, {d.mean()+1.96*se:+.4f}]")
    print()
