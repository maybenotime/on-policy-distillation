"""Aggregate per-shard entropy npz files into mean rho / ForkOverlap / dH over a dataset.

  python rho_agg.py --student 'ent_stu_p1w_sh*.npz' --teacher 'ent_JustRL_p1w_sh*.npz' --label '...'

Shards are matched by sorted filename; within a shard, keys are the record index.
"""
import argparse, glob, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--student", required=True)
ap.add_argument("--teacher", required=True)
ap.add_argument("--label", default="")
ap.add_argument("--min_len", type=int, default=25)
a = ap.parse_args()

sf, tf = sorted(glob.glob(a.student)), sorted(glob.glob(a.teacher))
# only shards where both sides exist
stem = lambda p: p.rsplit("_sh", 1)[-1]
tmap = {stem(p): p for p in tf}
pairs = [(p, tmap[stem(p)]) for p in sf if stem(p) in tmap]

rhos, forks, dhs, lens, skipped = [], [], [], [], 0
for ps, pt in pairs:
    S, T = np.load(ps), np.load(pt)
    for k in S.files:
        if k not in T.files:
            skipped += 1; continue
        hs, ht = S[k].astype(np.float64), T[k].astype(np.float64)
        n = min(len(hs), len(ht))
        if n < a.min_len or np.std(hs[:n]) == 0 or np.std(ht[:n]) == 0:
            skipped += 1; continue
        hs, ht = hs[:n], ht[:n]
        rhos.append(np.corrcoef(hs, ht)[0, 1])
        k20 = int(0.2 * n)
        forks.append(len(set(np.argsort(-hs)[:k20].tolist()) &
                         set(np.argsort(-ht)[:k20].tolist())) / k20)
        dhs.append(abs(hs.mean() - ht.mean()))
        lens.append(n)

r, f, d, L = map(np.array, (rhos, forks, dhs, lens))
se = r.std() / max(1, np.sqrt(len(r)))
print(f"{a.label:<34} rho={r.mean():.4f}+-{r.std():.3f} (se {se:.4f})  "
      f"Fork={100*f.mean():.1f}%  dH={d.mean():.3f}  len={L.mean():.0f}  "
      f"n={len(r)} shards={len(pairs)} (skipped {skipped})", flush=True)
