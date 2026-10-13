"""Stage 3 — entropy-shape metrics between a student and the teacher.

  rho          = Pearson(H_student[1..T], H_teacher[1..T]) per trajectory
  ForkOverlap  = |F_s & F_t| / floor(0.2T), F = top-20% highest-entropy positions
  dH           = |mean H_student - mean H_teacher|

Both are per-trajectory, then averaged over trajectories (the paper's definition).
"""
import argparse, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--student_ent", required=True)
ap.add_argument("--teacher_ent", required=True)
ap.add_argument("--label", default="")
ap.add_argument("--min_len", type=int, default=25)   # so floor(0.2T) >= 5
a = ap.parse_args()

S, T = np.load(a.student_ent), np.load(a.teacher_ent)
rhos, forks, dhs, skipped = [], [], [], 0
for k in S.files:
    hs, ht = S[k].astype(np.float64), T[k].astype(np.float64)
    n = min(len(hs), len(ht))
    if n < a.min_len or np.std(hs[:n]) == 0 or np.std(ht[:n]) == 0:
        skipped += 1; continue
    hs, ht = hs[:n], ht[:n]
    rhos.append(np.corrcoef(hs, ht)[0, 1])
    k20 = int(0.2 * n)
    fs = set(np.argsort(-hs)[:k20].tolist())
    ft = set(np.argsort(-ht)[:k20].tolist())
    forks.append(len(fs & ft) / k20)
    dhs.append(abs(hs.mean() - ht.mean()))

r, f, d = np.array(rhos), np.array(forks), np.array(dhs)
print(f"{a.label:<26} rho={r.mean():.3f}±{r.std():.3f}   "
      f"ForkOverlap={100*f.mean():.1f}%±{100*f.std():.1f}   "
      f"dH={d.mean():.3f}   n={len(r)} (skipped {skipped})")
