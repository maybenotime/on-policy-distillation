"""Does estimating the teacher entropy from a top-K truncation (+ tail term, as the online probe does with K=64)
change rho?  Uses the top-32 dump: H_trunc = -(sum_{topK} p log p) - tail*log(tail), tail = 1 - sum p."""
import numpy as np, sys
S = np.load(sys.argv[1]); T = np.load(sys.argv[2]); lab = sys.argv[3]
n = sum(1 for k in S.files if k.startswith("ids"))
rf, rt, rt16 = [], [], []
for i in range(n):
    hs = S[f"ent{i}"].astype(np.float32); ht = T[f"ent{i}"].astype(np.float32)
    if len(hs) < 25: continue
    lq = T[f"lp{i}"].astype(np.float32); q = np.exp(lq)
    def htr(K):
        qk = q[:, :K]; tail = np.clip(1 - qk.sum(-1), 1e-6, 1); return -(qk * np.log(np.clip(qk, 1e-12, 1))).sum(-1) - tail * np.log(tail)
    h32 = htr(32); h16 = htr(16)
    for arr, h in ((rf, ht), (rt, h32), (rt16, h16)):
        if hs.std() > 1e-6 and h.std() > 1e-6: arr.append(np.corrcoef(hs, h)[0, 1])
print(f"{lab:10s} rho full-vocab={np.mean(rf):.3f}   top-32+tail={np.mean(rt):.3f}   top-16+tail={np.mean(rt16):.3f}")
