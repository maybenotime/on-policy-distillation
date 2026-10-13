"""D0 diagnostic for the transplant experiment: on the student's own rollouts, at fork positions (student entropy top-20%),
how often do A, B and the student agree on the argmax / top-5?  usage: tp_diag.py student.npz A.npz B.npz"""
import sys, numpy as np
S, A, B = (np.load(f) for f in sys.argv[1:4])
n = sum(1 for k in S.files if k.startswith("ent"))
def jac(a, b, k=5):
    a, b = a[:, :k], b[:, :k]
    inter = np.array([len(set(x) & set(y)) for x, y in zip(a, b)])
    return inter / (2 * k - inter)
rows = {"A_vs_S": [], "B_vs_S": [], "A_vs_B": []}; jrows = {k: [] for k in rows}; rows_all = {k: [] for k in rows}
rho = {"S_A": [], "S_B": [], "A_B": []}; fo = {"S_A": [], "S_B": [], "A_B": []}
nfork = 0
for i in range(n):
    hs, ha, hb = S[f"ent{i}"].astype(np.float32), A[f"ent{i}"].astype(np.float32), B[f"ent{i}"].astype(np.float32)
    T = min(len(hs), len(ha), len(hb))
    if T < 25: continue
    hs, ha, hb = hs[:T], ha[:T], hb[:T]
    ids_s, ids_a, ids_b = S[f"ids{i}"][:T], A[f"ids{i}"][:T], B[f"ids{i}"][:T]
    k = int(0.2 * T); fork = np.argsort(-hs)[:k]; nfork += k
    for name, x, y in (("A_vs_S", ids_a, ids_s), ("B_vs_S", ids_b, ids_s), ("A_vs_B", ids_a, ids_b)):
        rows[name].append((x[fork, 0] == y[fork, 0]).mean()); rows_all[name].append((x[:, 0] == y[:, 0]).mean())
        jrows[name].append(jac(x[fork], y[fork]).mean())
    for name, x, y in (("S_A", hs, ha), ("S_B", hs, hb), ("A_B", ha, hb)):
        if x.std() > 0 and y.std() > 0: rho[name].append(np.corrcoef(x, y)[0, 1])
        fx, fy = set(np.argsort(-x)[:k]), set(np.argsort(-y)[:k]); fo[name].append(len(fx & fy) / k)
print(f"sequences {len(rows['A_vs_S'])}  fork positions {nfork}")
print(f"{'pair':8s} {'argmax@fork':>12s} {'argmax@all':>11s} {'jaccard5@fork':>14s}")
for name in rows: print(f"{name:8s} {np.mean(rows[name]):12.3f} {np.mean(rows_all[name]):11.3f} {np.mean(jrows[name]):14.3f}")
print("rho:", {k: round(float(np.mean(v)), 3) for k, v in rho.items()}, " ForkOverlap:", {k: round(float(np.mean(v)), 3) for k, v in fo.items()})
