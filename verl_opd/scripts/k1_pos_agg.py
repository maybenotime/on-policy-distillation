"""Position-bucketed |k1| / overlap@16 for two datasets, so the comparison is not
confounded by the both-wrong trajectories simply being longer."""
import argparse, glob
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--w", required=True); ap.add_argument("--c", required=True)
ap.add_argument("--label", default="")
ap.add_argument("--bins", default="0,128,512,1024,2048")
a = ap.parse_args()
EDGES = [int(x) for x in a.bins.split(",")]

def load(pat):
    K, O = [], []
    for p in sorted(glob.glob(pat)):
        z = np.load(p)
        n = sum(1 for f in z.files if f.startswith("k1_"))
        for i in range(n):
            k, o = z[f"k1_{i}"].astype(np.float64), z[f"ov_{i}"].astype(np.float64)
            if len(k): K.append(k); O.append(o)
    return K, O

KW, OW = load(a.w); KC, OC = load(a.c)
print(f"\n===== {a.label}  (both-wrong n={len(KW)}, both-correct n={len(KC)}) =====")
print(f"per-trajectory mean |k1| (all positions <= cap): "
      f"wrong={np.mean([np.abs(k).mean() for k in KW]):.4f}  "
      f"correct={np.mean([np.abs(k).mean() for k in KC]):.4f}")
print(f"{'position bin':>14} {'n_w':>8} {'n_c':>8} {'|k1|_w':>8} {'|k1|_c':>8} {'ratio':>7} "
      f"{'ov_w':>7} {'ov_c':>7} {'d_ov':>7}")
for lo, hi in zip(EDGES[:-1], EDGES[1:]):
    kw = np.concatenate([np.abs(k[lo:hi]) for k in KW if len(k) > lo]) if any(len(k) > lo for k in KW) else np.zeros(0)
    kc = np.concatenate([np.abs(k[lo:hi]) for k in KC if len(k) > lo]) if any(len(k) > lo for k in KC) else np.zeros(0)
    ow = np.concatenate([o[lo:hi] for o in OW if len(o) > lo]) if any(len(o) > lo for o in OW) else np.zeros(0)
    oc = np.concatenate([o[lo:hi] for o in OC if len(o) > lo]) if any(len(o) > lo for o in OC) else np.zeros(0)
    if len(kw) < 100 or len(kc) < 100:
        continue
    print(f"{f'{lo}-{hi}':>14} {len(kw):>8} {len(kc):>8} {kw.mean():>8.4f} {kc.mean():>8.4f} "
          f"{kw.mean()/kc.mean():>7.3f} {ow.mean():>7.4f} {oc.mean():>7.4f} {ow.mean()-oc.mean():>+7.4f}")
