"""Representational-sufficiency probe: can a linear read-out of the STUDENT's hidden states predict the TEACHER's
per-position entropy (and whether the teacher is confident, H_T < thr) on the student's own rollouts?
One student forward pass (mean of a few late layers, float16), then ridge regression / logistic probe with a
held-out split by sequence.  Reports R^2 and AUC for each teacher, alongside the same probe for the student's OWN
entropy (ceiling) and a baseline that uses the student's entropy only (1 feature).
usage: hidden_probe.py --student M --rollouts roll.jsonl --student_dump S.npz --teachers name=T.npz [...] --layers -4 -3 -2 -1 --max_pos 1500
"""
import argparse, json, numpy as np, torch
from transformers import AutoModelForCausalLM
ap = argparse.ArgumentParser()
ap.add_argument("--student", required=True); ap.add_argument("--rollouts", required=True); ap.add_argument("--student_dump", required=True)
ap.add_argument("--teachers", nargs="+", required=True); ap.add_argument("--layers", type=int, nargs="+", default=[-4, -3, -2, -1])
ap.add_argument("--max_pos", type=int, default=1500); ap.add_argument("--n_seq", type=int, default=1000); ap.add_argument("--thr", type=float, default=0.5)
ap.add_argument("--out", default="")
ap.add_argument("--gap_thr", type=float, default=0.5)
ap.add_argument("--gap_ref", default="", help="entropy dump defining the gap positions (|target H - ref H| >= gap_thr); default = the student_dump entropy")
a = ap.parse_args()
recs = [json.loads(l) for l in open(a.rollouts)][: a.n_seq]
GRD = np.load(a.gap_ref) if a.gap_ref else None; GR = []
S = np.load(a.student_dump); T = {kv.split("=")[0]: np.load(kv.split("=")[1]) for kv in a.teachers}
model = AutoModelForCausalLM.from_pretrained(a.student, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
X, HS, HT, SEQ = [], [], {k: [] for k in T}, []
with torch.inference_mode():
    for i, r in enumerate(recs):
        p, s = r["prompt_ids"], r["resp_ids"][: a.max_pos]
        if not s: continue
        x = torch.tensor([p + s], device="cuda")
        hs = model(x, use_cache=False, output_hidden_states=True).hidden_states
        h = torch.stack([hs[l][0, len(p) - 1: len(p) + len(s) - 1] for l in a.layers]).mean(0)   # (L, d)
        h = h.float(); h = h / (h.norm(dim=-1, keepdim=True) + 1e-6)
        X.append(h.cpu().numpy().astype(np.float16)); L = len(s)
        HS.append(S[f"ent{i}"].astype(np.float32)[:L]); SEQ.append(np.full(L, i))
        if GRD is not None: GR.append(GRD[f"ent{i}"].astype(np.float32)[:L])
        for k in T: HT[k].append(T[k][f"ent{i}"].astype(np.float32)[:L])
        if (i + 1) % 200 == 0: print(f"[hid] {i+1}/{len(recs)}", flush=True)
X = np.concatenate(X).astype(np.float32); HS = np.concatenate(HS); SEQ = np.concatenate(SEQ); HT = {k: np.concatenate(v) for k, v in HT.items()}
GR = np.concatenate(GR) if GRD is not None else HS
print(f"[hid] features {X.shape}", flush=True)
rng = np.random.default_rng(0); seqs = np.unique(SEQ); test_seqs = set(rng.choice(seqs, len(seqs)//5, replace=False))
te = np.array([q in test_seqs for q in SEQ]); tr = ~te
def ridge(Xtr, ytr, Xte, lam=1.0):
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6; A = (Xtr - mu) / sd; B = (Xte - mu) / sd
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (ytr - ytr.mean())); return B @ w + ytr.mean()
def r2(y, yh): return 1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()
def auc(y, sc):
    o = np.argsort(sc); r = np.empty(len(sc)); r[o] = np.arange(1, len(sc) + 1); n1 = y.sum(); n0 = len(y) - n1
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
sub = rng.choice(np.where(tr)[0], min(400000, tr.sum()), replace=False)   # cap the ridge training set
Xtr, Xte = X[sub], X[te]; Hs_feat_tr, Hs_feat_te = HS[sub][:, None], HS[te][:, None]
print(f"\n{'target':22s} {'R2 hidden':>10s} {'R2 H_S only':>12s} {'R2 hidden+H_S':>14s} | {'AUC conf hidden':>15s} {'AUC conf H_S':>13s}")
targets = {"student H (ceiling)": HS, **{f"teacher {k} H": v for k, v in HT.items()}}
out = {}
for name, y in targets.items():
    yh_h = ridge(Xtr, y[sub], Xte); yh_s = ridge(Hs_feat_tr, y[sub], Hs_feat_te); yh_b = ridge(np.c_[Xtr, Hs_feat_tr], y[sub], np.c_[Xte, Hs_feat_te])
    yt = y[te]; conf = yt < a.thr
    out[name] = (r2(yt, yh_h), r2(yt, yh_s), r2(yt, yh_b), auc(conf, -yh_h), auc(conf, -yh_s))
    print(f"{name:22s} {out[name][0]:10.3f} {out[name][1]:12.3f} {out[name][2]:14.3f} | {out[name][3]:15.3f} {out[name][4]:13.3f}", flush=True)
# residual structure: predict the teacher-student entropy difference (what OPD has to move)
print(f"\n{'target: dH = H_T - H_S':22s} {'R2 hidden':>10s} {'R2 H_S only':>12s}")
for k, v in HT.items():
    y = v - HS; yh_h = ridge(Xtr, y[sub], Xte); yh_s = ridge(Hs_feat_tr, y[sub], Hs_feat_te)
    print(f"{('dH '+k):22s} {r2(y[te], yh_h):10.3f} {r2(y[te], yh_s):12.3f}", flush=True)
# ---- gap-restricted view: only positions where teacher and student entropy disagree by >= gap_thr ----
print(f"\n[gap view: |H_T - H_ref| >= {a.gap_thr}, ref={'student' if GRD is None else a.gap_ref.split('/')[-1]}]  {'target':16s} {'gap pos%':>8s} {'R2 hidden(all-fit)':>18s} {'R2 hidden(gap-fit)':>18s} {'R2 H_S only(gap-fit)':>20s} {'AUC conf hidden':>15s}")
for k, v in HT.items():
    g = np.abs(v - GR) >= a.gap_thr; gtr = g & tr; gte = g & te
    if gte.sum() < 100 or gtr.sum() < 100: print(f"   {k}: too few gap positions"); continue
    y = v
    yh_all = ridge(Xtr, y[sub], X[gte])                                    # readout fitted on all positions, tested on gap
    gs = rng.choice(np.where(gtr)[0], min(400000, gtr.sum()), replace=False)
    yh_gap = ridge(X[gs], y[gs], X[gte]); yh_hs = ridge(HS[gs][:, None], y[gs], HS[gte][:, None])
    yt = y[gte]; conf = yt < a.thr
    out[f"gap {k}"] = (r2(yt, yh_all), r2(yt, yh_gap), r2(yt, yh_hs), auc(conf, -yh_gap) if 0 < conf.sum() < len(conf) else float("nan"))
    print(f"{'':30s} {k:16s} {100*g.mean():8.2f} {out[f'gap {k}'][0]:18.3f} {out[f'gap {k}'][1]:18.3f} {out[f'gap {k}'][2]:20.3f} {out[f'gap {k}'][3]:15.3f}", flush=True)
if a.out: json.dump({k: list(map(float, v)) for k, v in out.items()}, open(a.out, "w"))
print("HIDDEN_PROBE_DONE", flush=True)
