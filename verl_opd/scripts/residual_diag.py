"""Where does the entropy-shape gap survive training?  Post-mortem on a student/teacher top-32 dump pair
(fork_topk.py) on the student's own rollouts.  Positions are split by teacher entropy (H_T) and student entropy (H_S)
into 4 cells; for each cell we report share of positions, share of residual |dH| mass, mean dH (sign = student minus
teacher), and where the teacher's argmax sits in the student's ranking.  Tokens at the worst residual positions are
decoded and categorised.
usage: residual_diag.py --student S.npz --teacher T.npz --rollouts roll.jsonl --tok <model> --label L [--thr 0.5]
"""
import argparse, json, re, numpy as np
from collections import Counter
from transformers import AutoTokenizer
ap = argparse.ArgumentParser()
ap.add_argument("--student", required=True); ap.add_argument("--teacher", required=True); ap.add_argument("--rollouts", required=True)
ap.add_argument("--tok", required=True); ap.add_argument("--label", default=""); ap.add_argument("--thr", type=float, default=0.5)
a = ap.parse_args()
S, T = np.load(a.student), np.load(a.teacher); tok = AutoTokenizer.from_pretrained(a.tok)
recs = [json.loads(l) for l in open(a.rollouts)]
n = sum(1 for k in S.files if k.startswith("ids"))
HS, HT, RANK, QTOP_S, PTOP_T, TOKS, LPS, LQS = [], [], [], [], [], [], [], []
for i in range(n):
    ip, iq = S[f"ids{i}"], T[f"ids{i}"]
    if ip.shape[0] == 0: continue
    lp, lq = S[f"lp{i}"].astype(np.float32), T[f"lp{i}"].astype(np.float32)
    m = ip[:, :, None] == iq[:, None, :]
    rank = np.where(m[:, :, 0].any(-1), m[:, :, 0].argmax(-1) + 1, 99)      # teacher argmax's rank in student
    q_of_s_top = np.where(m[:, 0, :].any(-1), (lq * m[:, 0, :]).sum(-1), -12.0)  # teacher logp of student argmax
    p_of_t_top = np.where(m[:, :, 0].any(-1), (lp * m[:, :, 0]).sum(-1), -12.0)  # student logp of teacher argmax
    s = np.asarray(recs[i]["resp_ids"])[:ip.shape[0]]
    ms, mt = ip == s[:, None], iq == s[:, None]
    LPS.append(np.where(ms.any(-1), (lp * ms).sum(-1), -12.0)); LQS.append(np.where(mt.any(-1), (lq * mt).sum(-1), -12.0))
    HS.append(S[f"ent{i}"].astype(np.float32)); HT.append(T[f"ent{i}"].astype(np.float32))
    RANK.append(rank); QTOP_S.append(q_of_s_top); PTOP_T.append(p_of_t_top); TOKS.append(s)
HS, HT, RANK = map(np.concatenate, (HS, HT, RANK)); QTOP_S, PTOP_T, TOKS = map(np.concatenate, (QTOP_S, PTOP_T, TOKS))
LPS, LQS = np.concatenate(LPS), np.concatenate(LQS)
dH = HS - HT; absd = np.abs(dH); N = len(dH)
rho = np.corrcoef(HS, HT)[0, 1]
print(f"\n===== {a.label}   n={N}  rho={rho:.3f}  mean|dH|={absd.mean():.3f}  mean dH={dH.mean():+.3f}  H_S={HS.mean():.3f} H_T={HT.mean():.3f}")
thr = a.thr
cells = {"T conf / S conf": (HT < thr) & (HS < thr), "T conf / S unc": (HT < thr) & (HS >= thr),
         "T unc / S conf": (HT >= thr) & (HS < thr), "T unc / S unc": (HT >= thr) & (HS >= thr)}
print(f"{'cell (thr=%.1f nats)'%thr:22s} {'pos%':>6s} {'|dH| mass%':>10s} {'mean dH':>8s} {'Targmax rank in S: =1':>22s} {'<=4':>6s} {'<=20':>6s} {'>20':>6s} {'>32':>6s} {'p_S(Targmax)':>13s} {'q_T(Sargmax)':>13s} {'adv(sampled)':>13s}")
for name, c in cells.items():
    if c.sum() == 0: continue
    r = RANK[c]
    print(f"{name:22s} {100*c.mean():6.1f} {100*absd[c].sum()/absd.sum():10.1f} {dH[c].mean():+8.3f} {100*(r==1).mean():22.1f} {100*(r<=4).mean():6.1f} {100*(r<=20).mean():6.1f} {100*(r>20).mean():6.1f} {100*(r>32).mean():6.1f} {np.exp(PTOP_T[c]).mean():13.3f} {np.exp(QTOP_S[c]).mean():13.3f} {(LQS[c]-LPS[c]).mean():+13.3f}")
# unreachable share: positions where the teacher's argmax is outside the student's top-20
unreach = RANK > 20
print(f"teacher-argmax outside student top-20: {100*unreach.mean():.2f}% of positions, carrying {100*absd[unreach].sum()/absd.sum():.1f}% of |dH| mass;  outside top-32: {100*(RANK>32).mean():.2f}%")
# token categories at the worst 5% residual positions
def cat(tok_str):
    t = tok_str.strip()
    if t == "": return "space/newline"
    if re.fullmatch(r"[0-9.,]+", t): return "number"
    if re.fullmatch(r"[\\\\$^_{}()\\[\\]=+\\-*/<>|]+|\\\\[a-zA-Z]+", t): return "latex/operator"
    if t.lower() in {"wait","but","hmm","alternatively","however","actually","let","so","therefore","thus","check","verify","maybe","perhaps","okay","ok","first","now","then","again","no","yes","hold"}: return "connective"
    if re.fullmatch(r"[A-Za-z]+", t): return "word"
    return "other"
k = max(1, N // 20); worst = np.argsort(absd)[::-1][:k]
cw = Counter(); ca = Counter()
for j in worst: cw[cat(tok.decode([int(TOKS[j])]))] += absd[j]
for j in np.random.default_rng(0).choice(N, k, replace=False): ca[cat(tok.decode([int(TOKS[j])]))] += 1
tot = sum(cw.values()); print("worst-5% residual |dH| by token category (vs. random-position frequency):")
for c_, v in cw.most_common(): print(f"   {c_:16s} {100*v/tot:5.1f}%   (random: {100*ca[c_]/k:4.1f}%)")
ex = [tok.decode([int(TOKS[j])]) for j in worst[:40]]
print("   worst-40 tokens:", " | ".join(repr(e) for e in ex))
# per-sequence concentration of residual
seq_abs = []; off = 0
for i in range(n):
    L = S[f"ids{i}"].shape[0]
    if L: seq_abs.append(absd[off:off+L].mean()); off += L
seq_abs = np.array(seq_abs); print(f"per-sequence mean|dH|: p50={np.median(seq_abs):.3f} p90={np.percentile(seq_abs,90):.3f} max={seq_abs.max():.3f}; top-10% sequences carry {100*np.sort(seq_abs)[::-1][:len(seq_abs)//10].sum()/seq_abs.sum():.1f}% of the sum")
