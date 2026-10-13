"""Candidate *transferability* probes from a student/teacher top-K dump pair (fork_topk.py) on the
student's own rollouts, plus the sampled tokens from the rollout file.
  reach@K      share of the teacher's top-32 mass that sits on tokens the student already has in its top-K
               (transferable by re-weighting; the rest needs the student to lift tokens from deep in its tail)
  argmax rank  where the teacher's favourite token sits in the student's ranking (=1 / <=4 / >32)
  TV           per-position total-variation disagreement on the union of top-32 sets
  conc10       share of total TV carried by the 10 % most-disagreeing positions (sparse vs diffuse signal)
  lowH_share   E[TV | student entropy below median] / E[TV]  (disagreement at confident positions = "knowledge"
               the student lacks; at uncertain forks = "decisions" it can flip)
  adv          mean log q - log p on the sampled token (k1 signal), and its split by student entropy
usage: transfer_probe.py --student S.npz --teacher T.npz --rollouts roll.jsonl --label L
"""
import argparse, json, numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--student", required=True); ap.add_argument("--teacher", required=True)
ap.add_argument("--rollouts", required=True); ap.add_argument("--label", default="")
a = ap.parse_args()
S, T = np.load(a.student), np.load(a.teacher)
recs = [json.loads(l) for l in open(a.rollouts)]
n = sum(1 for k in S.files if k.startswith("ids"))
reach = {4: 0.0, 8: 0.0, 16: 0.0, 32: 0.0}; N = 0
r1 = r4 = r32 = 0; TV = []; HS = []; ADV = []; SAMP_IN_T32 = 0
for i in range(n):
    ip, iq = S[f"ids{i}"], T[f"ids{i}"]
    if ip.shape[0] == 0: continue
    lp, lq = S[f"lp{i}"].astype(np.float32), T[f"lp{i}"].astype(np.float32)
    p, q = np.exp(lp), np.exp(lq)
    m = ip[:, :, None] == iq[:, None, :]                      # (T,32s,32t)
    q_on_p = (q[:, None, :] * m).sum(-1)                      # teacher prob of each student-top token
    qmass = q.sum(-1)
    for K in reach: reach[K] += (q_on_p[:, :K].sum(-1) / qmass).sum()
    rank = np.where(m[:, :, 0].any(-1), m[:, :, 0].argmax(-1) + 1, 99)   # teacher argmax's rank in student
    r1 += (rank == 1).sum(); r4 += (rank <= 4).sum(); r32 += (rank > 32).sum()
    p_on_q = (p[:, :, None] * m).sum(1)                       # student prob of each teacher-top token
    tv = 0.5 * (np.abs(p - q_on_p).sum(-1) + np.where(m.any(1), 0, q).sum(-1))  # union TV (approx)
    TV.append(tv); HS.append(S[f"ent{i}"].astype(np.float32))
    # sampled token advantage
    s = np.asarray(recs[i]["resp_ids"])[:ip.shape[0]]
    ms = ip == s[:, None]; mt = iq == s[:, None]
    lps = np.where(ms.any(-1), (lp * ms).sum(-1), -12.0); lqs = np.where(mt.any(-1), (lq * mt).sum(-1), -12.0)
    SAMP_IN_T32 += mt.any(-1).sum()
    ADV.append(lqs - lps)
    N += ip.shape[0]
TV = np.concatenate(TV); HS = np.concatenate(HS); ADV = np.concatenate(ADV)
med = np.median(HS); lowH = HS < med
srt = np.sort(TV)[::-1]; conc10 = srt[: max(1, len(srt)//10)].sum() / srt.sum()
print(f"{a.label:<14} reach@4={reach[4]/N:.3f} @8={reach[8]/N:.3f} @16={reach[16]/N:.3f} @32={reach[32]/N:.3f} | "
      f"Targmax rank: =1 {r1/N:.3f} <=4 {r4/N:.3f} >32 {r32/N:.3f} | TV={TV.mean():.3f} conc10={conc10:.2f} "
      f"lowH_share={TV[lowH].mean()/TV.mean():.2f} corr(TV,H_S)={np.corrcoef(TV,HS)[0,1]:+.2f} | "
      f"adv={ADV.mean():+.3f} adv|lowH={ADV[lowH].mean():+.3f} adv|highH={ADV[~lowH].mean():+.3f} "
      f"|adv|={np.abs(ADV).mean():.3f} samp_in_T32={SAMP_IN_T32/N:.3f} n={N}")
