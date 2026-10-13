"""Overlap metrics between a student and a teacher top-k dump (fork_topk.py) on the SAME
rollouts, following arXiv:2604.13016 Sec. 2.3 (their default k=16):
  overlap ratio        E_t |S_p ∩ S_q| / k
  overlap-token adv    E_t mean_{v in I} pbar(v) (log qbar(v) - log pbar(v)),  pbar/qbar renormalised on I
  top-k mass           E_t sum_{S_p} p,  E_t sum_{S_q} q
  overlap mass         E_t sum_{I} p / sum_{S_p} p  (share of the student's top-k mass that sits on I)
  |dH|                 |mean H_p - mean H_q|
Usage: overlap_metrics.py --student S.npz --teacher T.npz --label L [--k 16 32]
"""
import argparse, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--student", required=True); ap.add_argument("--teacher", required=True)
ap.add_argument("--label", default=""); ap.add_argument("--k", type=int, nargs="+", default=[16, 32])
a = ap.parse_args()
S, T = np.load(a.student), np.load(a.teacher)
n = sum(1 for key in S.files if key.startswith("ids"))

for k in a.k:
    ov, adv, mp, mq, om, Hp, Hq, N = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
    for i in range(n):
        ip, iq = S[f"ids{i}"][:, :k], T[f"ids{i}"][:, :k]
        if ip.shape[0] == 0: continue
        lp, lq = S[f"lp{i}"][:, :k].astype(np.float32), T[f"lp{i}"][:, :k].astype(np.float32)
        T_, _ = ip.shape
        # membership of each student top-k token in the teacher top-k set, and its teacher logprob
        m = ip[:, :, None] == iq[:, None, :]                      # (T, k, k)
        inter = m.any(-1)                                         # (T, k): student token in overlap
        lq_on_p = np.where(inter, (lq[:, None, :] * m).sum(-1), 0.0)  # teacher logp of overlap tokens
        cnt = inter.sum(-1)                                       # |I| per position
        ov += (cnt / k).sum(); N += T_
        pk, qk = np.exp(lp), np.exp(lq)
        mp += pk.sum(-1).sum(); mq += qk.sum(-1).sum()
        om += (np.where(inter, pk, 0).sum(-1) / pk.sum(-1)).sum()
        has = cnt > 0
        if has.any():
            pI = np.where(inter, pk, 0)[has]; qI = np.where(inter, np.exp(lq_on_p), 0)[has]
            pbar = pI / pI.sum(-1, keepdims=True); qbar = qI / qI.sum(-1, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                A = np.where(inter[has], pbar * (np.log(qbar) - np.log(pbar)), 0.0)
            adv += (A.sum(-1) / cnt[has]).sum()
        Hp += S[f"ent{i}"].astype(np.float32).sum(); Hq += T[f"ent{i}"].astype(np.float32).sum()
    print(f"{a.label:<22} k={k:<3} overlap={100*ov/N:5.1f}%  adv={adv/N:+.4f}  "
          f"mass_p={mp/N:.3f} mass_q={mq/N:.3f} overlap_mass_p={om/N:.3f}  |dH|={abs(Hp-Hq)/N:.3f}  n_tok={N}")
