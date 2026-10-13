"""Score a rollout file (fork_rollout format) with a model: per-token log-prob of the actual token and its rank
in the model's next-token distribution.  Saves npz {lp<i>, rank<i>} and prints reachability statistics.
usage: reach_score.py --model M --rollouts roll.jsonl --out score.npz --label L
"""
import argparse, json, numpy as np, torch
from transformers import AutoModelForCausalLM
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--rollouts", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--label", default=""); ap.add_argument("--chunk", type=int, default=2048)
a = ap.parse_args()
recs = [json.loads(l) for l in open(a.rollouts)]
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
LP, RK = [], []
with torch.inference_mode():
    for i, r in enumerate(recs):
        p, s = r["prompt_ids"], r["resp_ids"]
        if not s: LP.append(np.zeros(0, np.float32)); RK.append(np.zeros(0, np.int32)); continue
        x = torch.tensor([p + s], device="cuda")
        logits = model(x, use_cache=False).logits[0][len(p) - 1: len(p) + len(s) - 1]
        y = torch.tensor(s, device="cuda")
        lp_l, rk_l = [], []
        for k0 in range(0, logits.shape[0], a.chunk):
            lg = logits[k0:k0 + a.chunk].float(); yy = y[k0:k0 + a.chunk]
            lsm = torch.log_softmax(lg, -1)
            lp_l.append(lsm.gather(1, yy[:, None])[:, 0].cpu())
            rk_l.append((lg > lg.gather(1, yy[:, None])).sum(-1).cpu() + 1)
        LP.append(torch.cat(lp_l).numpy().astype(np.float32)); RK.append(torch.cat(rk_l).numpy().astype(np.int32))
        if (i + 1) % 200 == 0: print(f"[score] {i+1}/{len(recs)}", flush=True)
np.savez_compressed(a.out, **{f"lp{i}": v for i, v in enumerate(LP)}, **{f"rank{i}": v for i, v in enumerate(RK)})
lp = np.concatenate(LP); rk = np.concatenate(RK); n = len(lp)
blocked_per_seq = np.array([(v < np.log(1e-4)).sum() for v in LP]); seq_ok = (blocked_per_seq == 0).mean()
print(f"[score] {a.label:<26} n_tok={n}  NLL={-lp.mean():.3f}  p50(logp)={np.median(lp):+.3f}  "
      f"rank>20={100*(rk>20).mean():.2f}%  rank>32={100*(rk>32).mean():.2f}%  rank>256={100*(rk>256).mean():.3f}%  "
      f"p<1e-4={100*(lp<np.log(1e-4)).mean():.3f}%  blocked/1k={1000*(lp<np.log(1e-4)).mean():.2f}  "
      f"seq_with_no_block={100*seq_ok:.1f}%  seq_logp_mean={np.mean([v.sum() for v in LP]):.0f}", flush=True)
