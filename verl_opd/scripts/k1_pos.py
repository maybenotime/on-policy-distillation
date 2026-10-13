"""Per-position k1 and top-k overlap between a student and a teacher on a fixed
student rollout, so |k1| can be compared at MATCHED POSITIONS across two datasets
(the raw per-trajectory mean is confounded with trajectory length).

Both models are resident at once and share the forward loop, so this costs one pass.
Positions are capped at --max_pos; that cap is what makes the two subsets comparable.
"""
import argparse, json
import numpy as np, torch
from transformers import AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--student", required=True)
ap.add_argument("--teacher", required=True)
ap.add_argument("--rollouts", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--k", type=int, default=16)
ap.add_argument("--max_pos", type=int, default=2048)
ap.add_argument("--chunk", type=int, default=256)
a = ap.parse_args()

recs = [json.loads(l) for l in open(a.rollouts)]
load = lambda m: AutoModelForCausalLM.from_pretrained(
    m, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
S, T = load(a.student), load(a.teacher)

k1_out, ov_out = [], []
with torch.inference_mode():
    for i, r in enumerate(recs):
        p, s = r["prompt_ids"], r["resp_ids"]
        s = s[: a.max_pos]
        if len(s) == 0:
            k1_out.append(np.zeros(0, np.float16)); ov_out.append(np.zeros(0, np.float16)); continue
        x = torch.tensor([p + s], device="cuda")
        lo, hi = len(p) - 1, len(p) + len(s) - 1
        sl_s = S(x, use_cache=False).logits[0][lo:hi]
        sl_t = T(x, use_cache=False).logits[0][lo:hi]
        y = torch.tensor(s, device="cuda")
        K1, OV = [], []
        for c in range(0, sl_s.shape[0], a.chunk):
            ls = torch.log_softmax(sl_s[c:c + a.chunk].float(), -1)
            lt = torch.log_softmax(sl_t[c:c + a.chunk].float(), -1)
            yy = y[c:c + a.chunk].unsqueeze(-1)
            K1.append((ls.gather(-1, yy) - lt.gather(-1, yy)).squeeze(-1).cpu())
            ts = ls.topk(a.k, -1).indices
            tt = lt.topk(a.k, -1).indices
            # |top-k(student) & top-k(teacher)| / k, per position
            OV.append((ts.unsqueeze(-1) == tt.unsqueeze(-2)).any(-1).sum(-1).float().div(a.k).cpu())
        k1_out.append(torch.cat(K1).numpy().astype(np.float16))
        ov_out.append(torch.cat(OV).numpy().astype(np.float16))
        if (i + 1) % 100 == 0:
            print(f"[k1pos] {i+1}/{len(recs)}", flush=True)

np.savez_compressed(a.out, **{f"k1_{i}": v for i, v in enumerate(k1_out)},
                    **{f"ov_{i}": v for i, v in enumerate(ov_out)})
print(f"[k1pos] wrote {a.out} n={len(k1_out)}", flush=True)
