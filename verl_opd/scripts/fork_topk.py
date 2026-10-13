"""Per-position top-k ids + log-probs of `--model` on a fixed trajectory (fork_entropy.py
sibling). Feeds overlap_metrics.py, which implements the dynamic metrics of
"Rethinking On-Policy Distillation" (arXiv:2604.13016, Sec. 2.3): overlap ratio,
overlap-token advantage, top-k mass. Ids are stored sorted by descending prob, so any
k' <= k can be sliced later. Output is large (T x k x 6 bytes) -> keep on /localscratch.
"""
import argparse, json
import numpy as np, torch
from transformers import AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--rollouts", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--k", type=int, default=32)
ap.add_argument("--chunk", type=int, default=1024)
a = ap.parse_args()

recs = [json.loads(l) for l in open(a.rollouts)]
model = AutoModelForCausalLM.from_pretrained(
    a.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()

ids_out, lp_out, ent_out = [], [], []
with torch.inference_mode():
    for i, r in enumerate(recs):
        p, s = r["prompt_ids"], r["resp_ids"]
        if len(s) == 0:
            ids_out.append(np.zeros((0, a.k), np.int32)); lp_out.append(np.zeros((0, a.k), np.float16))
            ent_out.append(np.zeros(0, np.float16)); continue
        x = torch.tensor([p + s], device="cuda")
        logits = model(x, use_cache=False).logits[0]
        sl = logits[len(p) - 1: len(p) + len(s) - 1]
        I, L, H = [], [], []
        for k0 in range(0, sl.shape[0], a.chunk):
            lp = torch.log_softmax(sl[k0:k0 + a.chunk].float(), -1)
            H.append((-(lp.exp() * lp).sum(-1)).cpu())
            t = torch.topk(lp, a.k, dim=-1)
            I.append(t.indices.cpu()); L.append(t.values.cpu())
        ids_out.append(torch.cat(I).numpy().astype(np.int32))
        lp_out.append(torch.cat(L).numpy().astype(np.float16))
        ent_out.append(torch.cat(H).numpy().astype(np.float16))
        if (i + 1) % 100 == 0:
            print(f"[topk] {i+1}/{len(recs)}", flush=True)

np.savez_compressed(a.out, **{f"ids{i}": v for i, v in enumerate(ids_out)},
                    **{f"lp{i}": v for i, v in enumerate(lp_out)},
                    **{f"ent{i}": v for i, v in enumerate(ent_out)})
print(f"[topk] wrote {a.out}  n={len(recs)} k={a.k}", flush=True)
