"""Stage 2 — per-position next-token entropy of `--model` on a fixed trajectory.

Run once with the student and once with the teacher on the SAME rollout file.
Entropy needs the full vocabulary distribution, so this is a HF forward pass;
vLLM's top-k logprobs would truncate the tail and bias H downward.
"""
import argparse, json
import numpy as np, torch
from transformers import AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--rollouts", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--chunk", type=int, default=512)
a = ap.parse_args()

recs = [json.loads(l) for l in open(a.rollouts)]
# no device_map= : that path requires `accelerate`, which this venv does not have
model = AutoModelForCausalLM.from_pretrained(
    a.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()

out = []
with torch.inference_mode():
    for i, r in enumerate(recs):
        p, s = r["prompt_ids"], r["resp_ids"]
        if len(s) == 0:
            out.append(np.zeros(0, dtype=np.float16)); continue
        ids = torch.tensor([p + s], device="cuda")
        logits = model(ids, use_cache=False).logits[0]
        # position t-1 holds the distribution that produced response token t
        sl = logits[len(p) - 1: len(p) + len(s) - 1]
        H = []
        for k in range(0, sl.shape[0], a.chunk):           # chunked: V is ~152k
            lp = torch.log_softmax(sl[k:k + a.chunk].float(), -1)
            H.append((-(lp.exp() * lp).sum(-1)).cpu())
        out.append(torch.cat(H).numpy().astype(np.float16))
        if (i + 1) % 100 == 0:
            print(f"[ent] {i+1}/{len(recs)}", flush=True)

np.savez_compressed(a.out, **{str(i): h for i, h in enumerate(out)})
print(f"[ent] wrote {a.out}  n={len(out)}", flush=True)
