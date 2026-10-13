"""Calibrate the global temperature for the uniform-sharpening control: find T such that mean_t H(softmax(z_A/T)) over the
student rollouts equals mean_t H_B (teacher B's full-vocab entropy on the same positions).
usage: tp_uniformT.py --model_a <A> --rollouts <jsonl> --ent_b <B entropy npz from fork_topk> --n 300 --out T.txt"""
import argparse, json, math, numpy as np, torch
from transformers import AutoModelForCausalLM
ap = argparse.ArgumentParser()
ap.add_argument("--model_a", required=True); ap.add_argument("--rollouts", required=True); ap.add_argument("--ent_b", required=True)
ap.add_argument("--n", type=int, default=300); ap.add_argument("--out", required=True); ap.add_argument("--chunk", type=int, default=1024)
a = ap.parse_args()
recs = [json.loads(l) for l in open(a.rollouts)][: a.n]
EB = np.load(a.ent_b)
model = AutoModelForCausalLM.from_pretrained(a.model_a, dtype=torch.bfloat16, attn_implementation="flash_attention_2").cuda().eval()
W = model.lm_head.weight.detach().float()
grid = [math.exp(x) for x in np.linspace(math.log(0.3), math.log(3.0), 25)]
sums = np.zeros(len(grid)); cnt = 0; hb_sum = 0.0
with torch.inference_mode():
    for i, r in enumerate(recs):
        p, s = r["prompt_ids"], r["resp_ids"]
        if len(s) == 0: continue
        hb = EB[f"ent{i}"].astype(np.float32); T = min(len(s), len(hb))
        ids = torch.tensor([p + s], device="cuda")
        hs = model.model(input_ids=ids, use_cache=False).last_hidden_state[0][len(p) - 1: len(p) + T - 1]
        for c in range(0, T, a.chunk):
            z = hs[c:c + a.chunk].float() @ W.T
            zc = z - z.max(-1, keepdim=True).values
            for g, t in enumerate(grid):
                lp = torch.log_softmax(zc / t, -1); sums[g] += float((-(lp.exp() * lp).sum(-1)).sum())
        hb_sum += float(hb[:T].sum()); cnt += T
        if (i + 1) % 50 == 0: print(f"[uT] {i+1}/{len(recs)}", flush=True)
meanH = sums / cnt; target = hb_sum / cnt
j = int(np.argmin(np.abs(meanH - target)))
# linear interpolation in log T between the two nearest grid points
lo, hi = (j - 1, j) if meanH[j] > target and j > 0 else (j, min(j + 1, len(grid) - 1))
if meanH[hi] != meanH[lo]:
    w = (target - meanH[lo]) / (meanH[hi] - meanH[lo]); T_star = math.exp(math.log(grid[lo]) + w * (math.log(grid[hi]) - math.log(grid[lo])))
else:
    T_star = grid[j]
print(f"positions {cnt}  mean H_A(T=1) {meanH[np.argmin(np.abs(np.array(grid)-1))]:.4f}  target mean H_B {target:.4f}  ->  T* = {T_star:.4f}")
print("grid:", " ".join(f"{t:.2f}:{h:.3f}" for t, h in zip(grid, meanH)))
open(a.out, "w").write(f"{T_star:.4f}\n")
