"""Qwen3-1.7B-Base <- Qwen3-4B-Base-GRPO (k1+PG, DAPO-17k): left = online rho and overlap@16 vs step (10-step MA, steps 1-200,
one shared y-axis so the flat overlap curve is visibly flat); right = six-benchmark average at the saved checkpoints, teacher's own accuracy as a dashed line."""
import csv, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
W, MAXSTEP = 10, 200
def load(p):
    d = {}
    for r in csv.DictReader(open(p)):
        for k, v in r.items():
            try: d.setdefault(k, []).append(float(v))
            except (ValueError, TypeError): d.setdefault(k, []).append(np.nan)
    return {k: np.array(v) for k, v in d.items()}
def ma(y, w=W):
    out = np.convolve(np.nan_to_num(y, nan=np.nanmean(y)), np.ones(w) / w, mode="valid")
    return np.concatenate([np.full(w - 1, np.nan), out])
d = load("probe_q3b17_BaseGRPO_k1pg.csv"); m = d["step"] <= MAXSTEP
ACC = {0: 5.2, 50: 10.4, 100: 10.8, 150: 10.6, 200: 11.3}; TEACHER = 22.4   # step 0 = untrained Base (9-25 eval)
C_R, C_O, C_A = "#1f77b4", "#ff7f0e", "#d62728"
fig, (a, b) = plt.subplots(1, 2, figsize=(9.5, 3.6))
a.plot(d["step"][m], ma(d["rho"])[m], c=C_R, lw=1.8, label=r"ESC $\rho$(student, teacher)")
a.plot(d["step"][m], ma(d["overlap16"])[m], c=C_O, lw=1.8, label="Overlap@16")
a.set_ylim(0.5, 0.9); a.set_ylabel("value")
a.legend(frameon=False, fontsize=9, loc="lower right")
a.set_xlabel("Training Step"); a.set_title("Shape and candidate-set alignment", fontsize=11)
a.grid(alpha=.25); a.spines[["top", "right"]].set_visible(False)
xs = sorted(ACC); ys = [ACC[k] for k in xs]
b.plot(xs, ys, c=C_A, lw=1.8, marker="o", ms=5, label="Qwen3-1.7B-Base ← Qwen3-4B-Base-GRPO")
b.axhline(TEACHER, c="k", lw=1.2, ls="--"); b.text(198, TEACHER + 0.3, "Qwen3-4B-Base-GRPO (teacher)", fontsize=8, ha="right", va="bottom")
b.set_xticks([0, 50, 100, 150, 200]); b.set_ylim(0, 26)
b.set_xlabel("Training Step"); b.set_ylabel("Average Performance"); b.set_title("Average Performance", fontsize=11)
b.legend(frameon=False, fontsize=8.5, loc="center right"); b.grid(alpha=.25); b.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
out = "paper_figs/fig_base_grpo"; fig.savefig(out + ".png", dpi=200); fig.savefig(out + ".pdf")
print("saved", out, f"rho {d['rho'][m][0]:.3f}->{np.nanmean(d['rho'][m][-10:]):.3f}  ov16 {d['overlap16'][m][0]:.3f}->{np.nanmean(d['overlap16'][m][-10:]):.3f}")
