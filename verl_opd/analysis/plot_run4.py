"""Qwen3-1.7B <- (RL-Math preferences ⊕ Qwen3-4B shape), run 4: left = rho(student, Qwen3-4B = shape source) and
overlap@16(student, RL-Math = preference source) on one axis (10-step MA, steps 1-200); right = six-benchmark average at
the saved checkpoints (step 0 = base student), teachers' own accuracy as dashed lines."""
import csv, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
W, MAXSTEP = 10, 200
def load(p):
    rows = {}
    for r in csv.DictReader(open(p)): rows[int(float(r["step"]))] = r
    d = {}
    for st in sorted(rows):
        for k, v in rows[st].items():
            try: d.setdefault(k, []).append(float(v))
            except (ValueError, TypeError): d.setdefault(k, []).append(np.nan)
    return {k: np.array(v) for k, v in d.items()}
def ma(y, w=W):
    out = np.convolve(np.nan_to_num(y, nan=np.nanmean(y)), np.ones(w) / w, mode="valid")
    return np.concatenate([np.full(w - 1, np.nan), out])
d = load("ser_q3_tpM500_shapeA4B_full.csv"); m = d["step"] <= MAXSTEP
ACC = {0: 13.0, 50: 25.4, 100: 26.0, 200: 25.3}; T_M5, T_4B = 40.3, 22.7
C_R, C_O, C_A = "#1f77b4", "#ff7f0e", "#9467bd"
fig, (a, b) = plt.subplots(1, 2, figsize=(9.5, 3.6))
a.plot(d["step"][m], ma(d["actor/tp/rho_s_B"])[m], c=C_R, lw=1.8, label=r"ESC $\rho$(student, Qwen3-4B)  [shape source]")
a.plot(d["step"][m], ma(d["actor/tp/overlap16_sA"])[m], c=C_O, lw=1.8, label="Overlap@16(student, RL-Math)  [preference source]")
a.set_ylim(0.65, 0.95); a.set_ylabel("value"); a.set_xlabel("Training Step")
a.set_title("Shape and candidate-set alignment", fontsize=11)
a.legend(frameon=False, fontsize=8.5, loc="lower right"); a.grid(alpha=.25); a.spines[["top", "right"]].set_visible(False)
xs = sorted(ACC); ys = [ACC[k] for k in xs]
b.plot(xs, ys, c=C_A, lw=1.8, marker="o", ms=5, label="Qwen3-1.7B ← RL-Math prefs ⊕ Qwen3-4B shape")
b.axhline(T_M5, c="#d62728", lw=1.2, ls="--"); b.text(198, T_M5 + 0.4, "Qwen3-4B-RL-Math (teacher)", color="#d62728", fontsize=8, ha="right", va="bottom")
b.axhline(T_4B, c="#7f7f7f", lw=1.2, ls="--"); b.text(198, T_4B + 0.4, "Qwen3-4B (teacher)", color="#7f7f7f", fontsize=8, ha="right", va="bottom")
b.set_xticks([0, 50, 100, 150, 200]); b.set_ylim(10, 44)
b.set_xlabel("Training Step"); b.set_ylabel("Average Performance"); b.set_title("Average Performance", fontsize=11)
b.legend(frameon=False, fontsize=8, loc="lower right"); b.grid(alpha=.25); b.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
out = "paper_figs/fig_run4_m500prefs_4bshape"; fig.savefig(out + ".png", dpi=200); fig.savefig(out + ".pdf")
print("saved", out, f"rho_B {d['actor/tp/rho_s_B'][m][0]:.3f}->{np.nanmean(d['actor/tp/rho_s_B'][m][-10:]):.3f}  ov16_A {d['actor/tp/overlap16_sA'][m][0]:.3f}->{np.nanmean(d['actor/tp/overlap16_sA'][m][-10:]):.3f}")
