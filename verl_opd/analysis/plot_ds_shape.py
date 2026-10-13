"""R1-Distill-1.5B <- (student preferences ⊕ JustRL shape), `ds15b_tpStu_shapeJustRL` (stopped by the node crash at step 145):
left = rho(student, JustRL = shape source) and overlap@16(student, base student = preference source) on one axis (10-step MA); right = six-benchmark
average (mean-6) at the saved checkpoints (step 0 = base student = preference source), JustRL's own accuracy as a dashed line.
Same layout as plot_run4.py."""
import csv, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
W, MAXSTEP = 10, 100   # the run died at step 145; show steps 1-100 (last evaluated checkpoint)
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
d = load("ser_tpStu_200.csv"); m = d["step"] <= MAXSTEP
BASE = 25.5; ACC = {0: BASE, 50: BASE + 2.4, 100: BASE + 2.2}; T_JR = 40.1
C_R, C_O, C_A = "#1f77b4", "#ff7f0e", "#2ca02c"
fig, (a, b) = plt.subplots(1, 2, figsize=(9.5, 3.6))
a.plot(d["step"][m], ma(d["actor/tp/rho_s_B"])[m], c=C_R, lw=1.8, label=r"ESC $\rho$(student, JustRL)  [shape source]")
a.plot(d["step"][m], ma(d["actor/tp/overlap16_sA"])[m], c=C_O, lw=1.8, label="Overlap@16(student, base student)  [preference source]")
a.set_xlim(0, 100); a.set_ylim(0.65, 1.0); a.set_ylabel("value"); a.set_xlabel("Training Step")
a.set_title("Shape and candidate-set alignment", fontsize=11)
a.legend(frameon=False, fontsize=8.5, loc="lower right"); a.grid(alpha=.25); a.spines[["top", "right"]].set_visible(False)
xs = sorted(ACC); ys = [ACC[k] for k in xs]
b.plot(xs, ys, c=C_A, lw=1.8, marker="o", ms=5, label="R1-Distill-1.5B ← student prefs ⊕ JustRL shape")
b.axhline(T_JR, c="#d62728", lw=1.2, ls="--"); b.text(99, T_JR + 0.4, "JustRL-1.5B (teacher)", color="#d62728", fontsize=8, ha="right", va="bottom")
b.axhline(BASE, c="#7f7f7f", lw=1.2, ls="--"); b.text(99, BASE + 0.4, "R1-Distill-1.5B base (preference source)", color="#7f7f7f", fontsize=8, ha="right", va="bottom")
b.set_xticks([0, 50, 100]); b.set_xlim(-3, 103); b.set_ylim(20, 44)
b.set_xlabel("Training Step"); b.set_ylabel("Average Performance"); b.set_title("Average Performance", fontsize=11)
b.legend(frameon=False, fontsize=8, loc="lower right"); b.grid(alpha=.25); b.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
out = "paper_figs/fig_ds_stuprefs_justrlshape"; fig.savefig(out + ".png", dpi=200); fig.savefig(out + ".pdf")
print("saved", out, f"rho_B {d['actor/tp/rho_s_B'][m][0]:.3f}->{np.nanmean(d['actor/tp/rho_s_B'][m][-10:]):.3f}  ov16_A {d['actor/tp/overlap16_sA'][m][0]:.3f}->{np.nanmean(d['actor/tp/overlap16_sA'][m][-10:]):.3f}  last step {int(d['step'][m][-1])}")
