"""rho and overlap@16 curves for the seven runs of the transplant tables (user, 9-25).
Row 1 Qwen3-1.7B: (a) rho vs Qwen3-4B-RL-Math, (b) rho vs Qwen3-4B, (c) overlap@16 vs the preference source.
Row 2 DeepSeek-R1-Distill-1.5B: (a) rho vs JustRL, (b) overlap@16 vs JustRL.
Direct runs = solid, transplants = dashed; 10-step moving average, steps 1-200.
Column sources: direct runs -> probe/rho, probe/overlap16; transplants -> tp/rho_s_A (preference source), tp/rho_s_B (shape
source), tp/overlap16_sA; run 3's tp overlap columns were not logged, its probe/overlap16 is the overlap with the synthetic
teacher, whose top-k equals A's (temperature preserves ranking)."""
import csv, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
W, MAXSTEP = 10, 200
def load(p):
    rows = {}
    for r in csv.DictReader(open(p)):
        rows[int(float(r["step"]))] = r            # last occurrence wins (resumed logs repeat steps)
    d = {}
    for st in sorted(rows):
        for k, v in rows[st].items():
            try: d.setdefault(k, []).append(float(v))
            except (ValueError, TypeError): d.setdefault(k, []).append(np.nan)
    return {k: np.array(v) for k, v in d.items()}
def ma(y, w=W):
    y = np.asarray(y, float)
    if np.all(np.isnan(y)): return y
    out = np.convolve(np.nan_to_num(y, nan=np.nanmean(y)), np.ones(w) / w, mode="valid")
    return np.concatenate([np.full(w - 1, np.nan), out])
def put(ax, d, col, label, c, ls):
    m = d["step"] <= MAXSTEP
    ax.plot(d["step"][m], ma(d[col])[m], c=c, lw=1.7, ls=ls, label=label)
Q_M5 = load("probe_q3i17_fromK4Bmath500.csv"); Q_4B = load("probe_q3i17_from4B.csv")
R3 = load("ser_q3_tpA4B_shapeM500.csv"); R4 = load("ser_q3_tpM500_shapeA4B_full.csv")
D_JR = load("probe_ds15b_fromJustRL.csv"); D_SH = load("ser_tpStu_200.csv"); D_LV = load("ser_live_jrstu_full.csv")
C = {"m5": "#d62728", "4b": "#7f7f7f", "r3": "#ff7f0e", "r4": "#9467bd", "jr": "#1f77b4", "sh": "#2ca02c", "lv": "#9467bd"}
fig, ax = plt.subplots(2, 3, figsize=(13.5, 6.2))
# ---- Qwen3 row
a = ax[0, 0]
put(a, Q_M5, "rho", "RL-Math ⊕ own shape (direct)", C["m5"], "-")
put(a, R4, "actor/tp/rho_s_A", "RL-Math ⊕ Qwen3-4B shape", C["r4"], "--")
put(a, R3, "actor/tp/rho_s_B", "Qwen3-4B ⊕ RL-Math shape", C["r3"], "--")
a.set_title(r"ESC $\rho$ vs Qwen3-4B-RL-Math", fontsize=11)
a = ax[0, 1]
put(a, Q_4B, "rho", "Qwen3-4B ⊕ own shape (direct)", C["4b"], "-")
put(a, R3, "actor/tp/rho_s_A", "Qwen3-4B ⊕ RL-Math shape", C["r3"], "--")
put(a, R4, "actor/tp/rho_s_B", "RL-Math ⊕ Qwen3-4B shape", C["r4"], "--")
a.set_title(r"ESC $\rho$ vs Qwen3-4B", fontsize=11)
a = ax[0, 2]
put(a, Q_M5, "overlap16", "RL-Math ⊕ own shape (direct)", C["m5"], "-")
put(a, Q_4B, "overlap16", "Qwen3-4B ⊕ own shape (direct)", C["4b"], "-")
put(a, R4, "actor/tp/overlap16_sA", "RL-Math ⊕ Qwen3-4B shape", C["r4"], "--")
put(a, R3, "actor/probe/overlap16", "Qwen3-4B ⊕ RL-Math shape", C["r3"], "--")
a.set_title("Overlap@16 vs preference source", fontsize=11)
# ---- DeepSeek row
a = ax[1, 0]
put(a, D_JR, "rho", "JustRL ⊕ own shape (direct)", C["jr"], "-")
put(a, D_LV, "actor/tp/rho_s_A", "JustRL ⊕ student shape (live)", C["lv"], "--")
put(a, D_SH, "actor/tp/rho_s_B", "student ⊕ JustRL shape", C["sh"], "--")
a.set_title(r"ESC $\rho$ vs JustRL-1.5B", fontsize=11)
a = ax[1, 1]
put(a, D_JR, "overlap16", "JustRL ⊕ own shape (direct)", C["jr"], "-")
put(a, D_LV, "actor/tp/overlap16_sA", "JustRL ⊕ student shape (live)", C["lv"], "--")
put(a, D_SH, "actor/tp/overlap16_sB", "student ⊕ JustRL shape", C["sh"], "--")
a.set_title("Overlap@16 vs JustRL-1.5B", fontsize=11)
ax[1, 2].axis("off")
ax[0, 0].set_ylabel("Qwen3-1.7B", fontsize=11); ax[1, 0].set_ylabel("R1-Distill-1.5B", fontsize=11)
for a in ax.ravel():
    if not a.axison: continue
    a.grid(alpha=.25); a.spines[["top", "right"]].set_visible(False); a.tick_params(labelsize=9)
    a.legend(frameon=False, fontsize=8, loc="lower right"); a.set_xlabel("Training Step", fontsize=10)
fig.tight_layout()
out = "paper_figs/fig_tp_curves"; fig.savefig(out + ".png", dpi=200); fig.savefig(out + ".pdf"); print("saved", out)
