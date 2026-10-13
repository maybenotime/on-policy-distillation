"""Paper figures: Fig 1(b) transplant bars, Fig 1(c) pre-rho vs gain, Fig 2 dynamics (rho / overlap@16)."""
import csv, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "legend.fontsize": 7.5, "pdf.fonttype": 42})
OUT = "paper_figs/"

def save(fig, name):
    fig.savefig(OUT + name + ".pdf", bbox_inches="tight"); fig.savefig(OUT + name + ".png", dpi=200, bbox_inches="tight")
    print("wrote", name)

# ---------------- Fig 1(b): transplant bars ----------------
fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.4), gridspec_kw={"width_ratios": [1, 1]})
def bars(a, labels, vals, pending, title, base):
    x = np.arange(len(labels)); cols = ["#9ecae1", "#3182bd", "#fdae6b", "#e6550d"]
    for i, (v, p) in enumerate(zip(vals, pending)):
        a.bar(x[i], 0 if p else v, color=cols[i], edgecolor="k", lw=.6, hatch="//" if p else None, alpha=.35 if p else 1)
        a.text(x[i], (v if not p else 0) + .25, "pending" if p else f"{v:+.1f}", ha="center", va="bottom", fontsize=7.5)
    a.axhline(0, c="k", lw=.8); a.set_xticks(x); a.set_xticklabels(labels, fontsize=7.5)
    a.set_title(title); a.set_ylabel(r"$\Delta$ mean-7 (points)"); a.spines[["top", "right"]].set_visible(False)
bars(ax[0], ["A", "A$\\oplus$shape(B)", "B", "B$\\oplus$shape(A)"], [0.7, 3.8, 14.5, 12.7], [0, 0, 0, 0],
     "Qwen3-1.7B\nA = Qwen3-4B,  B = Math-Step500", 12.8)
ax[0].set_ylim(-1, 17)
bars(ax[1], ["A", "A$\\oplus$shape(B)", "B", "B$\\oplus$shape(A)"], [0.0, 1.8, 3.0, 0.0], [0, 0, 0, 1],
     "DS-R1-Distill-1.5B\nA = R1-Distill-7B,  B = JustRL-1.5B", 25.6)
ax[1].set_ylim(-1, 5)
fig.text(0.5, -0.04, "same token preferences (argmax / top-k of the preference model), transplanted per-position entropy; all at step 50",
         ha="center", fontsize=7.5)
save(fig, "fig1b_transplant_bars")

# ---------------- Fig 1(c): pre-rho vs gain ----------------
DS = [("DeepScaleR-1.5B", .970, 6.0), ("R1-Distill-7B", .940, 0.0), ("Skywork-OR1-7B", .904, 4.3),
      ("Light-R1-7B", .895, 1.0), ("JustRL-1.5B", .856, 9.1), ("Nemotron-BroRL-1.5B", .832, 12.6)]
Q3 = [("Qwen3-4B", .896, 0.7), ("Qwen3-8B", .889, 2.4), ("8B-GRPO-MATH", .865, 5.1),
      ("4B-Code-1200", .837, 9.1), ("Math-Step500", .771, 16.2)]
def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y)); return np.corrcoef(rx, ry)[0, 1]
allx = [r for _, r, _ in DS + Q3]; ally = [g for _, _, g in DS + Q3]
fig, ax = plt.subplots(figsize=(3.6, 2.8))
for grp, mk, c, lab in ((DS, "o", "#1f77b4", "DS-R1-Distill-1.5B (16k)"), (Q3, "s", "#d62728", "Qwen3-1.7B (8k)")):
    ax.scatter([r for _, r, _ in grp], [g for _, _, g in grp], marker=mk, c=c, s=34, label=lab, zorder=3, edgecolor="k", lw=.4)
    OFF = {"DeepScaleR-1.5B": (-0.004, -0.15, "left"), "R1-Distill-7B": (0.010, 0.55, "right"),
           "Skywork-OR1-7B": (0.004, -0.2, "right"), "Light-R1-7B": (-0.003, 0.15, "left"),
           "JustRL-1.5B": (-0.0035, 0.6, "left"), "Nemotron-BroRL-1.5B": (-0.004, 0.5, "left"),
           "Qwen3-4B": (-0.003, -1.05, "left"), "Qwen3-8B": (0.004, 0.25, "right"),
           "8B-GRPO-MATH": (-0.004, 0.45, "left"), "4B-Code-1200": (0.004, -1.15, "right"),
           "Math-Step500": (0.006, -0.9, "right")}
    for n, r, g in grp:
        dx, dy, ha = OFF.get(n, (0.004, 0.3, "left"))
        ax.annotate(n, (r, g), (r + dx, g + dy), fontsize=6, ha=ha)
ax.annotate("length/termination\nchannel (see text)", (.970, 6.0), (.955, 9.5), fontsize=6, ha="center",
            arrowprops=dict(arrowstyle="->", lw=.6, color="gray"), color="gray")
ax.set_xlabel(r"pre-OPD shape correlation $\rho$"); ax.set_ylabel(r"$\Delta$ accuracy (mean-7, points)")
ax.invert_xaxis(); ax.axhline(0, c="gray", lw=.6, ls=":")
ax.set_title(f"Spearman = {spearman(allx, ally):+.2f} (n={len(allx)}); w/o DeepScaleR {spearman(allx[1:], ally[1:]):+.2f}", fontsize=8)
ax.legend(loc="upper left", frameon=False); ax.spines[["top", "right"]].set_visible(False)
save(fig, "fig1c_prerho_vs_gain")
print("spearman all", spearman(allx, ally), " w/o DeepScaleR", spearman(allx[1:], ally[1:]))

# ---------------- Fig 2: dynamics ----------------
def load(p):
    d = {}
    for r in csv.DictReader(open(p)):
        for k, v in r.items():
            try: d.setdefault(k, []).append(float(v))
            except (ValueError, TypeError): d.setdefault(k, []).append(np.nan)
    return {k: np.array(v) for k, v in d.items()}
def ma(y, w=10):
    y = np.asarray(y, float); k = np.ones(w) / w
    out = np.convolve(np.nan_to_num(y, nan=np.nanmean(y)), k, mode="valid")
    return np.concatenate([np.full(w - 1, np.nan), out])
DSR = [("JustRL-1.5B (+9.1)", "probe_ds15b_fromJustRL.csv", "#1f77b4"), ("Nemotron-BroRL (+12.6)", "probe_ds15b_fromNemoBroRL.csv", "#2ca02c"),
       ("DeepScaleR (+6.0)", "probe_ds15b_fromDeepScaleR.csv", "#17becf"), ("Skywork-7B (+4.3)", "probe_ds15b_fromSky7B.csv", "#ff7f0e"),
       ("Light-R1-7B (+1.0)", "probe_ds15b_fromLight7B.csv", "#bcbd22"), ("R1-Distill-7B (+0.0)", "probe_ds15b_fromDS7B.csv", "#7f7f7f")]
Q3R = [("Math-Step500 (+16.2)", "probe_q3i17_fromK4Bmath500.csv", "#d62728"), ("4B-Code-1200 (+9.1)", "probe_q3i17_fromK4Bcode1200.csv", "#e377c2"),
       ("8B-GRPO-MATH (+5.1)", "probe_q3i17_fromQ8Bgrpomath.csv", "#9467bd"), ("Qwen3-8B (+2.4)", "probe_q3i17_from8B.csv", "#8c564b"),
       ("Qwen3-4B (+0.7)", "probe_q3i17_from4B.csv", "#7f7f7f")]
fig, ax = plt.subplots(2, 2, figsize=(7.0, 4.6), sharex=True)
for row, (runs, stu) in enumerate(((DSR, "DS-R1-Distill-1.5B"), (Q3R, "Qwen3-1.7B"))):
    for lab, f, c in runs:
        try: d = load(f)
        except FileNotFoundError: print("missing", f); continue
        m = d["step"] <= 200
        ax[row, 0].plot(d["step"][m], ma(d["rho"])[m], c=c, lw=1.2, label=lab)
        ax[row, 1].plot(d["step"][m], ma(d["overlap16"])[m], c=c, lw=1.2, label=lab)
    ax[row, 0].set_ylabel(f"{stu}\n" + r"$\rho$(student, teacher)"); ax[row, 1].set_ylabel("overlap@16")
    ax[row, 0].legend(frameon=False, ncol=1); 
    for a in ax[row]: a.grid(alpha=.25); a.spines[["top", "right"]].set_visible(False)
ax[0, 0].set_title(r"(a) shape correlation $\rho$(student, teacher)", fontsize=9)
ax[0, 1].set_title("(b) support overlap@16", fontsize=9)
for a in ax[1]: a.set_xlabel("training step")
fig.tight_layout(); save(fig, "fig2_dynamics")
