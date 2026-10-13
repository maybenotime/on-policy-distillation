"""Entropy-gap (dH) trajectories during OPD, and whether dH tracks the eval gain."""
import csv, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = [  # csv, label, eval delta, colour
 ("probe_ds15b_fromJustRL.csv",      "DS-1.5B ← JustRL-1.5B",      +9.1, "#c0392b"),
 ("probe_ds15b_fromSky7B.csv",       "DS-1.5B ← Skywork-7B",       +4.3, "#e67e22"),
 ("probe_ds15b_fromDS7B.csv",        "DS-1.5B ← R1-Distill-7B",    +0.0, "#7f8c8d"),
 ("probe_q3i17_fromK4Bmath500.csv",  "Q3-1.7B ← Math-Step500",    +16.2, "#8e44ad"),
 ("probe_q3i17_from8B.csv",          "Q3-1.7B ← Qwen3-8B",         +2.4, "#2980b9"),
 ("probe_q3i17_from4B.csv",          "Q3-1.7B ← Qwen3-4B",         +0.7, "#16a085"),
]
def load(f):
    r = list(csv.DictReader(open(f)))
    st = np.array([float(x["step"]) for x in r])
    dh = np.array([float(x["dH"]) if x["dH"] else np.nan for x in r])
    rho = np.array([float(x["rho"]) if x["rho"] else np.nan for x in r])
    return st, dh, rho
def ma(x, w=20):
    if len(x) < w: return x
    return np.convolve(x, np.ones(w)/w, mode="valid")

fig = plt.figure(figsize=(15, 4.4))
gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1], wspace=0.28)
ax = fig.add_subplot(gs[0])
pts = []
for f, lab, d, c in RUNS:
    st, dh, rho = load(f)
    y = ma(dh); x = st[:len(y)]
    ax.plot(x, y, color=c, lw=1.8, label=f"{lab}  ({d:+.1f})")
    w = (st >= 181) & (st <= 200)            # the evaluated checkpoint, s200, for every run
    s0, s1 = np.nanmean(dh[:20]), np.nanmean(dh[w])
    r0, r1 = np.nanmean(rho[:20]), np.nanmean(rho[w])
    pts.append((lab, d, c, 1 - s1/s0, (r1-r0)/(1-r0)))
ax.set_yscale("log")
ax.set_xlabel("training step", fontsize=9.5)
ax.set_ylabel("entropy gap  dH = |H_student − H_teacher|  (nats, 20-step MA)", fontsize=9)
ax.axvline(200, color="#34495e", ls=":", lw=1.2)
ax.text(203, 0.05, "s200\n(evaluated)", fontsize=8, color="#34495e")
ax.set_title("the entropy gap closes fast in every run", fontsize=11)
ax.legend(fontsize=8, loc="upper left", ncol=1, framealpha=0.95)
ax.annotate("entropy blow-up\n(JustRL, after 210)", xy=(222, 0.55), xytext=(133, 0.13),
            fontsize=8, color="#c0392b", ha="center",
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2,
                            connectionstyle="arc3,rad=-0.2"))
ax.grid(alpha=0.25, lw=0.5, which="both"); ax.tick_params(labelsize=8.5)

for k, (idx, key, name) in enumerate([(1, 3, "entropy gap dH closed"), (2, 4, r"entropy-shape $\rho$ gap closed")]):
    a = fig.add_subplot(gs[idx])
    X = [p[key] for p in pts]; Y = [p[1] for p in pts]
    for p in pts:
        a.scatter(p[key], p[1], s=90, color=p[2], zorder=3, edgecolor="white", lw=1.2)
        a.annotate(p[0].split("← ")[1], (p[key], p[1]), fontsize=7.8,
                   xytext=(6, -3), textcoords="offset points", color=p[2])
    r = np.corrcoef(X, Y)[0, 1]
    from scipy.stats import spearmanr
    rs = spearmanr(X, Y).statistic
    a.set_xlabel(f"fraction of the {name}", fontsize=9.5)
    a.set_ylabel("MathArena gain (pts)", fontsize=9.5)
    a.set_title(f"Pearson r = {r:+.2f}   Spearman = {rs:+.2f}", fontsize=10.5)
    a.grid(alpha=0.25, lw=0.5); a.tick_params(labelsize=8.5)
    a.set_xlim(-0.05, 1.05); a.set_ylim(-2, 19)

fig.suptitle("Does the entropy gap track how much OPD helps?  (six DAPO-Math runs, k1+PG)", fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig("entropy_gap_vs_gain.png", dpi=160)
print("saved entropy_gap_vs_gain.png")
for lab, d, c, dhc, rhoc in pts:
    print(f"{lab:32s} delta {d:+5.1f}   dH closed {dhc:5.2f}   rho closed {rhoc:5.2f}")
