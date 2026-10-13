"""Single-column (ACL, 3.03 in wide) version of teaser3.py: panels stacked vertically.
usage: python teaser3_col.py <seq> <a> <b> <fork_pos> <k> <out_stem> [HT=1.0] [HS=0.6] [src_json]"""
import json, sys, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Patch
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
T, S = "#D55E00", "#0072B2"; INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e6e6"
SEQ, A, B, FORK, K, OUT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
HT = float(sys.argv[7]) if len(sys.argv) > 7 else 1.0; HS = float(sys.argv[8]) if len(sys.argv) > 8 else 0.6
SRC = sys.argv[9] if len(sys.argv) > 9 else "teaser_full_big.json"
d = json.load(open(SRC))[SEQ]
toks = d["toks"][A:B]; Ht = np.array(d["Ht"][A:B]); Hs = np.array(d["Hs"][A:B]); n = len(toks); x = np.arange(n)
import os
forks = [j for j in range(n) if Ht[j] > HT and Hs[j] < HS]; fork = FORK - A
if os.environ.get("SHADE"): forks = [int(p) - A for p in os.environ["SHADE"].split(",")]
BW = float(os.environ.get("BW", "0.5"))
esc_w = float(np.corrcoef(Hs, Ht)[0, 1]); tk = d["topk"][str(FORK)]
tT = dict(tk["T"]); tS = dict(tk["S"]); FLOOR = 0.01
topT = [t for t, v in tk["T"][:K] if v >= FLOOR]; topS = [t for t, v in tk["S"][:K] if v >= FLOOR]
cands = []
for t, _ in tk["T"][:4] + tk["S"][:4]:
    if t not in cands: cands.append(t)
cands.sort(key=lambda t: -(tT.get(t, 0) + tS.get(t, 0)))
pt = np.array([tT.get(t, 0.0) for t in cands]); ps = np.array([tS.get(t, 0.0) for t in cands]); shared = set(topT) & set(topS)
def _pre(m): return "".join(("" if t in ".,:;?" else " ") + t for t in d["toks"][FORK-m:FORK] if t != "␣").strip()
prefix = _pre(4) if len(_pre(4)) <= 14 else _pre(3); prefix = prefix.lstrip(".,:;?! ")
fig = plt.figure(figsize=(3.03, 4.0))
gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 1], hspace=0.82, left=0.2, right=0.98, top=0.9, bottom=0.085)
ax = fig.add_subplot(gs[0]); bx = fig.add_subplot(gs[1])
for i in forks: ax.axvspan(i - BW, i + BW, color=T, alpha=0.14, lw=0)
ax.plot(x, Ht, color=T, lw=1.3, marker="o", ms=2.4, label="Teacher", zorder=3)
ax.plot(x, Hs, color=S, lw=1.3, ls=(0, (3, 1.5)), marker="s", ms=2.2, label="Student", zorder=3)
ax.tick_params(colors=MUTED, length=2, labelsize=6); ax.set_xticks(x)
ax.set_xticklabels(toks, rotation=90, fontsize=5.5, color=MUTED)
for i, t in enumerate(ax.get_xticklabels()):
    if i in forks: t.set_color(INK); t.set_fontweight("bold")
ax.set_xlim(-0.6, n - 0.4); ax.set_ylim(-0.05, max(Ht.max(), Hs.max()) * 1.4)
ax.set_ylabel("Next-token entropy", color=INK, fontsize=6.8); ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color("#9a9a9a")
hd, lb = ax.get_legend_handles_labels()
ax.legend(hd + [Patch(fc=T, alpha=0.18, ec="none")], lb + ["Mismatch"], loc="upper left", frameon=False, ncol=2, fontsize=5.8, handlelength=1.8, columnspacing=0.9, labelspacing=0.3)
ax.set_title("Where to branch: entropy shape", loc="left", fontsize=7.6, color=INK, fontweight="bold", pad=11)
ax.text(0, 1.03, f"along the trajectory, ESC = {esc_w:.2f}", transform=ax.transAxes, fontsize=6.2, color=MUTED, va="bottom")
y = np.arange(len(cands))[::-1]; hh = 0.36
bx.barh(y + hh/2 + 0.02, pt, hh, color=T); bx.barh(y - hh/2 - 0.02, ps, hh, color=S)
bx.tick_params(colors=MUTED, length=2, labelsize=6); bx.set_yticks(y); bx.set_yticklabels(cands, fontsize=6.3)
for t in bx.get_yticklabels():
    if t.get_text() in shared: t.set_color(INK); t.set_fontweight("bold")
bx.set_xlabel("Probability", color=INK, fontsize=6.8); bx.grid(axis="x", color=GRID, lw=0.5); bx.set_axisbelow(True)
for s in ("top", "right"): bx.spines[s].set_visible(False)
for s in ("left", "bottom"): bx.spines[s].set_color("#9a9a9a")
bx.set_title("How to branch: token candidates", loc="left", fontsize=7.6, color=INK, fontweight="bold", pad=11)
bx.text(0, 1.03, (f"at “…{prefix} ___”, Overlap@k" if os.environ.get("SUB", "generic") == "generic" else
         f"at “…{prefix} ___”, Overlap@16 = {len(set(t for t,_ in tk['T'][:16]) & set(t for t,_ in tk['S'][:16]))}/16"), transform=bx.transAxes, fontsize=6.2, color=MUTED, va="bottom")
fx = (fork + 0.6) / n   # fork x in axes fraction; arrow starts below the tick labels and bends right of the lower title
fig.add_artist(ConnectionPatch(xyA=(fx, -0.42), coordsA=ax.transAxes, xyB=(0.985, 1.02), coordsB=bx.transAxes,
    arrowstyle="->", color="#9a9a9a", lw=0.8, ls=(0, (2, 2)), connectionstyle="arc3,rad=-0.35", clip_on=False, zorder=10))
fig.savefig(f"paper_figs/{OUT}.pdf"); fig.savefig(f"paper_figs/{OUT}.png", dpi=220)
print(f"{OUT}: window [{A},{B}) n={n} ESC_w={esc_w:.3f} fork {FORK} Overlap@{K}={len(shared)}/{K} shaded={[(A+j, toks[j]) for j in forks]}")
