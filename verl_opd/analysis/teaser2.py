"""Teaser variants from teaser_forks2.json (teaser_data4.py): windows where the teacher forks at an explicit branch while the
student is committed.  usage: python teaser2.py <seq> <a> <b> <fork_pos> <k> <out_stem>"""
import json, sys, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
T, S = "#D55E00", "#0072B2"; INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e6e6"
SEQ, A, B, FORK, K, OUT = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
SRC = sys.argv[7] if len(sys.argv) > 7 else "teaser_forks2.json"
h = next(x for x in json.load(open(SRC)) if x["seq"] == SEQ); lo = h["lo"]
toks = h["toks_all"][A-lo:B-lo]; Ht = np.array(h["Ht_win"][A-lo:B-lo]); Hs = np.array(h["Hs_win"][A-lo:B-lo]); n = len(toks); x = np.arange(n)
PUNCT = set(",.:;?!")
forks = [j for j in range(n) if Ht[j] > 1.0 and toks[j] not in PUNCT]; fork = FORK - A   # shaded forks: word tokens with H > 1
esc_w = float(np.corrcoef(Hs, Ht)[0, 1]); tk = h["topk"][str(FORK)]
tT = dict(tk["T"]); tS = dict(tk["S"]); FLOOR = 0.01   # top-k sets ignore near-zero ties (p < 1%) so that a committed model does not "share" its 2nd/3rd candidate
topT = [t for t, v in tk["T"][:K] if v >= FLOOR]; topS = [t for t, v in tk["S"][:K] if v >= FLOOR]
cands = []
for t, _ in tk["T"][:4] + tk["S"][:4]:
    if t not in cands: cands.append(t)
cands.sort(key=lambda t: -(tT.get(t, 0) + tS.get(t, 0)))
pt = np.array([tT.get(t, 0.0) for t in cands]); ps = np.array([tS.get(t, 0.0) for t in cands]); shared = set(topT) & set(topS)
def _pre(m): return "".join(("" if t in ".,:;?" else " ") + t for t in h["toks_all"][FORK-lo-m:FORK-lo] if t != "\u2423").strip()
prefix = _pre(4) if len(_pre(4)) <= 14 else _pre(3)
fig = plt.figure(figsize=(7.0, 2.35)); gs = fig.add_gridspec(1, 2, width_ratios=[2.1, 1], wspace=0.3, left=0.06, right=0.985, top=0.82, bottom=0.2)
ax = fig.add_subplot(gs[0]); bx = fig.add_subplot(gs[1])
for i in forks: ax.axvspan(i - 0.45, i + 0.45, color=T, alpha=0.10, lw=0)
ax.plot(x, Ht, color=T, lw=1.6, marker="o", ms=3, label="Teacher", zorder=3)
ax.plot(x, Hs, color=S, lw=1.6, ls=(0, (3, 1.5)), marker="s", ms=2.8, label="Student", zorder=3)
ax.tick_params(colors=MUTED, length=2); ax.set_xticks(x); ax.set_xticklabels([("\u2423" if t == "\u2423" else t) for t in toks], rotation=60, ha="right", fontsize=6.3, color=MUTED)
for i, t in enumerate(ax.get_xticklabels()):
    if i in forks: t.set_color(INK); t.set_fontweight("bold")
ax.set_xlim(-0.6, n - 0.4); ax.set_ylabel("Next-token entropy", color=INK); ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color("#9a9a9a")
ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=6.8, handlelength=2.2)
ax.set_title("Where to branch: entropy shape", loc="left", fontsize=8.2, color=INK, fontweight="bold", pad=14)
ax.text(0, 1.03, f"along the trajectory, ESC = {esc_w:.2f} (ignores overall entropy level)", transform=ax.transAxes, fontsize=6.6, color=MUTED, va="bottom")
y = np.arange(len(cands))[::-1]; hh = 0.36
bx.barh(y + hh/2 + 0.02, pt, hh, color=T); bx.barh(y - hh/2 - 0.02, ps, hh, color=S)
bx.tick_params(colors=MUTED, length=2); bx.set_yticks(y); bx.set_yticklabels(cands, fontsize=6.8)
for t in bx.get_yticklabels():
    if t.get_text() in shared: t.set_color(INK); t.set_fontweight("bold")
bx.set_xlabel("Probability at the fork", color=INK); bx.grid(axis="x", color=GRID, lw=0.5); bx.set_axisbelow(True)
for s in ("top", "right"): bx.spines[s].set_visible(False)
for s in ("left", "bottom"): bx.spines[s].set_color("#9a9a9a")
bx.set_title("How to branch: candidates", loc="left", fontsize=7.8, color=INK, fontweight="bold", pad=14)
bx.text(0, 1.03, f"at “…{prefix} ___”, Overlap@{K} = {len(shared)}/{K}", transform=bx.transAxes, fontsize=6.6, color=MUTED, va="bottom")
fig.add_artist(ConnectionPatch(xyA=(fork + 0.45, Ht[fork]), coordsA=ax.transData, xyB=(-0.03, 0.93), coordsB=bx.transAxes, arrowstyle="->", color="#9a9a9a", lw=0.7, ls=(0, (2, 2)), connectionstyle="arc3,rad=0.18"))
fig.savefig(f"paper_figs/{OUT}.pdf"); fig.savefig(f"paper_figs/{OUT}.png", dpi=220)
json.dump(dict(student="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", teacher="hbx/JustRL-DeepSeek-1.5B", rollout_seq=SEQ, prompt_idx=h["prompt_idx"], window=[A, B], fork_pos=FORK, k=K,
    tokens=toks, H_teacher=Ht.tolist(), H_student=Hs.tolist(), teacher_forks=[A+j for j in forks], esc_window=esc_w, esc_rollout=h["rho_all"], overlap16_rollout=h["ov16_all"],
    fork_candidates=cands, p_teacher=pt.tolist(), p_student=ps.tolist(), overlap_at_fork=len(shared)/K, teacher_topk_full=tk["T"], student_topk_full=tk["S"]), open(f"paper_figs/{OUT}_data.json", "w"), indent=1)
print(f"{OUT}: seq {SEQ} prompt {h['prompt_idx']} window [{A},{B}) ESC_w={esc_w:.3f} ESC_all={h['rho_all']:.3f} ov16={h['ov16_all']:.3f} fork {FORK} '{h['toks_all'][FORK-lo]}' Overlap@{K}={len(shared)}/{K} forks={[(A+j, toks[j]) for j in forks]}")
