"""Figure 1 teaser with real OPD data (layout/style from the draft teaser.py).
Pair: student DeepSeek-R1-Distill-Qwen-1.5B, teacher JustRL-DeepSeek-1.5B. One base-student rollout on DAPO-Math-17K
(fork/roll_ds15b-dapo.jsonl seq 353, prompt idx 15601); both models teacher-forced on the student's tokens
(teaser_data2.py -> teaser_full.json). Full-vocabulary entropies; ESC = Pearson over the window / rollout;
Overlap@k at the fork = |TopK(p) ∩ TopK(q)| / k."""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})
T, S = "#D55E00", "#0072B2"          # teacher, student
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e6e6"

SEQ, A, B, FORK, K = "353", 14, 40, 19, 3
d = json.load(open("teaser_full.json"))[SEQ]
toks = d["toks"][A:B]; Ht = np.array(d["Ht"][A:B]); Hs = np.array(d["Hs"][A:B])
n = len(toks); x = np.arange(n)
forks = [j for j in range(n) if Ht[j] > 1.0]          # teacher fork positions to shade
fork = FORK - A
esc_w = float(np.corrcoef(Hs, Ht)[0, 1]); esc_all = d["rho_all"]; ov16_all = d["ov16_all"]
tk = d["topk"][str(FORK)]
tT = {t: p for t, p in tk["T"]}; tS = {t: p for t, p in tk["S"]}
topT = [t for t, _ in tk["T"][:K]]; topS = [t for t, _ in tk["S"][:K]]
cands = []
for t, _ in tk["T"][:4] + tk["S"][:4]:
    if t not in cands: cands.append(t)
cands.sort(key=lambda t: -(tT.get(t, 0) + tS.get(t, 0)))
pt = np.array([tT.get(t, 0.0) for t in cands]); ps = np.array([tS.get(t, 0.0) for t in cands])
shared = set(topT) & set(topS); ov_k = len(shared) / K
prefix = "".join(("" if t in ".,:;" else " ") + t for t in d["toks"][FORK - 4:FORK]).strip()

fig = plt.figure(figsize=(7.0, 2.35))
gs = fig.add_gridspec(1, 2, width_ratios=[2.1, 1], wspace=0.3, left=0.06, right=0.985, top=0.82, bottom=0.2)
ax = fig.add_subplot(gs[0]); bx = fig.add_subplot(gs[1])

# Left: where to branch
for i in forks:
    ax.axvspan(i - 0.45, i + 0.45, color=T, alpha=0.10, lw=0)
ax.plot(x, Ht, color=T, lw=1.6, marker="o", ms=3, label="Teacher", zorder=3)
ax.plot(x, Hs, color=S, lw=1.6, ls=(0, (3, 1.5)), marker="s", ms=2.8, label="Student", zorder=3)
ax.tick_params(colors=MUTED, length=2)
ax.set_xticks(x); ax.set_xticklabels(toks, rotation=60, ha="right", fontsize=6.3, color=MUTED)
for i, t in enumerate(ax.get_xticklabels()):
    if i in forks:
        t.set_color(INK); t.set_fontweight("bold")
ax.set_xlim(-0.6, n - 0.4)
ax.set_ylabel("Next-token entropy", color=INK)
ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color("#9a9a9a")
ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=6.8, handlelength=2.2)
ax.set_title("Where to branch: entropy shape", loc="left", fontsize=8.2, color=INK, fontweight="bold", pad=14)
ax.text(0, 1.03, f"along the trajectory, ESC = {esc_w:.2f} (ignores overall entropy level)",
        transform=ax.transAxes, fontsize=6.6, color=MUTED, va="bottom")

# Right: how to branch at the fork
y = np.arange(len(cands))[::-1]; h = 0.36
bx.barh(y + h/2 + 0.02, pt, h, color=T)
bx.barh(y - h/2 - 0.02, ps, h, color=S)
bx.tick_params(colors=MUTED, length=2)
bx.set_yticks(y); bx.set_yticklabels(cands, fontsize=6.8)
for t in bx.get_yticklabels():
    if t.get_text() in shared:
        t.set_color(INK); t.set_fontweight("bold")
bx.set_xlabel("Probability at the fork", color=INK)
bx.grid(axis="x", color=GRID, lw=0.5); bx.set_axisbelow(True)
for s in ("top", "right"): bx.spines[s].set_visible(False)
for s in ("left", "bottom"): bx.spines[s].set_color("#9a9a9a")
bx.set_title("How to branch: candidates", loc="left", fontsize=7.8, color=INK, fontweight="bold", pad=14)
bx.text(0, 1.03, f"at “…{prefix} ___”, Overlap@{K} = {len(shared)}/{K}",
        transform=bx.transAxes, fontsize=6.6, color=MUTED, va="bottom")

con = ConnectionPatch(xyA=(fork + 0.45, Ht[fork]), coordsA=ax.transData, xyB=(-0.03, 0.93), coordsB=bx.transAxes,
                      arrowstyle="->", color="#9a9a9a", lw=0.7, ls=(0, (2, 2)), connectionstyle="arc3,rad=-0.15")
fig.add_artist(con)

fig.savefig("paper_figs/teaser.pdf"); fig.savefig("paper_figs/teaser.png", dpi=220)
json.dump(dict(student="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", teacher="hbx/JustRL-DeepSeek-1.5B",
               rollout_seq=int(SEQ), prompt_idx=d["prompt_idx"], window=[A, B], fork_pos=FORK, k=K,
               tokens=toks, H_teacher=Ht.tolist(), H_student=Hs.tolist(), teacher_forks=[A + j for j in forks],
               esc_window=esc_w, esc_rollout=esc_all, overlap16_rollout=ov16_all,
               fork_candidates=cands, p_teacher=pt.tolist(), p_student=ps.tolist(), overlap_at_fork=ov_k,
               teacher_topk_full=tk["T"], student_topk_full=tk["S"]),
          open("paper_figs/teaser_data.json", "w"), indent=1)
print(f"window [{A},{B}) ESC={esc_w:.3f}  rollout ESC={esc_all:.3f} ov16={ov16_all:.3f}  fork {FORK} '{d['toks'][FORK]}' "
      f"Overlap@{K}={len(shared)}/{K}  forks={[(A+j, toks[j]) for j in forks]}  prefix='{prefix}'")
