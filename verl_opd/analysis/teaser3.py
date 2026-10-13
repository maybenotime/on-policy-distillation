"""Teaser from full-rollout dumps (teaser_full_big.json, teaser_dump_full.py). Shading rule (user, 10-12): a band marks a
DISAGREEMENT fork = teacher high-entropy (H_t > HT) while the student is committed (H_s < HS); positions where both models
are high-entropy are not shaded.  usage: python teaser3.py <seq> <a> <b> <fork_pos> <k> <out_stem> [HT=1.0] [HS=0.6] [src_json]"""
import json, sys, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Patch
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
T, S = "#D55E00", "#0072B2"; INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e6e6"
SEQ, A, B, FORK, K, OUT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
HT = float(sys.argv[7]) if len(sys.argv) > 7 else 1.0; HS = float(sys.argv[8]) if len(sys.argv) > 8 else 0.6
SRC = sys.argv[9] if len(sys.argv) > 9 else "teaser_full_big.json"
d = json.load(open(SRC))[SEQ]
toks = d["toks"][A:B]; Ht = np.array(d["Ht"][A:B]); Hs = np.array(d["Hs"][A:B]); n = len(toks); x = np.arange(n)
import os
GAP = float(os.environ.get("GAP", "inf")); RGAP = float(os.environ.get("RGAP", "inf"))   # optional: gap rule / reverse (student-only) forks
forks = [j for j in range(n) if Ht[j] > HT and (Hs[j] < HS or Ht[j] - Hs[j] >= GAP)]; fork = FORK - A
if os.environ.get("SHADE"): forks = [int(p) - A for p in os.environ["SHADE"].split(",")]   # explicit list of absolute positions
BW = float(os.environ.get("BW", "0.5"))   # band half-width
rforks = [j for j in range(n) if Hs[j] > HT and Hs[j] - Ht[j] >= RGAP]
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
fig = plt.figure(figsize=(7.0, 2.35)); gs = fig.add_gridspec(1, 2, width_ratios=[2.0, 1.12], wspace=0.3, left=0.06, right=0.99, top=0.82, bottom=0.2)
ax = fig.add_subplot(gs[0]); bx = fig.add_subplot(gs[1])
for i in forks: ax.axvspan(i - BW, i + BW, color=T, alpha=0.14, lw=0)
for i in rforks: ax.axvspan(i - BW, i + BW, color=S, alpha=0.14, lw=0)
ax.plot(x, Ht, color=T, lw=1.6, marker="o", ms=3, label="Teacher", zorder=3)
ax.plot(x, Hs, color=S, lw=1.6, ls=(0, (3, 1.5)), marker="s", ms=2.8, label="Student", zorder=3)
ax.tick_params(colors=MUTED, length=2); ax.set_xticks(x)
ax.set_xticklabels(toks, rotation=60, ha="right", fontsize=6.3 if n <= 28 else 5.6, color=MUTED)
for i, t in enumerate(ax.get_xticklabels()):
    if i in forks or i in rforks: t.set_color(INK); t.set_fontweight("bold")
ax.set_xlim(-0.6, n - 0.4); ax.set_ylim(-0.05, max(Ht.max(), Hs.max()) * (1.28 if not rforks else 1.5)); ax.set_ylabel("Next-token entropy", color=INK); ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color("#9a9a9a")
hd, lb = ax.get_legend_handles_labels()
hd += [Patch(fc=T, alpha=0.18, ec="none")]; lb += ["Mismatch"]
if rforks: hd += [Patch(fc=S, alpha=0.18, ec="none")]; lb += ["student forks, teacher committed"]
ax.legend(hd, lb, loc="upper left", frameon=False, ncol=3 if len(lb) <= 3 else 2, fontsize=6.6, handlelength=2.0, columnspacing=1.2)
ax.set_title("Where to branch: entropy shape", loc="left", fontsize=8.2, color=INK, fontweight="bold", pad=14)
ax.text(0, 1.03, f"along the trajectory, ESC = {esc_w:.2f}", transform=ax.transAxes, fontsize=6.6, color=MUTED, va="bottom")
y = np.arange(len(cands))[::-1]; hh = 0.36
bx.barh(y + hh/2 + 0.02, pt, hh, color=T); bx.barh(y - hh/2 - 0.02, ps, hh, color=S)
bx.tick_params(colors=MUTED, length=2); bx.set_yticks(y); bx.set_yticklabels(cands, fontsize=6.8)
for t in bx.get_yticklabels():
    if t.get_text() in shared: t.set_color(INK); t.set_fontweight("bold")
bx.set_xlabel("Probability", color=INK); bx.grid(axis="x", color=GRID, lw=0.5); bx.set_axisbelow(True)
for s in ("top", "right"): bx.spines[s].set_visible(False)
for s in ("left", "bottom"): bx.spines[s].set_color("#9a9a9a")
bx.set_title("How to branch: token candidates", loc="left", fontsize=7.6, color=INK, fontweight="bold", pad=14)
bx.text(0, 1.03, (f"at “…{prefix} ___”, Overlap@k" if os.environ.get("SUB", "generic") == "generic" else
         f"at “…{prefix} ___”, Overlap@16 = {len(set(t for t,_ in tk['T'][:16]) & set(t for t,_ in tk['S'][:16]))}/16"), transform=bx.transAxes, fontsize=6.6, color=MUTED, va="bottom")
fig.add_artist(ConnectionPatch(xyA=(fork + 0.45, Ht[fork]), coordsA=ax.transData, xyB=(-0.03, 0.93), coordsB=bx.transAxes, arrowstyle="->", color="#9a9a9a", lw=0.7, ls=(0, (2, 2)), connectionstyle="arc3,rad=0.18"))
fig.savefig(f"paper_figs/{OUT}.pdf"); fig.savefig(f"paper_figs/{OUT}.png", dpi=220)
json.dump(dict(student="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", teacher="hbx/JustRL-DeepSeek-1.5B", rollout_file="roll_ds15b-dapo-big.jsonl", rollout_seq=int(SEQ), prompt_idx=d["prompt_idx"],
    window=[A, B], fork_pos=FORK, k=K, shade_rule=dict(H_teacher_gt=HT, H_student_lt=HS, gap=GAP, rev_gap=RGAP), student_forks=[A+j for j in rforks], tokens=toks, H_teacher=Ht.tolist(), H_student=Hs.tolist(), shaded_forks=[A+j for j in forks],
    esc_window=esc_w, esc_rollout=d["rho_all"], overlap16_rollout=d["ov16_all"], fork_candidates=cands, p_teacher=pt.tolist(), p_student=ps.tolist(), overlap_at_fork=len(shared)/K,
    teacher_topk_full=tk["T"], student_topk_full=tk["S"]), open(f"paper_figs/{OUT}_data.json", "w"), indent=1)
print(f"{OUT}: seq {SEQ} prompt {d['prompt_idx']} window [{A},{B}) n={n} ESC_w={esc_w:.3f} ESC_all={d['rho_all']:.3f} ov16={d['ov16_all']:.3f} fork {FORK} '{d['toks'][FORK]}' Overlap@{K}={len(shared)}/{K} shaded={[(A+j, toks[j], round(Ht[j],2), round(Hs[j],2)) for j in forks]}")
