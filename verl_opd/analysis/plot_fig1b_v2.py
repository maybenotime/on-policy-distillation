"""Figure 1(b), notation of the Counterfactual Teacher Setup: A = successful teacher, B = failing teacher;
q~^{B<-A} = B's token preferences with A's entropy shape, q~^{A<-B} = A's preferences with B's shape.
Bars = six-benchmark average gain over the base student.  Both students at step 100; Qwen3 B (no step-100 checkpoint) and q~^{B<-A} (step 100 not evaluated) use their step-200 values (marked).
DeepSeek's B is the student itself (zero divergence -> no learning signal), drawn as an empty bar at 0."""
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
PANELS = [
 ("Qwen3-1.7B  (step 100)", "$A$ = Qwen3-4B-RL-Math,  $B$ = Qwen3-4B", [0.5, 1.9, 15.3, 12.3], (-1, 17.5)),
 ("R1-Distill-1.5B  (step 100)", "$A$ = JustRL-1.5B,  $B$ = student itself", [0.0, 2.2, 4.8, 3.9], (-0.6, 6.0)),
]
LAB = ["$B$", r"$\tilde{q}^{\,B\leftarrow A}$", "$A$", r"$\tilde{q}^{\,A\leftarrow B}$"]
COL = ["#aec7e8", "#1f77b4", "#ffbb78", "#d62728"]
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5))
for ax, (title, sub, v, yl) in zip(axes, PANELS):
    for i, (l, y, c) in enumerate(zip(LAB, v, COL)):
        if y == 0.0:
            ax.bar(i, 0.0, color="none", edgecolor="#7f7f7f", lw=1.2, ls="--", width=0.72)
            ax.text(i, 0.12, "no signal\n(+0.0)", ha="center", va="bottom", fontsize=8, color="#7f7f7f")
        else:
            ax.bar(i, y, color=c, edgecolor="k", lw=0.6, width=0.72)
            tag = f"{y:+.1f}"
            ax.text(i, y + (0.25 if yl[1] > 10 else 0.12), tag, ha="center", va="bottom", fontsize=9)
    ax.axhline(0, c="k", lw=0.8)
    ax.set_xticks(range(4)); ax.set_xticklabels(LAB, fontsize=11)
    ax.set_ylim(*yl); ax.set_title(f"{title}\n{sub}", fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=.25)
axes[0].set_ylabel(r"$\Delta$ Average Performance (points)", fontsize=10)
fig.tight_layout()
out = "paper_figs/fig1b_transplant_bars_v2"; fig.savefig(out + ".png", dpi=200); fig.savefig(out + ".pdf"); print("saved", out)
