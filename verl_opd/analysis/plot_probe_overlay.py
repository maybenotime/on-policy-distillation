"""Overlay the online probe curves of several runs.  usage: plot_probe_overlay.py out.png title csv:label:color [...]"""
import sys, csv, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
out, title = sys.argv[1], sys.argv[2]
runs = [a.split(":") for a in sys.argv[3:]]
MAXSTEP = 200
def load(f):
    r = [x for x in csv.DictReader(open(f)) if float(x["step"]) <= MAXSTEP]
    g = lambda k: np.array([float(x[k]) if x[k] else np.nan for x in r])
    return {"step": g("step"), "rho": g("rho"), "ov": g("overlap16"), "fo": g("fork_overlap"), "dH": g("dH"),
            "k1": g("actor_distillation_abs_loss"), "Hs": g("student_entropy"), "Ht": g("teacher_entropy_topk")}
def ma(x, w=20):
    if len(x) < w: return x
    return np.convolve(x, np.ones(w) / w, mode="valid")
panels = [("rho", r"$\rho$ (entropy-curve correlation)", False), ("ov", "overlap@16", False), ("fo", "ForkOverlap", False),
          ("dH", "dH = |mean H_s − mean H_t| (log)", True), ("k1", "|k1| per token", False), ("Hs", "student entropy (solid) / teacher top-64 entropy (dashed)", False)]
fig, axes = plt.subplots(2, 3, figsize=(15, 7.6)); axes = axes.ravel()
for ax, (key, lab, logy) in zip(axes, panels):
    for f, name, col in runs:
        d = load(f); y = ma(d[key]); x = d["step"][: len(y)]
        ax.plot(x, y, color=col, lw=1.8, label=name)
        if key == "Hs":
            yt = ma(d["Ht"]); ax.plot(d["step"][: len(yt)], yt, color=col, lw=1.2, ls="--")
    if logy: ax.set_yscale("log")
    ax.set_title(lab, fontsize=10.5); ax.grid(alpha=0.25, lw=0.5); ax.tick_params(labelsize=8.5)
    ax.set_xlabel("training step", fontsize=9)
axes[0].legend(fontsize=9, loc="lower right")
fig.suptitle(title, fontsize=12.5); fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(out, dpi=160); print("saved", out)
