"""All monitored per-step quantities for several runs.  usage: plot_probe_full.py out.png title MAXSTEP csv:label:color[:maxstep] ..."""
import sys, csv, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
out, title, MAX = sys.argv[1], sys.argv[2], int(sys.argv[3])
runs = [a.split(":") for a in sys.argv[4:]]
def load(f, mx):
    r = [x for x in csv.DictReader(open(f)) if float(x["step"]) <= mx]
    g = lambda k: np.array([float(x[k]) if x.get(k) else np.nan for x in r])
    return {"step": g("step"), "rho": g("rho"), "ov": g("overlap16"), "fo": g("fork_overlap"), "dH": g("dH"), "k1": g("actor_distillation_abs_loss"),
            "Hs": g("student_entropy"), "Ht": g("teacher_entropy_topk"), "intop": g("sampled_in_topk"), "gn": g("actor_grad_norm"),
            "len": g("response_length_mean"), "clip": g("response_length_clip_ratio")}
def ma(x, w=10):
    x = np.asarray(x, float)
    if len(x) < w: return x
    return np.convolve(x, np.ones(w) / w, mode="valid")
panels = [("rho", r"$\rho$: corr. of per-position entropy curves (student vs teacher)", False, None),
          ("ov", "overlap@16: teacher top-16 ∩ student top-16 (mass-weighted)", False, None),
          ("fo", "ForkOverlap: overlap restricted to high-entropy (fork) positions", False, None),
          ("dH", "dH = |mean H_student − mean H_teacher|  (log)", True, None),
          ("k1", "|k1| = mean |log q − log p| on sampled tokens (distillation signal)", False, None),
          ("Hs", "student entropy (solid) / teacher top-64 entropy (dashed)", False, None),
          ("intop", "sampled_in_topk: share of student tokens inside teacher top-64 (health)", False, (0.9, 1.001)),
          ("gn", "actor grad norm (log)", True, None),
          ("len", "mean response length (solid) / clip ratio at max len (dashed, right axis)", False, None)]
fig, axes = plt.subplots(3, 3, figsize=(16.5, 11)); axes = axes.ravel()
for ax, (key, lab, logy, ylim) in zip(axes, panels):
    ax2 = ax.twinx() if key == "len" else None
    for spec in runs:
        f, name, col = spec[:3]; mx = int(spec[3]) if len(spec) > 3 else MAX
        d = load(f, mx); y = ma(d[key]); x = d["step"][: len(y)]
        ax.plot(x, y, color=col, lw=1.7, label=name)
        if key == "Hs": yt = ma(d["Ht"]); ax.plot(d["step"][: len(yt)], yt, color=col, lw=1.1, ls="--")
        if key == "len": yc = ma(d["clip"]); ax2.plot(d["step"][: len(yc)], yc, color=col, lw=1.0, ls="--")
    if logy: ax.set_yscale("log")
    if ylim: ax.set_ylim(*ylim)
    if ax2: ax2.set_ylim(0, 1); ax2.set_ylabel("clip ratio", fontsize=8)
    ax.set_title(lab, fontsize=9.5); ax.grid(alpha=0.25, lw=0.5); ax.tick_params(labelsize=8); ax.set_xlabel("training step", fontsize=8.5)
axes[0].legend(fontsize=8.5, loc="lower right")
fig.suptitle(title, fontsize=12); fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(out, dpi=150); print("saved", out)
