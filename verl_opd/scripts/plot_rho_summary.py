"""Per-rollout rho distributions, before vs after OPD, one panel per teacher."""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(f):
    d = np.load(f, allow_pickle=True)
    return [d[str(i)].astype(np.float32) for i in range(len(d.files))]

def rho(S, T):
    o = []
    for s, t in zip(S, T):
        n = min(len(s), len(t))
        if n < 25 or s[:n].std() < 1e-6 or t[:n].std() < 1e-6: continue
        o.append(float(np.corrcoef(s[:n], t[:n])[0, 1]))
    return np.array(o)

RUNS = [("JustRL-1.5B", "ent_TJustRL_ds15b-dapo.npz", "fromJustRL", "+9.1"),
        ("Skywork-OR1-Math-7B", "ent_TSky7B_ds15b-dapo.npz", "fromSky7B", "+4.3"),
        ("R1-Distill-7B", "ent_TDS7B_ds15b-dapo.npz", "fromDS7B", "+0.0")]
sp = load("ent_student_ds15b-dapo.npz")
fig, axes = plt.subplots(1, 3, figsize=(14, 3.9), sharex=True, sharey=True)
bins = np.linspace(0.6, 1.0, 60)
for ax, (name, tf, tag, gain) in zip(axes, RUNS):
    rp = rho(sp, load(tf))
    rq = rho(load(f"ent_student_ds15b-{tag}-post.npz"), load(f"ent_teacher_ds15b-{tag}-post.npz"))
    ax.hist(rp, bins=bins, color="#7f8c8d", alpha=0.8, label=f"before  {np.median(rp):.3f}")
    ax.hist(rq, bins=bins, color="#27ae60", alpha=0.8, label=f"after   {np.median(rq):.3f}")
    ax.axvline(np.median(rp), color="#2c3e50", ls="--", lw=1.1)
    ax.axvline(np.median(rq), color="#145a32", ls="--", lw=1.1)
    ax.annotate("", xy=(np.median(rq), 620), xytext=(np.median(rp), 620),
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.6))
    ax.set_title(f"teacher: {name}   ({gain} pts)", fontsize=10.5)
    ax.set_xlabel(r"per-rollout $\rho$(student H, teacher H)", fontsize=9.5)
    ax.legend(fontsize=9, loc="upper left"); ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelsize=8.5)
axes[0].set_ylabel("rollouts (of 1000)", fontsize=9.5)
fig.suptitle("How far OPD moves the student's entropy shape toward the teacher — R1-Distill-1.5B on DAPO-Math", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig("/localhome/USER/Projects/opd_experiments/results/entropy_shape_ds15b_summary.png", dpi=160)
print("saved summary")
