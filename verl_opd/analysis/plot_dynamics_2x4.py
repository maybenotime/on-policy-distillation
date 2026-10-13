"""Training-dynamics figure, 2 rows x 4 columns (4th = accuracy at the saved checkpoints) (user spec, 9-25):
rows = students (R1-Distill-1.5B, Qwen3-1.7B); columns = ESC rho(student, teacher), overlap@16, dH;
each panel shows one successful and one failed teacher for that student.
Per-step online probes from the k1_probe plugin (probe_*.csv), 10-step moving average, steps 1-200.
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = 10
MAXSTEP = 200
ROWS = [
    ("R1-Distill-1.5B", [("JustRL-1.5B", "probe_ds15b_fromJustRL.csv", "#1f77b4", "-"),
                         ("R1-Distill-7B", "probe_ds15b_fromDS7B.csv", "#7f7f7f", "-")]),
    ("Qwen3-1.7B", [("Qwen3-4B-RL-Math", "probe_q3i17_fromK4Bmath500.csv", "#d62728", "-"),
                    ("Qwen3-8B", "probe_q3i17_from8B.csv", "#7f7f7f", "-")]),
]
COLS = [("rho", r"ESC $\rho$(student, teacher)"), ("overlap16", "Overlap@16"), ("dH", r"$\Delta H$")]
# Accuracy at the saved checkpoints: mean of AIME24 (I/II merged), AIME25, AMC, BeyondAIME, HMMT-25, HMMT-26 (MathArena, 4 samples/problem).
ACC = {  # student -> {teacher: {step: acc}}, base at step 0; acc = mean of the six benchmarks (AIME24 I/II merged)
    "R1-Distill-1.5B": {"base": 25.5, "JustRL-1.5B": {50: 28.7, 100: 30.2, 150: 34.1, 200: 33.5},
                        "R1-Distill-7B": {200: 25.6}},
    "Qwen3-1.7B": {"base": 13.0, "Qwen3-4B-RL-Math": {50: 27.0, 100: 27.8, 150: 29.0, 200: 28.3},
                   "Qwen3-8B": {200: 15.3}},
}
# Teachers' own six-benchmark accuracy (same protocol; log §2.6), drawn as dashed horizontal lines.
TEACHER_ACC = {"JustRL-1.5B": 40.1, "R1-Distill-7B": 43.0, "Qwen3-4B-RL-Math": 40.3, "Qwen3-8B": 26.1}


def load(p):
    d = {}
    for r in csv.DictReader(open(p)):
        for k, v in r.items():
            try:
                d.setdefault(k, []).append(float(v))
            except (ValueError, TypeError):
                d.setdefault(k, []).append(np.nan)
    return {k: np.array(v) for k, v in d.items()}


def ma(y, w=W):
    y = np.asarray(y, float)
    out = np.convolve(np.nan_to_num(y, nan=np.nanmean(y)), np.ones(w) / w, mode="valid")
    return np.concatenate([np.full(w - 1, np.nan), out])


fig, ax = plt.subplots(2, 4, figsize=(13.5, 5.4), sharex=True)
for r, (stu, runs) in enumerate(ROWS):
    for lab, f, c, ls in runs:
        d = load(f)
        m = d["step"] <= MAXSTEP
        for j, (key, _) in enumerate(COLS):
            ax[r, j].plot(d["step"][m], ma(d[key])[m], c=c, lw=1.6, ls=ls, label=lab)
    acc = ACC[stu]
    for lab, f, c, ls in runs:
        pts = sorted(acc[lab].items())
        xs = [0] + [k for k, _ in pts]; ys = [acc["base"]] + [v for _, v in pts]
        ax[r, 3].plot(xs, ys, c=c, lw=1.6, ls=ls, marker="o", ms=5, label=lab)
        ax[r, 3].axhline(TEACHER_ACC[lab], c=c, lw=1.2, ls="--", alpha=0.9)
        ax[r, 3].text(198, TEACHER_ACC[lab] + 0.4, f"{lab} (teacher)", color=c, fontsize=7.5, va="bottom", ha="right")
    ax[r, 3].set_xticks([0, 50, 100, 150, 200])
    ax[r, 0].set_ylabel(stu, fontsize=11)
    ax[r, 0].legend(frameon=False, fontsize=9, loc="lower right")
    for a in ax[r]:
        a.grid(alpha=.25)
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(labelsize=9)
for j, (_, title) in enumerate(COLS + [(None, "Average Performance")]):
    ax[0, j].set_title(title, fontsize=11)
    ax[1, j].set_xlabel("Training Step", fontsize=10)
fig.tight_layout()
out = "/localhome/sya225/Projects/opd_experiments/results/paper_figs/fig_dynamics_2x4"
fig.savefig(out + ".png", dpi=200)
fig.savefig(out + ".pdf")
print("saved", out)
for stu, runs in ROWS:
    for lab, f, *_ in runs:
        d = load(f); m = d["step"] <= MAXSTEP
        print(f"{stu:16s} {lab:22s} rho {d['rho'][m][0]:.3f}->{np.nanmean(d['rho'][m][-10:]):.3f}  "
              f"ov16 {d['overlap16'][m][0]:.3f}->{np.nanmean(d['overlap16'][m][-10:]):.3f}  "
              f"dH {d['dH'][m][0]:.3f}->{np.nanmean(d['dH'][m][-10:]):.3f}")
