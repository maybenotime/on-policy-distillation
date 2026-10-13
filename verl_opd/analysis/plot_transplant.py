"""Transplant runs: how far the student's entropy shape moves toward each teacher.
usage: plot_transplant.py out.png"""
import csv, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def series(f):
    a = np.array([[float(x) for x in l.split()] for l in open(f) if len(l.split()) == 7])
    return dict(zip(("step", "rB", "rA", "rq", "rho", "ov", "len"), (a[:, i] for i in range(7))))
def ma(x, w=5):
    return np.convolve(x, np.ones(w) / w, mode="valid") if len(x) >= w else np.asarray(x)
R3 = series("q3i17_tpA4B_shapeM500_series.txt")      # A=4B preferences, shape=M500
R5 = series("q3i17_tpA4B_uniform_series.txt")        # A/T*, target = sharpened A
R4 = series("q3i17_tpM500_shapeA4B_series.txt")      # A=M500 preferences, shape=4B
# direct-teaching references (probe CSVs of the ordinary OPD runs)
def csvser(f, col):
    r = list(csv.DictReader(open(f)))
    return np.array([float(x["step"]) for x in r]), np.array([float(x[col]) for x in r])
sM, rM = csvser("probe_q3i17_fromK4Bmath500.csv", "rho")
s4, r4 = csvser("probe_q3i17_from4B.csv", "rho")
fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
# panel 1: rho to the SHAPE SOURCE
ax[0].axhline(0.772, color="green", ls=":", lw=1, label="pre-ρ(student, M500) = 0.772")
y = ma(R3["rB"]); ax[0].plot(R3["step"][:len(y)], y, color="crimson", lw=1.8, label="run 3: 4B ⊕ shape(M500) → ρ vs M500")
y = ma(rM); ax[0].plot(sM[:len(y)], y, color="green", lw=1.6, ls="--", label="M500 teaching directly → ρ vs M500")
y = ma(R4["rB"]); ax[0].plot(R4["step"][:len(y)], y, color="navy", lw=1.8, label="run 4: M500 ⊕ shape(4B) → ρ vs 4B")
y = ma(r4); ax[0].plot(s4[:len(y)], y, color="navy", lw=1.6, ls="--", label="4B teaching directly → ρ vs 4B")
ax[0].set_title("ρ(student, shape source)", fontsize=10.5); ax[0].set_ylim(0.74, 0.96); ax[0].legend(fontsize=7.5, loc="lower right")
# panel 2: rho to the PREFERENCE SOURCE
y = ma(R3["rA"]); ax[1].plot(R3["step"][:len(y)], y, color="crimson", lw=1.8, label="run 3 → ρ vs 4B (preferences)")
y = ma(R4["rA"]); ax[1].plot(R4["step"][:len(y)], y, color="navy", lw=1.8, label="run 4 → ρ vs M500 (preferences)")
y = ma(R5["rB"]); ax[1].plot(R5["step"][:len(y)], y, color="gray", lw=1.6, label="run 5 uniform T* → ρ vs sharpened 4B")
ax[1].set_title("ρ(student, preference source)", fontsize=10.5); ax[1].set_ylim(0.74, 0.96); ax[1].legend(fontsize=7.5, loc="lower right")
# panel 3: response length
for d, c, lab in ((R3, "crimson", "run 3 (4B prefs)"), (R4, "navy", "run 4 (M500 prefs)"), (R5, "gray", "run 5 (4B/T*)")):
    y = ma(d["len"]); ax[2].plot(d["step"][:len(y)], y, color=c, lw=1.8, label=lab)
_, lM = csvser("probe_q3i17_fromK4Bmath500.csv", "response_length_mean")
y = ma(lM); ax[2].plot(sM[:len(y)], y, color="green", lw=1.6, ls="--", label="M500 direct")
_, l4 = csvser("probe_q3i17_from4B.csv", "response_length_mean")
y = ma(l4); ax[2].plot(s4[:len(y)], y, color="purple", lw=1.6, ls="--", label="4B direct")
ax[2].set_title("mean response length (tokens)", fontsize=10.5); ax[2].legend(fontsize=7.5)
for a in ax: a.grid(alpha=.25, lw=.5); a.set_xlabel("training step", fontsize=9); a.set_xlim(0, 200); a.tick_params(labelsize=8.5)
fig.suptitle("Entropy-shape transplant, Qwen3-1.7B student (5-step MA; dashed = the same teacher taught directly)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig("transplant_q3i17.png", dpi=150); print("saved transplant_q3i17.png")
