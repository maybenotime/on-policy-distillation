"""DeepSeek pair: the two 50-step transplant runs against JustRL teaching directly
for 271 steps — does the transplant still have headroom at step 50?"""
import csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(p):
    d = {}
    with open(p) as fh:
        for r in csv.DictReader(fh):
            for k, v in r.items():
                d.setdefault(k, []).append(float(v) if v not in ("", None) else np.nan)
    return {k: np.array(v) for k, v in d.items()}

J, R, S = load("ser_JustRL.csv"), load("ser_tpR17B.csv"), load("ser_tpStu.csv")

def sm(y, w=5):
    y = np.asarray(y, float)
    if len(y) < w: return y
    k = np.ones(w) / w
    out = np.convolve(np.nan_to_num(y, nan=np.nanmean(y)), k, mode="same")
    out[:w // 2] = y[:w // 2]; out[-(w // 2):] = y[-(w // 2):]
    return out

RUNS = [
    ("JustRL direct (271 steps)", J, "actor/probe/rho",        "actor/probe/overlap16", "#1f77b4"),
    ("A=R1-7B  + shape(JustRL)",  R, "actor/tp/rho_s_B",       "actor/tp/overlap16_sB", "#d62728"),
    ("A=student + shape(JustRL)", S, "actor/tp/rho_s_B",       "actor/tp/overlap16_sB", "#2ca02c"),
]

fig, ax = plt.subplots(2, 3, figsize=(16, 8))
for name, d, rho_k, ov_k, c in RUNS:
    x = d["step"]
    ax[0, 0].plot(x, sm(d[rho_k]), color=c, label=name)
    ax[0, 1].plot(x, sm(d[ov_k]), color=c, label=name)
    ax[0, 2].plot(x, sm(d["response_length/mean"]), color=c, label=name)
    ax[1, 0].plot(x, sm(d["actor/entropy"]), color=c, label=name)
    k1 = d["actor/tp/abs_k1_A"] if not np.all(np.isnan(d.get("actor/tp/abs_k1_A", [np.nan]))) \
        else d["actor/distillation/abs_loss"]
    ax[1, 1].plot(x, sm(k1), color=c, label=name)
    ax[1, 2].plot(x, sm(d["actor/grad_norm"]), color=c, label=name)

TIT = [("rho(student, JustRL)", "rho"), ("overlap@16 (student vs JustRL)", "overlap"),
       ("mean response length", "tokens"), ("student entropy", "nats"),
       ("|k1| against the training target", "nats"), ("grad norm", "")]
for a, (t, yl) in zip(ax.ravel(), TIT):
    a.set_title(t); a.set_xlabel("step"); a.set_ylabel(yl)
    a.axvline(50, ls=":", c="gray", lw=1)
    a.grid(alpha=.3); a.legend(fontsize=7)
fig.suptitle("DeepSeek-R1-Distill-1.5B: entropy-shape transplant (50 steps) vs JustRL teaching directly\n"
             "dotted line = step 50, where both transplant runs were stopped", fontsize=11)
fig.tight_layout()
fig.savefig("tp_vs_justrl_curves.png", dpi=130)
print("wrote tp_vs_justrl_curves.png")

# the quantitative version of "is it still moving at step 50?"
print(f"\n{'run':<28} {'rho@1-10':>9} {'rho@41-50':>10} {'slope/10st (31-50)':>19}  {'len@41-50':>10}")
for name, d, rho_k, ov_k, c in RUNS:
    r, x = d[rho_k], d["step"]
    a = np.nanmean(r[(x >= 1) & (x <= 10)]); b = np.nanmean(r[(x >= 41) & (x <= 50)])
    m = (x >= 31) & (x <= 50)
    sl = np.polyfit(x[m], np.nan_to_num(r[m], nan=np.nanmean(r[m])), 1)[0] * 10
    ln = np.nanmean(d["response_length/mean"][(x >= 41) & (x <= 50)])
    print(f"{name:<28} {a:>9.4f} {b:>10.4f} {sl:>+19.4f}  {ln:>10.0f}")
x = J["step"]; r = J["actor/probe/rho"]
for lo, hi in [(41, 50), (91, 100), (141, 150), (191, 200), (241, 250)]:
    m = (x >= lo) & (x <= hi)
    if m.sum():
        print(f"JustRL direct rho @{lo}-{hi}: {np.nanmean(r[m]):.4f}   "
              f"len {np.nanmean(J['response_length/mean'][m]):.0f}")
