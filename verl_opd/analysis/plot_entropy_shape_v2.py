"""Entropy-shape figure, two panels (Before OPD / After OPD): teacher vs student per-token entropy on one rollout.
Same example-sequence selection and 9-token moving average as plot_entropy_shape.py; the rho histogram panel,
the suptitle and the |dH| annotation are dropped (user, 9-25).

usage: python plot_entropy_shape_v2.py <stu_pre.npz> <tea_pre.npz> <stu_post.npz> <tea_post.npz> <out_stem>
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = 9
WIN = 500
MINLEN = 25


def load(f):
    d = np.load(f, allow_pickle=True)
    return [d[str(i)].astype(np.float32) for i in range(len(d.files))]


def ma(x, w=W):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def per_seq_rho(S, T):
    out = []
    for s, t in zip(S, T):
        n = min(len(s), len(t))
        if n < MINLEN or s[:n].std() < 1e-6 or t[:n].std() < 1e-6:
            out.append(np.nan)
            continue
        out.append(float(np.corrcoef(s[:n], t[:n])[0, 1]))
    return np.array(out)


def pick(R, S, T):
    med = np.nanmedian(R)
    c = []
    for i in range(len(R)):
        n = min(len(S[i]), len(T[i]))
        if n < WIN:
            continue
        s, t = S[i][:WIN], T[i][:WIN]
        if s.std() < 1e-6 or t.std() < 1e-6:
            continue
        c.append((abs(float(np.corrcoef(s, t)[0, 1]) - med), i))
    return sorted(c)[0][1] if c else int(np.nanargmin(abs(R - med)))


C_T, C_S = "#c0392b", "#2471a3"


def curve_panel(ax, s, t, title, show_ylabel):
    n = min(len(s), len(t), WIN)
    rho = float(np.corrcoef(s[:n], t[:n])[0, 1])
    sm, tm = ma(s[:n]), ma(t[:n])
    x = np.arange(len(sm))
    ax.fill_between(x, sm, tm, color="#e67e22", alpha=0.55, lw=0, label="mismatch", zorder=1)
    ax.plot(x, tm, color=C_T, lw=1.8, label="teacher", zorder=3)
    ax.plot(x, sm, color=C_S, lw=1.8, label="student", zorder=3)
    ax.set_title(title, fontsize=12, pad=8)
    ax.text(0.98, 0.95, f"$\\rho$ = {rho:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=11,
            bbox=dict(fc="white", ec="#bdc3c7", alpha=0.9, pad=4))
    ax.set_xlabel("Token Position", fontsize=11)
    if show_ylabel:
        ax.set_ylabel("Entropy", fontsize=11)
    ax.tick_params(labelsize=9)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=9.5, loc="upper left", framealpha=0.9)


def main():
    sp_f, tp_f, sq_f, tq_f, out = sys.argv[1:6]
    sp, tp, sq, tq = load(sp_f), load(tp_f), load(sq_f), load(tq_f)
    rp, rq = per_seq_rho(sp, tp), per_seq_rho(sq, tq)
    ip, iq = pick(rp, sp, tp), pick(rq, sq, tq)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    curve_panel(axes[0], sp[ip], tp[ip], "Before OPD", True)
    curve_panel(axes[1], sq[iq], tq[iq], "After OPD", False)
    fig.tight_layout()
    fig.savefig(out + ".png", dpi=200)
    fig.savefig(out + ".pdf")
    print("saved", out, f"| pre seq {ip} rho {rp[ip]:.3f} | post seq {iq} rho {rq[iq]:.3f}")


main()
