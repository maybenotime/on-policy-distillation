"""Entropy-shape figure: teacher vs student per-token entropy, before vs after OPD.

usage: python plot_entropy_shape.py <stu_pre.npz> <tea_pre.npz> <stu_post.npz> <tea_post.npz> <out.png> <title>
Each npz holds 1000 per-token entropy arrays (key = sequence index); student and
teacher files of a pair are aligned token-by-token on the same rollouts.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = 9           # moving-average window for the curve panels
WIN = 500       # tokens shown in the curve panels (both must be comparable)
MINLEN = 25

def load(f):
    d = np.load(f, allow_pickle=True)
    return [d[str(i)].astype(np.float32) for i in range(len(d.files))]

def ma(x, w=W):
    if len(x) < w:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="valid")

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
    """Sequence whose rho over the displayed window is closest to the group median."""
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
    ax.fill_between(x, sm, tm, color="#e67e22", alpha=0.55, lw=0,
                    label="mismatch", zorder=1)
    ax.plot(x, tm, color=C_T, lw=1.8, label="teacher", zorder=3)
    ax.plot(x, sm, color=C_S, lw=1.8, label="student", zorder=3)
    gap = float(np.mean(np.abs(sm - tm)))
    ax.set_title(title, fontsize=11, pad=8)
    ax.text(0.98, 0.95, f"$\\rho$ = {rho:.2f}\nmean |ΔH| = {gap:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9.5,
            bbox=dict(fc="white", ec="#bdc3c7", alpha=0.9, pad=4))
    ax.set_xlabel(f"token position (first {WIN} tokens of one rollout)", fontsize=9.5)
    if show_ylabel:
        ax.set_ylabel(f"entropy (nats, {W}-token moving avg)", fontsize=9.5)
    ax.tick_params(labelsize=8.5)
    ax.grid(alpha=0.25, lw=0.5)

def main():
    sp_f, tp_f, sq_f, tq_f, out, title = sys.argv[1:7]
    sp, tp, sq, tq = load(sp_f), load(tp_f), load(sq_f), load(tq_f)
    rp, rq = per_seq_rho(sp, tp), per_seq_rho(sq, tq)
    ip, iq = pick(rp, sp, tp), pick(rq, sq, tq)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3),
                             gridspec_kw=dict(width_ratios=[1, 1, 0.75]))
    curve_panel(axes[0], sp[ip], tp[ip],
                "before OPD — base student rolls out", True)
    curve_panel(axes[1], sq[iq], tq[iq],
                "after OPD — trained student rolls out", False)
    ymin = min(axes[0].get_ylim()[0], axes[1].get_ylim()[0])
    ymax = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
    axes[0].set_ylim(ymin, ymax); axes[1].set_ylim(ymin, ymax)
    axes[0].legend(fontsize=9, loc="upper left", framealpha=0.9)

    ax = axes[2]
    bins = np.linspace(0.3, 1.0, 45)
    ax.hist(rp[~np.isnan(rp)], bins=bins, color="#7f8c8d", alpha=0.75,
            label=f"before  (median {np.nanmedian(rp):.3f})")
    ax.hist(rq[~np.isnan(rq)], bins=bins, color="#27ae60", alpha=0.75,
            label=f"after  (median {np.nanmedian(rq):.3f})")
    ax.axvline(np.nanmedian(rp), color="#2c3e50", ls="--", lw=1.2)
    ax.axvline(np.nanmedian(rq), color="#145a32", ls="--", lw=1.2)
    ax.set_title("all 1000 rollouts", fontsize=11, pad=8)
    ax.set_xlabel(r"per-rollout $\rho$(student H, teacher H)", fontsize=9.5)
    ax.set_ylabel("rollouts", fontsize=9.5)
    ax.tick_params(labelsize=8.5)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=8.5, loc="upper left")

    fig.suptitle(title, fontsize=12.5, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=160)
    print("saved", out)
    print(f"pre  rho mean {np.nanmean(rp):.4f}  example seq {ip} (len {len(sp[ip])}, rho {rp[ip]:.3f})")
    print(f"post rho mean {np.nanmean(rq):.4f}  example seq {iq} (len {len(sq[iq])}, rho {rq[iq]:.3f})")

main()
