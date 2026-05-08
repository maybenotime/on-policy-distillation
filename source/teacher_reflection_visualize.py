"""
Visualize teacher reflection probe results.

Reads pkls produced by teacher_reflection_probe.py (one or more) and
generates:
  1) P(reflection | prefix) vs relative position curve (per kw + aggregate),
     with all input pkls overlaid for cross-(student, teacher) comparison.
  2) 4-cell contingency table (reflection ⨯ correctness) at each sampled
     position fraction, per input pkl.
  3) A handful of case studies: sampled continuations that contain a
     reflection keyword AND end up correct — these are the concrete
     "teacher backtracked and recovered" examples.

Usage:
    python source/teacher_reflection_visualize.py \\
        --input_paths results/reflection_probe_Qwen3-1.7B_Qwen3-8B.pkl \\
                      results/reflection_probe_Qwen2.5-7B_Qwen2.5-14B.pkl \\
        --output_dir results/figures/reflection
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def label_for(metadata):
    s = Path(metadata["student_model"]).name
    t = Path(metadata["teacher_model"]).name
    return f"{s} → {t}"


def aggregate_logits_curves(records, num_bins):
    """records: list of {kw_probs_binned: {kw -> array [num_bins]}}.
    Returns dict[kw -> (mean[num_bins], lo, hi)] over records."""
    if not records:
        return {}
    keys = list(records[0]["kw_probs_binned"].keys())
    out = {}
    for kw in keys:
        arr = np.stack([r["kw_probs_binned"][kw] for r in records])  # [n, num_bins]
        # Mean + 95% CI
        mean = np.nanmean(arr, axis=0)
        sem = np.nanstd(arr, axis=0) / np.sqrt(np.sum(~np.isnan(arr), axis=0).clip(min=1))
        ci = 1.96 * sem
        out[kw] = (mean, mean - ci, mean + ci)
    # Aggregate (sum) curve
    agg = np.nansum(np.stack([np.nanmean(np.stack([r["kw_probs_binned"][kw] for r in records]), axis=0)
                              for kw in keys]), axis=0)
    out["__any__"] = (agg, agg, agg)
    return out


def plot_curves(all_data: list[tuple[str, dict]], output_dir: Path, format: str):
    """all_data: list of (label, pkl_dict). One curve per label per kw."""
    # Combined aggregate plot
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]
    for ci, (label, data) in enumerate(all_data):
        records = data.get("logits_records", [])
        if not records:
            continue
        num_bins = len(records[0]["kw_probs_binned"][next(iter(records[0]["kw_probs_binned"]))])
        x = np.linspace(0, 100, num_bins)
        agg = aggregate_logits_curves(records, num_bins)
        m, lo, hi = agg["__any__"]
        ax.plot(x, m, label=label, color=colors[ci % len(colors)], linewidth=2)
    ax.set_xlabel("Relative position in completion (%)")
    ax.set_ylabel("Σ P(reflection token | prefix)")
    ax.set_title("Aggregate: teacher's next-token reflection probability vs position\n"
                 "(sum of {wait, alternatively, actually, instead, hmm, reconsider, rethink} variants)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"reflection_aggregate.{format}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved reflection_aggregate.{format}")

    # Per-keyword grid (one subplot per keyword)
    for label, data in all_data:
        records = data.get("logits_records", [])
        if not records:
            continue
        num_bins = len(records[0]["kw_probs_binned"][next(iter(records[0]["kw_probs_binned"]))])
        x = np.linspace(0, 100, num_bins)
        agg = aggregate_logits_curves(records, num_bins)
        kws = [k for k in agg if k != "__any__"]
        n_rows = (len(kws) + 2) // 3
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows), squeeze=False)
        for i, kw in enumerate(kws):
            ax = axes[i // 3][i % 3]
            m, lo, hi = agg[kw]
            ax.plot(x, m, color="#1f77b4")
            ax.fill_between(x, lo, hi, color="#1f77b4", alpha=0.2)
            ax.set_title(f"P({kw} | prefix)")
            ax.set_xlabel("Position (%)")
            ax.set_ylabel("Probability")
            ax.grid(True, alpha=0.3)
        for j in range(len(kws), n_rows * 3):
            axes[j // 3][j % 3].axis("off")
        fig.suptitle(f"Per-keyword P(reflection | prefix): {label}", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        slug = label.replace(" → ", "__").replace("/", "_").replace(" ", "_")
        fig.savefig(output_dir / f"reflection_perkw_{slug}.{format}", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved reflection_perkw_{slug}.{format}")


def contingency_table(all_data: list[tuple[str, dict]], output_path: Path):
    """For each (label, position_fraction), build the 4-cell table:
    reflection ⨯ correct_after_recovery."""
    lines = ["# Reflection ⨯ Correctness contingency tables\n"]
    for label, data in all_data:
        rec = data.get("sampling_records", [])
        if not rec:
            continue
        lines.append(f"\n## {label}")
        meta = data["metadata"]
        lines.append(f"Filter={meta['filter']}, num_records={len(rec)}, "
                     f"num_samples={meta['num_samples']}, temperature={meta['temperature']}\n")
        for pidx, frac in enumerate(meta["position_fractions"]):
            a = b = c = d = 0
            for r in rec:
                for s in r["positions"][pidx]["samples"]:
                    if s["has_reflection"] and s["is_correct_after_recovery"]:
                        a += 1
                    elif s["has_reflection"] and not s["is_correct_after_recovery"]:
                        b += 1
                    elif not s["has_reflection"] and s["is_correct_after_recovery"]:
                        c += 1
                    else:
                        d += 1
            total = a + b + c + d
            p_reflect = (a + b) / total if total else 0.0
            p_correct_given_refl = a / (a + b) if (a + b) else 0.0
            p_correct_given_no_refl = c / (c + d) if (c + d) else 0.0
            lines.append(f"### Position {frac*100:.0f}%")
            lines.append(f"|              | Correct | Incorrect |")
            lines.append(f"|--------------|---------|-----------|")
            lines.append(f"| Reflection   | {a:5d}   | {b:9d}   |")
            lines.append(f"| No-reflection| {c:5d}   | {d:9d}   |")
            lines.append(f"")
            lines.append(f"- P(reflection in first 100 tokens) = {p_reflect:.3f}")
            lines.append(f"- P(correct | reflection) = {p_correct_given_refl:.3f}")
            lines.append(f"- P(correct | no reflection) = {p_correct_given_no_refl:.3f}")
            lines.append(f"- Recovery uplift = {p_correct_given_refl - p_correct_given_no_refl:+.3f}\n")
    output_path.write_text("\n".join(lines))
    print(f"Saved contingency table → {output_path}")


def write_case_studies(all_data, output_path, max_examples_per_label: int = 8):
    lines = ["# Reflection-Recovery Case Studies\n",
             "Concrete examples where the teacher emitted a reflection keyword "
             "(in first 100 tokens of its continuation) AND ended up producing "
             "the correct final answer. Deduplicated to one example per "
             "(trajectory, position) — at most one entry per student trajectory.\n"]
    for label, data in all_data:
        rec = data.get("sampling_records", [])
        if not rec:
            continue
        lines.append(f"\n## {label}\n")
        examples = []
        for r in rec:
            for pidx, pos in enumerate(r["positions"]):
                for s in pos["samples"]:
                    if s["has_reflection"] and s["is_correct_after_recovery"]:
                        examples.append({
                            "trajectory_idx": r["trajectory_idx"],
                            "position_fraction": pos["position_fraction"],
                            "first_chars": s["first_chars"],
                        })
        if not examples:
            lines.append("(no recovery examples found)\n")
            continue
        # Dedupe to ONE example per trajectory (prefer later positions).
        examples.sort(key=lambda e: -e["position_fraction"])
        seen_traj = set()
        unique = []
        for e in examples:
            if e["trajectory_idx"] in seen_traj:
                continue
            seen_traj.add(e["trajectory_idx"])
            unique.append(e)
        for ex in unique[:max_examples_per_label]:
            lines.append(f"\n**Trajectory {ex['trajectory_idx']}, prefix at "
                         f"{ex['position_fraction']*100:.0f}%**")
            lines.append("```")
            lines.append(ex["first_chars"])
            lines.append("```")
        lines.append(f"\n_Total recovery examples: {len(examples)} across "
                     f"{len(seen_traj)} unique trajectories._\n")
    output_path.write_text("\n".join(lines))
    print(f"Saved case studies → {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_paths", nargs="+", required=True)
    parser.add_argument("--output_dir", default="results/figures/reflection")
    parser.add_argument("--format", default="png", choices=["png", "pdf"])
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_data = []
    for p in args.input_paths:
        with open(p, "rb") as f:
            data = pickle.load(f)
        label = label_for(data["metadata"])
        all_data.append((label, data))
        print(f"Loaded {label} from {p}")

    plot_curves(all_data, out_dir, args.format)
    contingency_table(all_data, out_dir / "contingency.md")
    write_case_studies(all_data, out_dir / "case_studies.md")
    print("Done.")


if __name__ == "__main__":
    main()
