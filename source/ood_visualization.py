"""
Visualization for OOD analysis results.

Generates 4 aggregate figures + individual rollout case studies:
  1. Cumulative PPL vs position (correct vs incorrect)
  2. Teacher completion accuracy vs position (correct vs incorrect)
  3. Teacher entropy vs position (correct vs incorrect)
  4. Combined three-metric plot (shared x-axis, dual y-axes)
  5. Individual rollout examples (1 correct + 1 incorrect case study each)

Usage:
    python source/ood_visualization.py \
        --input_path results/ood_analysis.pkl \
        --output_dir results/ \
        --format pdf
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


NUM_BINS = 100
DEFAULT_PPL_WINDOW_SIZE = 50


def compute_sliding_window_ppl(logps, window_size: int = DEFAULT_PPL_WINDOW_SIZE) -> np.ndarray:
    """Compute sliding window PPL from per-token log probs. Works with torch tensors or numpy arrays."""
    if isinstance(logps, torch.Tensor):
        logps = logps.float().numpy()
    logps = np.asarray(logps, dtype=np.float64)
    T = len(logps)
    result = np.full(T, np.nan)
    cumsum = np.cumsum(logps)
    for i in range(T):
        start = max(0, i - window_size + 1)
        window_len = i - start + 1
        window_sum = cumsum[i] - (cumsum[start - 1] if start > 0 else 0.0)
        result[i] = np.exp(-window_sum / window_len)
    return result


def load_results(input_path: str) -> dict:
    with open(input_path, "rb") as f:
        return pickle.load(f)


def bin_by_relative_position(values_list: list[np.ndarray], num_bins: int = NUM_BINS):
    """
    Bin variable-length per-token arrays into fixed percentile buckets.
    Each rollout is first reduced to one value per bin (mean within that rollout's
    tokens falling in the bin), then we average across rollouts for equal weighting.
    Returns (bin_centers, mean_per_bin, ci_lower, ci_upper).
    """
    bin_centers = np.linspace(0, 100, num_bins)
    per_rollout = np.full((len(values_list), num_bins), np.nan)

    for r_idx, values in enumerate(values_list):
        T = len(values)
        if T == 0:
            continue
        bin_accum = [[] for _ in range(num_bins)]
        for t in range(T):
            bin_idx = min(int(t / T * num_bins), num_bins - 1)
            bin_accum[bin_idx].append(float(values[t]))
        for b_idx, b in enumerate(bin_accum):
            if b:
                per_rollout[r_idx, b_idx] = np.mean(b)

    means = np.full(num_bins, np.nan)
    ci_lo = np.full(num_bins, np.nan)
    ci_hi = np.full(num_bins, np.nan)

    for i in range(num_bins):
        col = per_rollout[:, i]
        valid = col[~np.isnan(col)]
        if len(valid) >= 2:
            means[i] = np.mean(valid)
            se = np.std(valid, ddof=1) / np.sqrt(len(valid))
            ci_lo[i] = means[i] - 1.96 * se
            ci_hi[i] = means[i] + 1.96 * se
        elif len(valid) == 1:
            means[i] = valid[0]
            ci_lo[i] = valid[0]
            ci_hi[i] = valid[0]

    return bin_centers, means, ci_lo, ci_hi


def split_by_correctness(records):
    correct = [r for r in records if r["is_correct"]]
    incorrect = [r for r in records if not r["is_correct"]]
    return correct, incorrect


SKIP_PREFIX_FRAC = 0.05


def trim_prefix(values_list: list[np.ndarray], skip_frac: float = SKIP_PREFIX_FRAC) -> list[np.ndarray]:
    """Drop the first skip_frac of each sequence to avoid unstable early-token values."""
    return [v[int(len(v) * skip_frac):] for v in values_list]


def plot_metric_by_correctness(
    ax, correct_records, incorrect_records, metric_key, ylabel, title,
    num_bins=NUM_BINS, use_binning=True,
):
    """Plot a per-token metric split by correctness with CI bands."""
    for group, label, color in [
        (correct_records, "Correct", "#2ecc71"),
        (incorrect_records, "Incorrect", "#e74c3c"),
    ]:
        if not group:
            continue
        values_list = [np.array(r[metric_key].float()) if hasattr(r[metric_key], 'float') else np.array(r[metric_key])
                       for r in group]

        if metric_key in ("cumulative_ppl", "sliding_window_ppl"):
            values_list = trim_prefix(values_list)

        if use_binning:
            x, mean, ci_lo, ci_hi = bin_by_relative_position(values_list, num_bins)
            mask = ~np.isnan(mean)
            ax.plot(x[mask], mean[mask], label=f"{label} (n={len(group)})", color=color, linewidth=1.5)
            ax.fill_between(x[mask], ci_lo[mask], ci_hi[mask], alpha=0.2, color=color)

    ax.set_xlabel("Relative Sequence Position (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_accuracy_by_correctness(ax, correct_records, incorrect_records):
    """Plot teacher completion accuracy at sampled positions."""
    for group, label, color in [
        (correct_records, "Correct", "#2ecc71"),
        (incorrect_records, "Incorrect", "#e74c3c"),
    ]:
        if not group:
            continue
        fractions = np.array(group[0]["sample_fractions"]) * 100
        acc_matrix = np.array([r["completion_accuracy"] for r in group])
        mean_acc = np.mean(acc_matrix, axis=0)
        if len(group) >= 2:
            se = np.std(acc_matrix, axis=0, ddof=1) / np.sqrt(len(group))
        else:
            se = np.zeros_like(mean_acc)

        ax.plot(fractions, mean_acc, marker="o", markersize=4,
                label=f"{label} (n={len(group)})", color=color, linewidth=1.5)
        ax.fill_between(fractions, mean_acc - 1.96 * se, mean_acc + 1.96 * se,
                        alpha=0.2, color=color)

    ax.set_xlabel("Relative Sequence Position (%)")
    ax.set_ylabel("Teacher Completion Accuracy")
    ax.set_title("Teacher Completion Accuracy vs Position")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)


def plot_combined(axes, correct_records, incorrect_records):
    """Three metrics on two subplots (correct / incorrect) with dual y-axes."""
    for ax, group, group_label in [
        (axes[0], correct_records, "Correct"),
        (axes[1], incorrect_records, "Incorrect"),
    ]:
        if not group:
            continue

        ppl_values = [np.array(r["sliding_window_ppl"]) if isinstance(r["sliding_window_ppl"], np.ndarray)
                      else np.array(r["sliding_window_ppl"].float()) if hasattr(r["sliding_window_ppl"], 'float')
                      else np.array(r["sliding_window_ppl"]) for r in group]
        x_ppl, mean_ppl, _, _ = bin_by_relative_position(ppl_values)
        mask_ppl = ~np.isnan(mean_ppl)
        line_ppl, = ax.plot(x_ppl[mask_ppl], mean_ppl[mask_ppl], color="#3498db",
                             linewidth=1.5, label="Sliding Window PPL")
        ax.set_xlabel("Relative Sequence Position (%)")
        ax.set_ylabel("Sliding Window PPL", color="#3498db")
        ax.tick_params(axis="y", labelcolor="#3498db")

        ax2 = ax.twinx()
        entropy_values = [np.array(r["teacher_entropy"].float()) if hasattr(r["teacher_entropy"], 'float')
                          else np.array(r["teacher_entropy"]) for r in group]
        x_ent, mean_ent, _, _ = bin_by_relative_position(entropy_values)
        mask_ent = ~np.isnan(mean_ent)
        line_ent, = ax2.plot(x_ent[mask_ent], mean_ent[mask_ent], color="#e67e22",
                              linewidth=1.5, linestyle="--", label="Teacher Entropy")
        ax2.set_ylabel("Teacher Entropy (nats)", color="#e67e22")
        ax2.tick_params(axis="y", labelcolor="#e67e22")

        ax3 = ax.twinx()
        ax3.spines["right"].set_position(("axes", 1.15))
        fractions = np.array(group[0]["sample_fractions"]) * 100
        acc_matrix = np.array([r["completion_accuracy"] for r in group])
        mean_acc = np.mean(acc_matrix, axis=0)
        line_acc, = ax3.plot(fractions, mean_acc, color="#2ecc71", linewidth=1.5,
                              linestyle="-.", marker="o", markersize=3, label="Completion Accuracy")
        ax3.set_ylabel("Completion Accuracy", color="#2ecc71")
        ax3.tick_params(axis="y", labelcolor="#2ecc71")
        ax3.set_ylim(-0.05, 1.05)

        lines = [line_ppl, line_ent, line_acc]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc="upper left")
        ax.set_title(f"{group_label} (n={len(group)})")
        ax.grid(True, alpha=0.3)


def select_example_rollouts(correct, incorrect, num_examples=2):
    """Select example rollouts: pick by median length to avoid outliers."""
    examples = {}
    for group, label in [(correct, "correct"), (incorrect, "incorrect")]:
        if not group:
            continue
        lengths = [len(r["completion_token_ids"]) for r in group]
        sorted_indices = np.argsort(lengths)
        # Pick num_examples spread across the group (median-ish)
        step = max(1, len(sorted_indices) // (num_examples + 1))
        picked = [sorted_indices[step * (i + 1)] for i in range(min(num_examples, len(sorted_indices)))]
        examples[label] = [group[i] for i in picked]
    return examples


def plot_single_rollout_ppl(ax, record):
    """Plot raw cumulative PPL curve for a single rollout."""
    ppl = np.array(record["cumulative_ppl"].float()) if hasattr(record["cumulative_ppl"], 'float') else np.array(record["cumulative_ppl"])
    T = len(ppl)
    x = np.linspace(0, 100, T)
    color = "#2ecc71" if record["is_correct"] else "#e74c3c"
    label = "Correct" if record["is_correct"] else "Incorrect"
    ax.plot(x, ppl, color=color, linewidth=1.2, label=label)
    ax.set_xlabel("Relative Sequence Position (%)")
    ax.set_ylabel("Cumulative PPL")
    ax.grid(True, alpha=0.3)


def plot_single_rollout_entropy(ax, record):
    """Plot raw teacher entropy curve for a single rollout."""
    entropy = np.array(record["teacher_entropy"].float()) if hasattr(record["teacher_entropy"], 'float') else np.array(record["teacher_entropy"])
    T = len(entropy)
    x = np.linspace(0, 100, T)
    color = "#2ecc71" if record["is_correct"] else "#e74c3c"
    ax.plot(x, entropy, color=color, linewidth=1.0, alpha=0.8)
    ax.set_xlabel("Relative Sequence Position (%)")
    ax.set_ylabel("Entropy (nats)")
    ax.grid(True, alpha=0.3)


def plot_single_rollout_accuracy(ax, record):
    """Plot teacher completion accuracy at sampled positions for a single rollout."""
    fractions = np.array(record["sample_fractions"]) * 100
    acc = np.array(record["completion_accuracy"])
    color = "#2ecc71" if record["is_correct"] else "#e74c3c"
    ax.plot(fractions, acc, marker="o", markersize=5, color=color, linewidth=1.5)
    ax.set_xlabel("Relative Sequence Position (%)")
    ax.set_ylabel("Completion Accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)


def plot_single_rollout_combined(ax, record):
    """Three metrics on one plot for a single rollout."""
    ppl = np.array(record["cumulative_ppl"].float()) if hasattr(record["cumulative_ppl"], 'float') else np.array(record["cumulative_ppl"])
    entropy = np.array(record["teacher_entropy"].float()) if hasattr(record["teacher_entropy"], 'float') else np.array(record["teacher_entropy"])
    T = len(ppl)
    x_token = np.linspace(0, 100, T)

    line_ppl, = ax.plot(x_token, ppl, color="#3498db", linewidth=1.2, label="Cumulative PPL")
    ax.set_xlabel("Relative Sequence Position (%)")
    ax.set_ylabel("Cumulative PPL", color="#3498db")
    ax.tick_params(axis="y", labelcolor="#3498db")

    ax2 = ax.twinx()
    line_ent, = ax2.plot(x_token, entropy, color="#e67e22", linewidth=1.0, linestyle="--", label="Entropy")
    ax2.set_ylabel("Entropy (nats)", color="#e67e22")
    ax2.tick_params(axis="y", labelcolor="#e67e22")

    ax3 = ax.twinx()
    ax3.spines["right"].set_position(("axes", 1.15))
    fractions = np.array(record["sample_fractions"]) * 100
    acc = np.array(record["completion_accuracy"])
    line_acc, = ax3.plot(fractions, acc, color="#2ecc71", linewidth=1.5, linestyle="-.", marker="o", markersize=3, label="Accuracy")
    ax3.set_ylabel("Accuracy", color="#2ecc71")
    ax3.tick_params(axis="y", labelcolor="#2ecc71")
    ax3.set_ylim(-0.05, 1.05)

    lines = [line_ppl, line_ent, line_acc]
    ax.legend(lines, [l.get_label() for l in lines], loc="upper left")
    ax.grid(True, alpha=0.3)


def plot_case_studies(examples, output_dir, tag, fmt, dpi):
    """Generate per-rollout case study figures."""
    for label, rollouts in examples.items():
        for i, record in enumerate(rollouts):
            T = len(record["completion_token_ids"])
            prompt_short = record["prompt"][:80].replace("\n", " ").replace("$", r"\$")
            suptitle = f"[{label.upper()}] len={T}  prompt: {prompt_short}..."

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            plot_single_rollout_ppl(axes[0, 0], record)
            axes[0, 0].set_title("Cumulative PPL")

            plot_single_rollout_entropy(axes[0, 1], record)
            axes[0, 1].set_title("Teacher Entropy")

            plot_single_rollout_accuracy(axes[1, 0], record)
            axes[1, 0].set_title("Completion Accuracy")

            plot_single_rollout_combined(axes[1, 1], record)
            axes[1, 1].set_title("Combined")

            fig.suptitle(suptitle, fontsize=11, y=0.98)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            fname = f"ood_case_{label}_{i}_{tag}.{fmt}"
            fig.savefig(output_dir / fname, dpi=dpi, bbox_inches="tight")
            print(f"Saved case study: {fname}")
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize OOD analysis results")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_dir", default="results/")
    parser.add_argument("--format", default="pdf", choices=["pdf", "png"])
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--ppl_window_size", type=int, default=DEFAULT_PPL_WINDOW_SIZE,
                        help="Window size for sliding window PPL computation")
    args = parser.parse_args()

    data = load_results(args.input_path)
    records = data["records"]
    metadata = data["metadata"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for r in records:
        if "sliding_window_ppl" not in r and "teacher_logps" in r:
            r["sliding_window_ppl"] = compute_sliding_window_ppl(r["teacher_logps"], args.ppl_window_size)

    correct, incorrect = split_by_correctness(records)
    print(f"Loaded {len(records)} records: {len(correct)} correct, {len(incorrect)} incorrect")

    tag = f"{Path(metadata['student_model']).name}_vs_{Path(metadata['teacher_model']).name}"

    # Figure 1: Cumulative PPL
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    plot_metric_by_correctness(ax1, correct, incorrect, "cumulative_ppl",
                               "Cumulative PPL", "Teacher Cumulative PPL on Student Trajectories")
    fig1.tight_layout()
    fig1.savefig(output_dir / f"ood_ppl_{tag}.{args.format}", dpi=args.dpi)
    print(f"Saved figure 1: ood_ppl_{tag}.{args.format}")

    # Figure 1b: Sliding Window PPL
    fig1b, ax1b = plt.subplots(figsize=(10, 6))
    plot_metric_by_correctness(ax1b, correct, incorrect, "sliding_window_ppl",
                               f"Sliding Window PPL (w={args.ppl_window_size})",
                               f"Teacher Sliding Window PPL (window={args.ppl_window_size}) on Student Trajectories")
    fig1b.tight_layout()
    fig1b.savefig(output_dir / f"ood_sliding_ppl_{tag}.{args.format}", dpi=args.dpi)
    print(f"Saved figure: ood_sliding_ppl_{tag}.{args.format}")

    # Figure 2: Teacher Completion Accuracy
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    plot_accuracy_by_correctness(ax2, correct, incorrect)
    fig2.tight_layout()
    fig2.savefig(output_dir / f"ood_accuracy_{tag}.{args.format}", dpi=args.dpi)
    print(f"Saved figure 2: ood_accuracy_{tag}.{args.format}")

    # Figure 3: Teacher Entropy
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    plot_metric_by_correctness(ax3, correct, incorrect, "teacher_entropy",
                               "Entropy (nats)", "Teacher Entropy on Student Trajectories")
    fig3.tight_layout()
    fig3.savefig(output_dir / f"ood_entropy_{tag}.{args.format}", dpi=args.dpi)
    print(f"Saved figure 3: ood_entropy_{tag}.{args.format}")

    # Figure 4: Combined three-metric plot (correct vs incorrect)
    fig4, axes4 = plt.subplots(1, 2, figsize=(22, 6))
    fig4.suptitle("Combined Metrics vs Sequence Position", fontsize=14)
    plot_combined(axes4, correct, incorrect)
    fig4.tight_layout(rect=[0, 0, 1, 0.95])
    fig4.savefig(output_dir / f"ood_combined_{tag}.{args.format}", dpi=args.dpi, bbox_inches="tight")
    print(f"Saved figure 4: ood_combined_{tag}.{args.format}")

    # Figure 5+: Individual rollout case studies
    examples = select_example_rollouts(correct, incorrect, num_examples=2)
    plot_case_studies(examples, output_dir, tag, args.format, args.dpi)

    plt.close("all")
    print("Done.")


if __name__ == "__main__":
    main()
