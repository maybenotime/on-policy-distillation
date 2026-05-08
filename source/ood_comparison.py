"""
Cross-student comparison visualization for OOD analysis.

Loads multiple pkl files (one per student size, same teacher) and generates
overlay plots comparing how different-capacity students diverge from the teacher.

Usage:
    python source/ood_comparison.py \
        --input_paths results/ood_gsm8k_100_Qwen2.5-0.5B-Instruct_Qwen2.5-14B-Instruct.pkl \
                      results/ood_gsm8k_100_Qwen2.5-3B-Instruct_Qwen2.5-14B-Instruct.pkl \
                      results/ood_gsm8k_100_Qwen2.5-7B-Instruct_Qwen2.5-14B-Instruct.pkl \
        --output_dir figures/comparison/ \
        --format png
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ood_visualization import (
    bin_by_relative_position,
    compute_sliding_window_ppl,
    split_by_correctness,
    trim_prefix,
    NUM_BINS,
    DEFAULT_PPL_WINDOW_FRAC,
)

STUDENT_COLORS = {
    "0.5B": "#1f77b4",
    "1.7B": "#9467bd",
    "1.7B-Base": "#8c564b",
    "3B": "#ff7f0e",
    "7B": "#2ca02c",
    "14B": "#d62728",
}

STUDENT_ORDER = ["0.5B", "1.7B-Base", "1.7B", "3B", "7B", "14B"]


def get_student_label(model_name: str) -> str:
    name = Path(model_name).name
    # Prefer the most specific match: -Base variants before bare size tags.
    if "1.7B-Base" in name:
        return "1.7B-Base"
    if "1.7B" in name:
        return "1.7B"
    for tag in STUDENT_ORDER:
        if tag in name and tag not in ("1.7B", "1.7B-Base"):
            return tag
    return name


def load_all(input_paths: list[str]) -> list[tuple[str, dict]]:
    results = []
    for p in input_paths:
        with open(p, "rb") as f:
            data = pickle.load(f)
        label = get_student_label(data["metadata"]["student_model"])
        for r in data["records"]:
            if "teacher_logps" in r:
                r["sliding_window_ppl"] = compute_sliding_window_ppl(r["teacher_logps"])
        results.append((label, data))
    results.sort(key=lambda x: STUDENT_ORDER.index(x[0]) if x[0] in STUDENT_ORDER else 99)
    return results


def _to_numpy(v):
    if isinstance(v, torch.Tensor):
        return v.float().numpy()
    return np.asarray(v)


def plot_metric_overlay(ax, all_data, metric_key, ylabel, title, group_filter=None, trim=False):
    for label, data in all_data:
        records = data["records"]
        if group_filter == "correct":
            records = [r for r in records if r["is_correct"]]
        elif group_filter == "incorrect":
            records = [r for r in records if not r["is_correct"]]

        if not records:
            continue

        has_key = any(metric_key in r for r in records)
        if not has_key:
            continue

        values_list = [_to_numpy(r[metric_key]) for r in records if metric_key in r]
        if trim:
            values_list = trim_prefix(values_list)

        color = STUDENT_COLORS.get(label, "#999999")
        x, mean, ci_lo, ci_hi = bin_by_relative_position(values_list, NUM_BINS)
        mask = ~np.isnan(mean)
        n = len(values_list)
        ax.plot(x[mask], mean[mask], label=f"{label} (n={n})", color=color, linewidth=1.5)
        ax.fill_between(x[mask], ci_lo[mask], ci_hi[mask], alpha=0.15, color=color)

    ax.set_xlabel("Relative Sequence Position (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_accuracy_overlay(ax, all_data, title, group_filter=None):
    for label, data in all_data:
        records = data["records"]
        if group_filter == "correct":
            records = [r for r in records if r["is_correct"]]
        elif group_filter == "incorrect":
            records = [r for r in records if not r["is_correct"]]

        if not records:
            continue

        fractions = np.array(records[0]["sample_fractions"]) * 100
        acc_matrix = np.array([r["completion_accuracy"] for r in records])
        mean_acc = np.mean(acc_matrix, axis=0)
        se = np.std(acc_matrix, axis=0, ddof=1) / np.sqrt(len(records)) if len(records) >= 2 else np.zeros_like(mean_acc)

        color = STUDENT_COLORS.get(label, "#999999")
        n = len(records)
        ax.plot(fractions, mean_acc, marker="o", markersize=4, label=f"{label} (n={n})",
                color=color, linewidth=1.5)
        ax.fill_between(fractions, mean_acc - 1.96 * se, mean_acc + 1.96 * se, alpha=0.15, color=color)

    ax.set_xlabel("Relative Sequence Position (%)")
    ax.set_ylabel("Teacher Completion Accuracy")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)


def plot_reward_overlay(ax, all_data, title, group_filter=None):
    for label, data in all_data:
        records = data["records"]
        if group_filter == "correct":
            records = [r for r in records if r["is_correct"]]
        elif group_filter == "incorrect":
            records = [r for r in records if not r["is_correct"]]

        has_rewards = any("token_rewards" in r for r in records)
        if not has_rewards or not records:
            continue

        values_list = [_to_numpy(r["token_rewards"]) for r in records if "token_rewards" in r]

        color = STUDENT_COLORS.get(label, "#999999")
        x, mean, ci_lo, ci_hi = bin_by_relative_position(values_list, NUM_BINS)
        mask = ~np.isnan(mean)
        n = len(values_list)
        ax.plot(x[mask], mean[mask], label=f"{label} (n={n})", color=color, linewidth=1.5)
        ax.fill_between(x[mask], ci_lo[mask], ci_hi[mask], alpha=0.15, color=color)

    ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Relative Sequence Position (%)")
    ax.set_ylabel("Token Reward (log p_teacher - log p_student)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_positive_reward_fraction(ax, all_data, title, group_filter=None):
    for label, data in all_data:
        records = data["records"]
        if group_filter == "correct":
            records = [r for r in records if r["is_correct"]]
        elif group_filter == "incorrect":
            records = [r for r in records if not r["is_correct"]]

        has_rewards = any("token_rewards" in r for r in records)
        if not has_rewards or not records:
            continue

        frac_positive_list = []
        for r in records:
            if "token_rewards" not in r:
                continue
            rewards = _to_numpy(r["token_rewards"])
            frac_positive_list.append((rewards > 0).astype(np.float64))

        color = STUDENT_COLORS.get(label, "#999999")
        x, mean, ci_lo, ci_hi = bin_by_relative_position(frac_positive_list, NUM_BINS)
        mask = ~np.isnan(mean)
        n = len(frac_positive_list)
        ax.plot(x[mask], mean[mask], label=f"{label} (n={n})", color=color, linewidth=1.5)
        ax.fill_between(x[mask], ci_lo[mask], ci_hi[mask], alpha=0.15, color=color)

    ax.axhline(y=0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Relative Sequence Position (%)")
    ax.set_ylabel("Fraction of Positive Rewards")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)


def main():
    parser = argparse.ArgumentParser(description="Cross-student OOD comparison visualization")
    parser.add_argument("--input_paths", nargs="+", required=True)
    parser.add_argument("--output_dir", default="figures/comparison/")
    parser.add_argument("--format", default="png", choices=["pdf", "png"])
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--dataset_tag", default=None,
                        help="Dataset tag for filenames (auto-detected if not given)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_data = load_all(args.input_paths)
    student_labels = [label for label, _ in all_data]
    print(f"Loaded {len(all_data)} student configs: {student_labels}")

    for label, data in all_data:
        correct, incorrect = split_by_correctness(data["records"])
        print(f"  {label}: {len(correct)} correct, {len(incorrect)} incorrect")

    dataset_tag = args.dataset_tag
    if dataset_tag is None:
        fname = Path(args.input_paths[0]).stem
        parts = fname.split("_")
        for i, p in enumerate(parts):
            if p in ("gsm8k", "math", "train") and i + 1 < len(parts):
                dataset_tag = f"{p}_{parts[i+1]}"
                break
        if dataset_tag is None:
            dataset_tag = "unknown"

    teacher_name = Path(all_data[0][1]["metadata"]["teacher_model"]).name
    tag = f"{dataset_tag}_teacher_{teacher_name}"

    # 1. Sliding Window PPL — correct/incorrect panels
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(f"Sliding Window PPL by Student Size (Teacher: {teacher_name}, Dataset: {dataset_tag})", fontsize=13)
    plot_metric_overlay(axes[0], all_data, "sliding_window_ppl",
                        "Sliding Window PPL", "Correct Trajectories",
                        group_filter="correct", trim=True)
    plot_metric_overlay(axes[1], all_data, "sliding_window_ppl",
                        "Sliding Window PPL", "Incorrect Trajectories",
                        group_filter="incorrect", trim=True)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_dir / f"cmp_sliding_ppl_{tag}.{args.format}", dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: cmp_sliding_ppl_{tag}.{args.format}")
    plt.close(fig)

    # 2. Teacher Completion Accuracy — correct/incorrect panels
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(f"Teacher Completion Accuracy by Student Size (Teacher: {teacher_name}, Dataset: {dataset_tag})", fontsize=13)
    plot_accuracy_overlay(axes[0], all_data, "Correct Trajectories", group_filter="correct")
    plot_accuracy_overlay(axes[1], all_data, "Incorrect Trajectories", group_filter="incorrect")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_dir / f"cmp_accuracy_{tag}.{args.format}", dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: cmp_accuracy_{tag}.{args.format}")
    plt.close(fig)

    # 3. Teacher Entropy — correct/incorrect panels
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(f"Teacher Entropy by Student Size (Teacher: {teacher_name}, Dataset: {dataset_tag})", fontsize=13)
    plot_metric_overlay(axes[0], all_data, "teacher_entropy",
                        "Entropy (nats)", "Correct Trajectories", group_filter="correct")
    plot_metric_overlay(axes[1], all_data, "teacher_entropy",
                        "Entropy (nats)", "Incorrect Trajectories", group_filter="incorrect")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_dir / f"cmp_entropy_{tag}.{args.format}", dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: cmp_entropy_{tag}.{args.format}")
    plt.close(fig)

    # 4. Token Rewards — correct/incorrect panels (skip students without this field)
    has_any_rewards = any(
        any("token_rewards" in r for r in data["records"])
        for _, data in all_data
    )
    if has_any_rewards:
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        fig.suptitle(f"Token Rewards by Student Size (Teacher: {teacher_name}, Dataset: {dataset_tag})", fontsize=13)
        plot_reward_overlay(axes[0], all_data, "Correct Trajectories", group_filter="correct")
        plot_reward_overlay(axes[1], all_data, "Incorrect Trajectories", group_filter="incorrect")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(output_dir / f"cmp_rewards_{tag}.{args.format}", dpi=args.dpi, bbox_inches="tight")
        print(f"Saved: cmp_rewards_{tag}.{args.format}")
        plt.close(fig)

        # 5. Positive reward fraction — correct/incorrect panels
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        fig.suptitle(f"Positive Reward Fraction by Student Size (Teacher: {teacher_name}, Dataset: {dataset_tag})", fontsize=13)
        plot_positive_reward_fraction(axes[0], all_data,
                                      "Correct Trajectories",
                                      group_filter="correct")
        plot_positive_reward_fraction(axes[1], all_data,
                                      "Incorrect Trajectories",
                                      group_filter="incorrect")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(output_dir / f"cmp_reward_quality_{tag}.{args.format}", dpi=args.dpi, bbox_inches="tight")
        print(f"Saved: cmp_reward_quality_{tag}.{args.format}")
        plt.close(fig)
    else:
        print("Skipping reward plots — no pkl files contain token_rewards field")

    plt.close("all")
    print("Done.")


if __name__ == "__main__":
    main()
