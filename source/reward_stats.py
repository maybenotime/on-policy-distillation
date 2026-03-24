"""
Static statistics for per-token rewards produced by token_reward_analysis.py.

For each teacher model:
  - Randomly samples --sample_n records from each of the correct / incorrect groups
    and prints per-sample rows.
  - Appends a summary row per group showing the sample-level mean of every metric
    (i.e. compute per-sample stats first, then average across samples in the group).

Usage:
    python reward_stats.py --input results/token_rewards_test.pkl [--sample_n 10] [--seed 42]
"""

import argparse
import pickle
import random

import numpy as np


def sample_stats(rewards: np.ndarray) -> dict:
    """Per-sample stats dict.  Returns NaN fields when the sub-array is empty."""
    pos = rewards[rewards > 0]
    neg = rewards[rewards < 0]
    total = rewards.size

    def _ms(arr):
        if arr.size == 0:
            return float("nan"), float("nan")
        return float(arr.mean()), float(arr.var())

    pos_mean, pos_var = _ms(pos)
    neg_mean, neg_var = _ms(neg)
    return {
        "total":    total,
        "pos_pct":  100 * pos.size / total,
        "pos_mean": pos_mean,
        "pos_var":  pos_var,
        "neg_pct":  100 * neg.size / total,
        "neg_mean": neg_mean,
        "neg_var":  neg_var,
    }


def fmt_row(label: str, correct_str: str, s: dict) -> str:
    def _f(v, fmt):
        return "     nan" if v != v else format(v, fmt)   # nan check
    return (
        f"  {label:<8} {correct_str:<14} {s['total']:>6}  "
        f"{s['pos_pct']:>5.1f}%  {_f(s['pos_mean'], '+9.4f')}  {_f(s['pos_var'], '8.4f')}  "
        f"{s['neg_pct']:>5.1f}%  {_f(s['neg_mean'], '+9.4f')}  {_f(s['neg_var'], '8.4f')}"
    )


def group_mean_row(label: str, correct_str: str, stats_list: list[dict]) -> str:
    """Average each metric across sample-level stats dicts (ignoring NaN)."""
    def nanmean(vals):
        vals = [v for v in vals if v == v]   # drop NaN
        return float(np.mean(vals)) if vals else float("nan")

    keys = ["total", "pos_pct", "pos_mean", "pos_var", "neg_pct", "neg_mean", "neg_var"]
    avg = {k: nanmean([s[k] for s in stats_list]) for k in keys}
    return fmt_row(label, correct_str, avg)


def analyse(input_path: str, sample_n: int | None, seed: int) -> None:
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    metadata      = data["metadata"]
    records       = data["records"]
    teacher_names = metadata["teacher_models"]

    print("=" * 75)
    print(f"Student : {metadata['student_model']}")
    print(f"Samples : {len(records)}")
    print(f"Temp    : {metadata['temperature']}   Max new tokens: {metadata['max_new_tokens']}")
    print("=" * 75)

    correct_all   = [r for r in records if r.get("is_correct") is True]
    incorrect_all = [r for r in records if r.get("is_correct") is False]

    rng = random.Random(seed)
    if sample_n is not None:
        correct_show   = rng.sample(correct_all,   min(sample_n, len(correct_all)))
        incorrect_show = rng.sample(incorrect_all, min(sample_n, len(incorrect_all)))
    else:
        correct_show, incorrect_show = correct_all, incorrect_all

    HDR = (
        f"  {'Sample':<8} {'Correct':<14} {'Tokens':>6}  "
        f"{'Pos%':>6}  {'Pos mean':>9}  {'Pos var':>8}  "
        f"{'Neg%':>6}  {'Neg mean':>9}  {'Neg var':>8}"
    )
    SEP = (
        f"  {'-'*8} {'-'*14} {'-'*6}  "
        f"{'-'*6}  {'-'*9}  {'-'*8}  "
        f"{'-'*6}  {'-'*9}  {'-'*8}"
    )

    for tname in teacher_names:
        print(f"\nTeacher: {tname}")
        print(HDR)
        print(SEP)

        # ---- sampled correct rows ----
        correct_all_stats = []
        for rec in correct_all:
            correct_all_stats.append(
                sample_stats(rec["teacher_rewards"][tname].float().numpy())
            )

        if correct_show:
            print(f"  -- ✓ Correct ({len(correct_all)} total, showing {len(correct_show)}) --")
            for rec in correct_show:
                s = sample_stats(rec["teacher_rewards"][tname].float().numpy())
                print(fmt_row(str(records.index(rec)), "✓", s))

        # ---- sampled incorrect rows ----
        incorrect_all_stats = []
        for rec in incorrect_all:
            incorrect_all_stats.append(
                sample_stats(rec["teacher_rewards"][tname].float().numpy())
            )

        if incorrect_show:
            print(f"  -- ✗ Incorrect ({len(incorrect_all)} total, showing {len(incorrect_show)}) --")
            for rec in incorrect_show:
                s = sample_stats(rec["teacher_rewards"][tname].float().numpy())
                print(fmt_row(str(records.index(rec)), "✗", s))

        # ---- group summary rows (sample-level mean over ALL records) ----
        if correct_all_stats or incorrect_all_stats:
            print(SEP)
            if correct_all_stats:
                n = len(correct_all_stats)
                print(group_mean_row(f"[avg/{n}]", f"✓ correct ({n})", correct_all_stats))
            if incorrect_all_stats:
                n = len(incorrect_all_stats)
                print(group_mean_row(f"[avg/{n}]", f"✗ incorrect ({n})", incorrect_all_stats))

    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",    default="results/token_rewards_test.pkl")
    parser.add_argument("--sample_n", type=int, default=None,
                        help="Show N random samples per correct/incorrect group (default: all)")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()
    analyse(args.input, args.sample_n, args.seed)


if __name__ == "__main__":
    main()
