"""
Static statistics for per-token rewards produced by token_reward_analysis.py.

For each teacher model reports (pooled over all samples × all tokens):
  - total tokens analysed
  - positive reward count / proportion
  - negative reward count / proportion
  - mean and variance of positive rewards
  - mean and variance of negative rewards

Usage:
    python reward_stats.py --input results/token_rewards_test.pkl
"""

import argparse
import pickle

import numpy as np


def stats_of(arr: np.ndarray) -> dict:
    """Mean and variance of a 1-D array; returns NaN when empty."""
    if arr.size == 0:
        return {"mean": float("nan"), "var": float("nan")}
    return {"mean": float(arr.mean()), "var": float(arr.var())}


def analyse(input_path: str) -> None:
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    metadata = data["metadata"]
    records  = data["records"]
    teacher_names = metadata["teacher_models"]

    print("=" * 62)
    print(f"Student : {metadata['student_model']}")
    print(f"Samples : {len(records)}")
    print(f"Temp    : {metadata['temperature']}   Max new tokens: {metadata['max_new_tokens']}")
    print("=" * 62)

    # Accumulate all rewards per teacher across every sample
    all_rewards: dict[str, list[np.ndarray]] = {t: [] for t in teacher_names}
    for rec in records:
        for tname in teacher_names:
            all_rewards[tname].append(rec["teacher_rewards"][tname].float().numpy())

    for tname in teacher_names:
        print(f"\nTeacher: {tname}")
        print(f"  {'Sample':<8} {'Correct':<8} {'Tokens':>6}  {'Pos%':>6}  {'Pos mean':>9}  {'Pos var':>8}  {'Neg%':>6}  {'Neg mean':>9}  {'Neg var':>8}")
        print(f"  {'-'*8} {'-'*8} {'-'*6}  {'-'*6}  {'-'*9}  {'-'*8}  {'-'*6}  {'-'*9}  {'-'*8}")
        for i, rec in enumerate(records):
            rewards = rec["teacher_rewards"][tname].float().numpy()
            pos = rewards[rewards > 0]
            neg = rewards[rewards < 0]
            total = rewards.size
            ps = stats_of(pos)
            ns = stats_of(neg)
            is_correct = rec.get("is_correct")
            correct_str = "✓" if is_correct is True else ("✗" if is_correct is False else "N/A")
            print(
                f"  {i:<8} {correct_str:<8} {total:>6}  {100*pos.size/total:>5.1f}%  "
                f"{ps['mean']:>+9.4f}  {ps['var']:>8.4f}  "
                f"{100*neg.size/total:>5.1f}%  "
                f"{ns['mean']:>+9.4f}  {ns['var']:>8.4f}"
            )

    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/token_rewards_test.pkl")
    args = parser.parse_args()
    analyse(args.input)


if __name__ == "__main__":
    main()
