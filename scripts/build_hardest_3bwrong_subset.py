"""Within the existing 7k "both 7B+14B wrong" subset, find rows where 3B is ALSO wrong
(pass@4=False). Sample 2784 of them to match the t14corr_3bwrong subset size.
"""
import json
import random
from pathlib import Path

SRC = "/localhome/sya225/Projects/Self-Distillation/data/final_math_dataset_updated.jsonl"
SEED = 42
TARGET_N = 2784

# Original 7000-row idx list (14B+7B both wrong, seed=42 random subset of 10281)
bothwrong_idx = json.load(open("data/subset_q2both_wrong_7k_idx.json"))
# 3B pass@4 results
correct_by_idx = {int(k): v for k, v in json.load(open(
    "results/correctness/Qwen2.5-3B-Instruct_pass4_idx_to_correct.json")).items()}

# Filter: 14B wrong (already) + 7B wrong (already) + 3B wrong (pass@4=False)
all_three_wrong = [i for i in bothwrong_idx if correct_by_idx.get(i) is False]
also_3b_correct = [i for i in bothwrong_idx if correct_by_idx.get(i) is True]
print(f"7B+14B-both-wrong-7k: {len(bothwrong_idx)}")
print(f"  also 3B-wrong (3 ways wrong): {len(all_three_wrong)}")
print(f"  but 3B correct anyway:        {len(also_3b_correct)}")

rng = random.Random(SEED)
sampled = sorted(rng.sample(all_three_wrong, TARGET_N))

# Load problems
rows = []
with open(SRC) as f:
    for line in f:
        rows.append(json.loads(line))

OUT_DIR = Path("data")
swift = OUT_DIR / f"prompts_math_swift_all3wrong_{TARGET_N}.jsonl"
idxf = OUT_DIR / f"subset_all3wrong_{TARGET_N}_idx.json"
with open(swift, "w") as f:
    for j in sampled:
        f.write(json.dumps({"messages": [{"role": "user", "content": rows[j]["problem"]}]}) + "\n")
with open(idxf, "w") as f:
    json.dump(sampled, f)
print(f"\n[all3wrong] {len(sampled)} → {swift}")
