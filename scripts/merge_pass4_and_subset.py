"""Merge 20 pass@4 shards → cross-tab against final_math_dataset_updated.jsonl
Q2.5-14B correctness → split 14B-correct rows into {3B-correct, 3B-incorrect}
→ sample equal count from each, write 2 swift-format jsonl files.
"""
import json
import random
import glob
from pathlib import Path

REPO = "/scratch/sya225/on-policy-distillation"
SRC_DATASET = "/localhome/sya225/Projects/Self-Distillation/data/final_math_dataset_updated.jsonl"
SEED = 42

# 1. Merge shards: produce idx -> is_correct_any
shard_files = sorted(glob.glob("results/correctness/Qwen2.5-3B-Instruct_pass4_finalmath_shard*of20.json"))
print(f"Found {len(shard_files)} shards")
correct_by_idx = {}
for sf in shard_files:
    d = json.load(open(sf))
    for r in d["records"]:
        correct_by_idx[r["idx"]] = r["is_correct_any"]
print(f"Merged: {len(correct_by_idx)} idx covered, {sum(correct_by_idx.values())} correct ({sum(correct_by_idx.values())/len(correct_by_idx)*100:.2f}%)")

# 2. Cross-tab with final_math_dataset
rows = []
with open(SRC_DATASET) as f:
    for line in f:
        rows.append(json.loads(line))
print(f"Dataset rows: {len(rows)}")

t14_correct_idx = []
for i, d in enumerate(rows):
    t14 = d.get("Qwen2.5-14B-Instruct", {}).get("correctness_math_verify")
    if t14 is True:
        t14_correct_idx.append(i)
print(f"14B-correct rows: {len(t14_correct_idx)}")

t14_3b_correct = []
t14_3b_wrong = []
missing = 0
for i in t14_correct_idx:
    if i not in correct_by_idx:
        missing += 1
        continue
    if correct_by_idx[i]:
        t14_3b_correct.append(i)
    else:
        t14_3b_wrong.append(i)
print(f"  14B correct + 3B correct (pass@4): {len(t14_3b_correct)}")
print(f"  14B correct + 3B WRONG (all 4):    {len(t14_3b_wrong)}")
print(f"  missing 3B coverage:               {missing}")

# 3. Equal sample
N = min(len(t14_3b_correct), len(t14_3b_wrong))
print(f"\nSample size each: {N}")
rng = random.Random(SEED)
sampled_correct = sorted(rng.sample(t14_3b_correct, N))
sampled_wrong = sorted(rng.sample(t14_3b_wrong, N))

OUT_DIR = Path("data")

def write_swift(idx_list, path, idx_path, tag):
    with open(path, "w") as f:
        for j in idx_list:
            f.write(json.dumps({"messages": [{"role": "user", "content": rows[j]["problem"]}]}) + "\n")
    with open(idx_path, "w") as f:
        json.dump(idx_list, f)
    print(f"[{tag}] wrote {len(idx_list)} → {path}")

write_swift(sampled_correct,
            OUT_DIR / f"prompts_math_swift_t14_3bcorrect_{N}.jsonl",
            OUT_DIR / f"subset_t14_3bcorrect_{N}_idx.json",
            "14Bcorr_3Bcorr")
write_swift(sampled_wrong,
            OUT_DIR / f"prompts_math_swift_t14_3bwrong_{N}.jsonl",
            OUT_DIR / f"subset_t14_3bwrong_{N}_idx.json",
            "14Bcorr_3Bwrong")

# 4. Also dump full pass4 merged result file like the original Qwen2.5-3B-Instruct.json
merged_records = []
for idx in sorted(correct_by_idx.keys()):
    merged_records.append({"idx": idx, "is_correct_any": correct_by_idx[idx]})
out_full = "results/correctness/Qwen2.5-3B-Instruct_pass4_finalmath_merged.json"
with open(out_full, "w") as f:
    json.dump({
        "metadata": {
            "model": "Qwen/Qwen2.5-3B-Instruct",
            "n_prompts": len(merged_records),
            "n_rollouts": 4,
            "temperature": 1.0,
            "max_tokens": 8192,
            "data_file": "final_math_dataset_prompts.jsonl",
            "seed": 42,
            "merged_from_shards": len(shard_files),
        },
        "accuracy_pass_at_4": sum(correct_by_idx.values()) / len(correct_by_idx),
        "n_correct_pass_at_4": sum(correct_by_idx.values()),
        "records": merged_records,
    }, f, indent=1)
print(f"\n[merged] wrote {out_full}")
