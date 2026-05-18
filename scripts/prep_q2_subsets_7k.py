"""Build two subsets from final_math_dataset_updated.jsonl:
1. Both Q2.5-7B-Inst and Q2.5-14B-Inst correct → random 7k → swift format
2. Both Q2.5-7B-Inst and Q2.5-14B-Inst wrong   → random 7k → swift format

Writes:
  data/prompts_math_swift_q2both_correct_7k.jsonl
  data/prompts_math_swift_q2both_wrong_7k.jsonl
  data/subset_q2both_correct_7k_idx.json   (original final_math_dataset row indices)
  data/subset_q2both_wrong_7k_idx.json
"""
import json
import random
from pathlib import Path

SRC = "/localhome/sya225/Projects/Self-Distillation/data/final_math_dataset_updated.jsonl"
OUT_DIR = Path("data")
SEED = 42
N_SAMPLES = 7000

correct_idx = []
wrong_idx = []
rows = []
with open(SRC) as f:
    for i, line in enumerate(f):
        d = json.loads(line)
        rows.append(d)
        t7 = d.get("Qwen2.5-7B-Instruct", {}).get("correctness_math_verify")
        t14 = d.get("Qwen2.5-14B-Instruct", {}).get("correctness_math_verify")
        if t7 is True and t14 is True:
            correct_idx.append(i)
        elif t7 is False and t14 is False:
            wrong_idx.append(i)

print(f"total rows: {len(rows)}")
print(f"both correct (7B & 14B): {len(correct_idx)}")
print(f"both wrong   (7B & 14B): {len(wrong_idx)}")

rng = random.Random(SEED)

def sample_and_write(idx_list, out_swift_path, out_idx_path, tag):
    sampled = sorted(rng.sample(idx_list, N_SAMPLES))
    with open(out_swift_path, "w") as f:
        for j in sampled:
            problem = rows[j]["problem"]
            f.write(json.dumps({"messages": [{"role": "user", "content": problem}]}) + "\n")
    with open(out_idx_path, "w") as f:
        json.dump(sampled, f)
    print(f"[{tag}] wrote {len(sampled)} → {out_swift_path}")

sample_and_write(
    correct_idx,
    OUT_DIR / "prompts_math_swift_q2both_correct_7k.jsonl",
    OUT_DIR / "subset_q2both_correct_7k_idx.json",
    "both_correct",
)
sample_and_write(
    wrong_idx,
    OUT_DIR / "prompts_math_swift_q2both_wrong_7k.jsonl",
    OUT_DIR / "subset_q2both_wrong_7k_idx.json",
    "both_wrong",
)
