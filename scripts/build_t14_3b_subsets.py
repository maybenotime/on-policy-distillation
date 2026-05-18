"""Cross-tab 14B-correct (from final_math_dataset) × 3B-correct/wrong (pass@4 ours).
Sample equal counts, write 2 swift-format jsonl files."""
import json
import random
from pathlib import Path

SRC = "/localhome/sya225/Projects/Self-Distillation/data/final_math_dataset_updated.jsonl"
P4 = "results/correctness/Qwen2.5-3B-Instruct_pass4_idx_to_correct.json"
SEED = 42

correct_by_idx = {int(k): v for k, v in json.load(open(P4)).items()}
print(f"3B pass@4 coverage: {len(correct_by_idx)} ({sum(correct_by_idx.values())} correct)")

rows = []
with open(SRC) as f:
    for line in f:
        rows.append(json.loads(line))
print(f"Dataset rows: {len(rows)}")

t14_correct = []
for i, d in enumerate(rows):
    if d.get("Qwen2.5-14B-Instruct", {}).get("correctness_math_verify") is True:
        t14_correct.append(i)
print(f"14B-correct rows: {len(t14_correct)}")

both_corr, t14corr_3bwrong = [], []
miss = 0
for i in t14_correct:
    if i not in correct_by_idx:
        miss += 1
        continue
    if correct_by_idx[i]:
        both_corr.append(i)
    else:
        t14corr_3bwrong.append(i)
print(f"  14B correct & 3B correct (pass@4): {len(both_corr)}")
print(f"  14B correct & 3B WRONG (all 4):    {len(t14corr_3bwrong)}")
print(f"  uncovered:                         {miss}")

N = min(len(both_corr), len(t14corr_3bwrong))
print(f"\nEqual sample N = {N}")
rng = random.Random(SEED)
samp_corr = sorted(rng.sample(both_corr, N))
samp_wrong = sorted(rng.sample(t14corr_3bwrong, N))

OUT_DIR = Path("data")
def write(idx_list, swift_p, idx_p, tag):
    with open(swift_p, "w") as f:
        for j in idx_list:
            f.write(json.dumps({"messages": [{"role": "user", "content": rows[j]["problem"]}]}) + "\n")
    with open(idx_p, "w") as f:
        json.dump(idx_list, f)
    print(f"[{tag}] {len(idx_list)} → {swift_p}")

write(samp_corr,
      OUT_DIR / f"prompts_math_swift_t14corr_3bcorr_{N}.jsonl",
      OUT_DIR / f"subset_t14corr_3bcorr_{N}_idx.json",
      "14Bcorr+3Bcorr")
write(samp_wrong,
      OUT_DIR / f"prompts_math_swift_t14corr_3bwrong_{N}.jsonl",
      OUT_DIR / f"subset_t14corr_3bwrong_{N}_idx.json",
      "14Bcorr+3Bwrong")
