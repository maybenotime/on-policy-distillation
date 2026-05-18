"""Build prompts jsonl for compute_pass_at_k.py from final_math_dataset_updated.jsonl.

Output format expected by compute_pass_at_k.py: {prompt, answer} per line.
Preserves original 20k-row order so idx in output records maps back to dataset row.
"""
import json
from pathlib import Path

SRC = "/localhome/sya225/Projects/Self-Distillation/data/final_math_dataset_updated.jsonl"
OUT = Path("data/final_math_dataset_prompts.jsonl")

n = 0
with open(SRC) as fi, open(OUT, "w") as fo:
    for line in fi:
        d = json.loads(line)
        fo.write(json.dumps({"prompt": d["problem"], "answer": str(d["answer"])}) + "\n")
        n += 1
print(f"wrote {n} rows → {OUT}")
