"""DeepMath subset (jsonl with question/final_answer) -> verl parquet in the DAPO-17k format.
usage: deepmath_to_verl.py --src both_wrong_7333.jsonl --n 7000 --seed 0 --out verl_data/deepmath_p1w
"""
import argparse, json, os, shutil, numpy as np, pandas as pd
ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True); ap.add_argument("--n", type=int, default=7000); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True); ap.add_argument("--test_from", default="/home/sya225/opd/verl_data/dapo_math17k/test.parquet")
a = ap.parse_args()
PREFIX = ("Solve the following math problem step by step. The last line of your response should be of the form "
          "Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n")
SUFFIX = "\n\nRemember to put your answer on its own line after \"Answer:\"."
rows = [json.loads(l) for l in open(a.src)]
rng = np.random.default_rng(a.seed)
if a.n and a.n < len(rows): rows = [rows[i] for i in sorted(rng.choice(len(rows), a.n, replace=False))]
recs = [{"data_source": "math_deepmath", "prompt": [{"role": "user", "content": PREFIX + r["question"] + SUFFIX}],
         "ability": "MATH", "reward_model": {"ground_truth": str(r["final_answer"]), "style": "rule-lighteval/MATH_v2"},
         "extra_info": {"split": "train", "index": i, "src_idx": int(r["src_idx"]), "difficulty": float(r["difficulty"])}}
        for i, r in enumerate(rows)]
os.makedirs(a.out, exist_ok=True)
pd.DataFrame(recs).to_parquet(f"{a.out}/train.parquet", index=False); shutil.copy(a.test_from, f"{a.out}/test.parquet")
d = np.array([r["difficulty"] for r in rows])
print(f"wrote {a.out}/train.parquet: {len(recs)} rows from {a.src}; difficulty mean {d.mean():.2f}; test = AIME24 copy")
