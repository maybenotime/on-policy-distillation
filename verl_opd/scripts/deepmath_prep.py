"""Sample N problems from zwhe99/DeepMath-103K (fixed seed) -> parquet used by gen/grade/build.
usage: deepmath_prep.py --n 45000 --seed 0 --out /home/sya225/opd/verl_data/deepmath/sample.parquet
"""
import argparse, numpy as np, pandas as pd
from datasets import load_dataset
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=45000); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
ap.add_argument("--order", choices=["random", "tails"], default="random",
                help="tails: hard stream (difficulty>=6.5, desc) interleaved 3:1 with easy stream (difficulty<6.5, asc)")
ap.add_argument("--ratio", type=int, default=3)
a = ap.parse_args()
d = load_dataset("zwhe99/DeepMath-103K", split="train")
df = d.remove_columns([c for c in d.column_names if c.startswith("r1_solution")]).to_pandas()
df["src_idx"] = np.arange(len(df))
rng = np.random.default_rng(a.seed)
order = rng.permutation(len(df))            # full permutation: extending N later keeps the prefix
sub = df.iloc[order[:a.n]].reset_index(drop=True)
if a.order == "tails":
    sub["perm"] = np.arange(len(sub))
    hard = sub[sub.difficulty >= 6.5].sort_values(["difficulty", "perm"], ascending=[False, True])
    easy = sub[sub.difficulty < 6.5].sort_values(["difficulty", "perm"], ascending=[True, True])
    rows, hi, ei = [], 0, 0
    while hi < len(hard) or ei < len(easy):
        for _ in range(a.ratio):
            if hi < len(hard): rows.append(hard.iloc[hi]); hi += 1
        if ei < len(easy): rows.append(easy.iloc[ei]); ei += 1
    sub = pd.DataFrame(rows).drop(columns=["perm"]).reset_index(drop=True)
    print(f"tails order: hard stream {len(hard)} (diff>=6.5 desc), easy stream {len(easy)} (asc), ratio {a.ratio}:1")
    print("first 12 difficulties:", list(sub.difficulty[:12]))
sub.to_parquet(a.out, index=False)
print(f"wrote {a.out}: {len(sub)} rows; difficulty mean {sub.difficulty.mean():.2f}")
