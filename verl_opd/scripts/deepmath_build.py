"""Intersect two graded files -> both-correct / both-wrong subsets (first K of the sampled order).
usage: deepmath_build.py --data sample.parquet --a grade_JustRL.jsonl --b grade_DS7B.jsonl --k 10000 --outdir DIR
"""
import argparse, json, os
import pandas as pd
ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
ap.add_argument("--k", type=int, default=10000); ap.add_argument("--outdir", required=True)
ap.add_argument("--name_a", default="JustRL"); ap.add_argument("--name_b", default="DS7B")
x = ap.parse_args()
df = pd.read_parquet(x.data)
def load(f): return {json.loads(l)["src_idx"]: json.loads(l) for l in open(f)}
A, B = load(x.a), load(x.b)
common = [i for i in df.src_idx if i in A and i in B]
df = df[df.src_idx.isin(common)].copy()
df[f"correct_{x.name_a}"] = df.src_idx.map(lambda i: A[i]["correct"])
df[f"correct_{x.name_b}"] = df.src_idx.map(lambda i: B[i]["correct"])
df[f"ntok_{x.name_a}"] = df.src_idx.map(lambda i: A[i]["ntok"]); df[f"ntok_{x.name_b}"] = df.src_idx.map(lambda i: B[i]["ntok"])
both_c = df[df[f"correct_{x.name_a}"] & df[f"correct_{x.name_b}"]]
both_w = df[~df[f"correct_{x.name_a}"] & ~df[f"correct_{x.name_b}"]]
if len(df) == 0:
    print("graded on both: 0   both correct: 0   both wrong: 0   (nothing graded yet)"); raise SystemExit
print(f"graded on both: {len(df)}  {x.name_a} acc {df[f'correct_{x.name_a}'].mean():.3f}  {x.name_b} acc {df[f'correct_{x.name_b}'].mean():.3f}")
print(f"both correct: {len(both_c)}   both wrong: {len(both_w)}   need {x.k} each")
os.makedirs(x.outdir, exist_ok=True)
df.to_parquet(f"{x.outdir}/all_graded.parquet", index=False)
for name, sub in [("both_correct", both_c), ("both_wrong", both_w)]:
    sub = sub.iloc[:x.k]
    sub.to_parquet(f"{x.outdir}/{name}_{len(sub)}.parquet", index=False)
    with open(f"{x.outdir}/{name}_{len(sub)}.jsonl", "w") as f:
        for r in sub.itertuples():
            f.write(json.dumps({"src_idx": int(r.src_idx), "question": r.question, "final_answer": r.final_answer,
                                "difficulty": float(r.difficulty), "topic": r.topic}) + "\n")
    print(f"  wrote {name}_{len(sub)}  difficulty mean {sub.difficulty.mean():.2f}")
