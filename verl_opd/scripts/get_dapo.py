"""Fetch DAPO-Math-17k (train) and AIME-2024 (val placeholder) parquet in verl format."""
import os, pandas as pd
from huggingface_hub import hf_hub_download
out = "/home/USER/opd/verl_data/dapo_math17k"; os.makedirs(out, exist_ok=True)
tr = hf_hub_download("BytedTsinghua-SIA/DAPO-Math-17k", "data/dapo-math-17k.parquet", repo_type="dataset")
te = hf_hub_download("BytedTsinghua-SIA/AIME-2024", "data/aime-2024.parquet", repo_type="dataset")
for src, name in [(tr, "train"), (te, "test")]:
    df = pd.read_parquet(src); df.to_parquet(f"{out}/{name}.parquet")
    print(name, len(df), list(df.columns))
    r = df.iloc[0]; print("  prompt:", r["prompt"][0]["content"][:300].replace("\n", " "))
    print("  reward_model:", str(r["reward_model"])[:120], "| data_source:", r["data_source"], "| ability:", r.get("ability"))
P = tr_df = pd.read_parquet(f"{out}/train.parquet")["prompt"].apply(lambda p: p[0]["content"])
print("unique prompts:", P.nunique(), " chars p50/p90/max:", P.str.len().quantile([.5,.9,1]).round().tolist())
