import pandas as pd
out = "/home/USER/opd/verl_data/dapo_math17k"
for name in ["train", "test"]:
    df = pd.read_parquet(f"{out}/{name}.parquet")
    key = df["prompt"].apply(lambda p: p[0]["content"])
    df = df[~key.duplicated()].reset_index(drop=True)
    df["extra_info"] = [{"split": name, "index": i} for i in range(len(df))]
    df.to_parquet(f"{out}/{name}.parquet"); print(name, len(df), "rows after dedupe")
