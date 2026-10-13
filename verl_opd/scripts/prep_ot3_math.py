#!/usr/bin/env python
"""Build a verl-format MATH OPD dataset from OpenThoughts3-1.2M (companion of prep_ocr.py).

Selects domain=math. Unlike the code set, the math region mixes several sources,
so we collect a large pool of math prompts (up to --pool) and then draw a seeded
random sample, instead of taking the first N rows. Only the `human` turn is kept.
Schema identical to ot3_code30k (empty ground_truth; OPD ignores the reward).
"""
import argparse, json, random
from pathlib import Path

import datasets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=30000)
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--pool", type=int, default=200000, help="math rows to collect before sampling")
    ap.add_argument("--max_scan", type=int, default=1300000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="/home/USER/opd/verl_data/ot3_math30k")
    args = ap.parse_args()

    ds = datasets.load_dataset("open-thoughts/OpenThoughts3-1.2M", split="train", streaming=True)
    rows, scanned, seen_dom, seen_src = [], 0, {}, {}
    for ex in ds:
        scanned += 1
        if scanned > args.max_scan:
            break
        dom, src = ex.get("domain"), ex.get("source")
        seen_dom[dom] = seen_dom.get(dom, 0) + 1
        if dom != "math":
            continue
        seen_src[src] = seen_src.get(src, 0) + 1
        conv = ex.get("conversations") or []
        if not conv or conv[0].get("from") != "human":
            continue
        q = (conv[0].get("value") or "").strip()
        if not q:
            continue
        rows.append({"question": q, "difficulty": ex.get("difficulty"), "source": src})
        if len(rows) >= args.pool:
            break
        if len(rows) % 20000 == 0:
            print(f"  collected {len(rows)}/{args.pool}  (scanned {scanned})", flush=True)

    print(f"scanned={scanned}  math pool={len(rows)}")
    print("domain histogram (scanned):", json.dumps(seen_dom))
    print("math source histogram (pool):", json.dumps(dict(sorted(seen_src.items(), key=lambda x: -x[1]))))

    # dedupe on question text, then seeded sample
    seen, uniq = set(), []
    for r in rows:
        if r["question"] in seen:
            continue
        seen.add(r["question"]); uniq.append(r)
    print(f"unique questions: {len(uniq)}")
    want = args.n_train + args.n_test
    random.Random(args.seed).shuffle(uniq)
    picked = uniq[:want]
    if len(picked) < want:
        print(f"WARNING: only {len(picked)} rows available, wanted {want}")

    def to_verl(r, idx, split):
        return {
            "data_source": "open-thoughts/OpenThoughts3-math",
            "prompt": [{"role": "user", "content": r["question"]}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": ""},
            "extra_info": {"split": split, "index": idx,
                           "difficulty": r["difficulty"], "source": r["source"]},
        }

    test_rows = [to_verl(r, i, "test") for i, r in enumerate(picked[:args.n_test])]
    train_rows = [to_verl(r, i, "train") for i, r in enumerate(picked[args.n_test:])]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    datasets.Dataset.from_list(train_rows).to_parquet(str(out / "train.parquet"))
    datasets.Dataset.from_list(test_rows).to_parquet(str(out / "test.parquet"))
    print(f"wrote {len(train_rows)} train / {len(test_rows)} test -> {out}")
    hist = {}
    for r in train_rows:
        s = r["extra_info"]["source"]; hist[s] = hist.get(s, 0) + 1
    print("train source histogram:", json.dumps(dict(sorted(hist.items(), key=lambda x: -x[1]))))
    lens = sorted(len(r["prompt"][0]["content"]) for r in train_rows)
    print(f"prompt chars: min={lens[0]} p50={lens[len(lens)//2]} p90={lens[int(len(lens)*.9)]} max={lens[-1]}")


if __name__ == "__main__":
    main()
