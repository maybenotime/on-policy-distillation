#!/usr/bin/env python
"""Build a verl-format code OPD dataset from OpenThoughts3-1.2M.

Selects domain=code AND source=nvidia/OpenCodeReasoning (competitive-programming
problems), deliberately excluding stackexchange_codegolf — code golf optimises
for shortest source, a style we do not want to distil into the student.

Only the `human` turn is kept. The dataset's `gpt` answers are 40-60k-char
QwQ-32B thinking traces, which on-policy distillation never uses: the student
generates on-policy and the teacher supplies the target distribution.

Output schema matches verl's RL parquet contract (see
verl_repo/examples/data_preprocess/gsm8k.py).
"""
import argparse, json
from pathlib import Path

import datasets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=30000)
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--out_dir", default="/home/USER/opd/verl_data/ot3_code30k")
    ap.add_argument("--max_scan", type=int, default=400000,
                    help="stop scanning after this many rows (code region is the first ~250k)")
    args = ap.parse_args()

    want = args.n_train + args.n_test
    ds = datasets.load_dataset("open-thoughts/OpenThoughts3-1.2M", split="train", streaming=True)

    rows, scanned, seen_src = [], 0, {}
    for ex in ds:
        scanned += 1
        if scanned > args.max_scan:
            break
        dom, src = ex.get("domain"), ex.get("source")
        seen_src[src] = seen_src.get(src, 0) + 1
        if dom != "code" or src != "nvidia/OpenCodeReasoning":
            continue
        conv = ex.get("conversations") or []
        if not conv or conv[0].get("from") != "human":
            continue
        q = (conv[0].get("value") or "").strip()
        if not q:
            continue
        rows.append({"question": q, "difficulty": ex.get("difficulty"), "source": src})
        if len(rows) >= want:
            break
        if len(rows) % 5000 == 0:
            print(f"  collected {len(rows)}/{want}  (scanned {scanned})", flush=True)

    print(f"scanned={scanned}  collected={len(rows)}")
    print("source histogram (scanned):", json.dumps(
        dict(sorted(seen_src.items(), key=lambda x: -x[1])[:6]), indent=2))
    if len(rows) < want:
        print(f"WARNING: only {len(rows)} rows available, wanted {want}")

    def to_verl(r, idx, split):
        return {
            "data_source": "open-thoughts/OpenThoughts3-code",
            "prompt": [{"role": "user", "content": r["question"]}],
            "ability": "code",
            # No test cases in OT3 -> no rule-based grading. Paired with
            # zero_reward.py; distillation ignores this channel entirely.
            "reward_model": {"style": "rule", "ground_truth": ""},
            "extra_info": {"split": split, "index": idx,
                           "difficulty": r["difficulty"], "source": r["source"]},
        }

    test_rows = [to_verl(r, i, "test") for i, r in enumerate(rows[:args.n_test])]
    train_rows = [to_verl(r, i, "train") for i, r in enumerate(rows[args.n_test:])]

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    datasets.Dataset.from_list(train_rows).to_parquet(str(out / "train.parquet"))
    datasets.Dataset.from_list(test_rows).to_parquet(str(out / "test.parquet"))
    print(f"wrote {len(train_rows)} train / {len(test_rows)} test -> {out}")

    lens = [len(r["prompt"][0]["content"]) for r in train_rows]
    lens.sort()
    print(f"prompt chars: min={lens[0]} p50={lens[len(lens)//2]} p90={lens[int(len(lens)*.9)]} max={lens[-1]}")


if __name__ == "__main__":
    main()
