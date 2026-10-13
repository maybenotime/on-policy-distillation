"""Grade generations with math_verify against final_answer.
usage: deepmath_grade.py --data sample.parquet --gen gen_A_shard0.jsonl [gen_A_shard1.jsonl ...] --out grade_A.jsonl
Output row: {src_idx, correct, finish_reason, ntok, pred}
"""
import argparse, json, os, sys
import pandas as pd
from math_verify import parse, verify
from math_verify.errors import TimeoutException

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--gen", nargs="+", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
gold = dict(zip(pd.read_parquet(a.data).src_idx, pd.read_parquet(a.data).final_answer))

def grade(ans, resp):
    try:
        g = parse("\\boxed{" + str(ans) + "}")
        # keep only the last \boxed{...} of the response so early scratch work can't match
        k = resp.rfind("\\boxed")
        tail = resp[k:] if k >= 0 else ""
        if not tail: return False, ""
        p = parse(tail)
        return bool(verify(g, p)), (str(p[0]) if p else "")
    except TimeoutException:
        return False, "TIMEOUT"
    except Exception as e:
        return False, f"ERR {type(e).__name__}"

n = nc = 0
with open(a.out, "w") as fo:
    for gf in a.gen:
        if not os.path.exists(gf):
            print(f"[grade] missing (not started yet): {gf}"); continue
        for line in open(gf):
            try: r = json.loads(line)
            except Exception: continue
            ok, pred = (False, "") if r["finish_reason"] == "length" else grade(gold[r["src_idx"]], r["response"])
            fo.write(json.dumps({"src_idx": r["src_idx"], "correct": ok, "finish_reason": r["finish_reason"],
                                 "ntok": r["ntok"], "pred": pred[:80]}) + "\n")
            n += 1; nc += ok
print(f"[grade] {a.out}: {nc}/{n} correct = {100*nc/max(n,1):.1f}%")
