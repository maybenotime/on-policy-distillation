"""Re-extract LCB code from saved raw completions under two rules.

official : lcb_runner's extract_code -- the LAST fenced block, whatever it is.
fixed    : the last fenced block that actually looks like an implementation
           (defines something, or reads stdin). A trailing usage demo such as
           `print(solve([1,2,3]))` has none of those markers.

Both files are written so the identical generations can be scored twice; the
difference is the size of the extraction artefact, with sampling noise removed.
"""
import json, re, sys, pathlib

O = pathlib.Path("/home/USER/opd/bench/official")
MARK = ("def ", "class ", "input(", "sys.stdin", "import ")

def blocks(txt):
    lines = txt.split("\n")
    idx = [i for i, l in enumerate(lines) if "```" in l]
    if len(idx) < 2:
        return []
    return ["\n".join(lines[idx[i] + 1:idx[i + 1]]) for i in range(0, len(idx) - 1, 2)]

def official(txt):
    lines = txt.split("\n")
    idx = [i for i, l in enumerate(lines) if "```" in l]
    if len(idx) < 2:
        return ""
    return "\n".join(lines[idx[-2] + 1:idx[-1]])

def fixed(txt):
    bs = blocks(txt)
    if not bs:
        return official(txt)
    cand = [b for b in bs if any(m in b for m in MARK)]
    return (cand[-1] if cand else bs[-1])

for tag in sys.argv[1:]:
    raw = json.loads((O / f"lcb_v6_{tag}_raw.json").read_text())
    for name, fn in (("", official), ("_fix", fixed)):
        recs = [{"question_id": r["question_id"],
                 "code_list": [fn(t) for t in r["raw_list"]]} for r in raw]
        (O / f"lcb_v6_{tag}{name}.json").write_text(json.dumps(recs))
    off = [official(t) for r in raw for t in r["raw_list"]]
    fix = [fixed(t)    for r in raw for t in r["raw_list"]]
    diff = sum(1 for a, b in zip(off, fix) if a != b)
    bad = sum(1 for c in off if not any(m in c for m in MARK))
    print(f"{tag:<20} n={len(off)}  官方抽取无实现标记={bad} ({100*bad/len(off):.1f}%)  "
          f"两种规则不同={diff} ({100*diff/len(off):.1f}%)")
