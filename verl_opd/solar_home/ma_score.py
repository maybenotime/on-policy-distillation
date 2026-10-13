"""mean-7 from a MathArena run log: parse the 'Done. Printing final status' blocks."""
import re, sys
COMPS = ["aime_2024_I", "aime_2024_II", "aime_2025", "amc", "beyondaime",
         "hmmt_feb_2025", "hmmt_feb_2026"]
for path in sys.argv[1:]:
    cur, acc = None, {}
    for ln in open(path, errors="ignore"):
        m = re.search(r"Done\. Printing final status from logs/status/\S*?/(\w+)_local/", ln)
        if m:
            cur = m.group(1); acc[cur] = []          # a re-run overwrites the earlier block
            continue
        if cur:
            mm = re.match(r"\s*Problem\s+(\d+):\s*([OX]+)", ln)
            if mm:
                g = mm.group(2); acc[cur].append(g.count("O") / len(g))
    vals, parts = [], []
    for c in COMPS:
        v = acc.get(c)
        if v:
            a = 100 * sum(v) / len(v); parts.append(f"{a:.1f}"); vals.append(a)
        else:
            parts.append("--")
    tag = path.split("/")[-1].replace("v5_", "").replace(".log", "")
    print(f"{tag:<42} " + " / ".join(parts) +
          (f"  -> mean-{len(vals)} {sum(vals)/len(vals):.1f}" if vals else "  (no data)"))
