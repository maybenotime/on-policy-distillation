"""mean-7 across several MathArena logs that each covered a subset of the competitions."""
import re, sys
COMPS = ["aime_2024_I", "aime_2024_II", "aime_2025", "amc", "beyondaime",
         "hmmt_feb_2025", "hmmt_feb_2026"]
acc = {}
for path in sys.argv[2:]:
    cur = None
    for ln in open(path, errors="ignore"):
        m = re.search(r"Done\. Printing final status from logs/status/\S*?/(\w+)_local/", ln)
        if m:
            cur = m.group(1); acc[cur] = []
            continue
        if cur:
            mm = re.match(r"\s*Problem\s+(\d+):\s*([OX]+)", ln)
            if mm:
                g = mm.group(2); acc[cur].append(g.count("O") / len(g))
vals, parts = [], []
for c in COMPS:
    v = [x for x in acc.get(c, []) if True]
    if v:
        a = 100 * sum(v) / len(v); parts.append(f"{a:.1f}"); vals.append(a)
    else:
        parts.append("--")
print(f"{sys.argv[1]:<34} " + " / ".join(parts) +
      (f"  -> mean-{len(vals)} {sum(vals)/len(vals):.1f}" if vals else "  (no data)"))
