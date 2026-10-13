import re, sys
path, comp = sys.argv[1], "beyondaime"
cur, rows = None, []
for ln in open(path, errors="ignore"):
    m = re.search(r"Done\. Printing final status from logs/status/\S*?/(\w+)_local/", ln)
    if m:
        cur = m.group(1)
        if cur == comp: rows = []
        continue
    if cur == comp:
        mm = re.match(r"\s*Problem\s+(\d+):\s*([OX]+)", ln)
        if mm: rows.append(mm.group(2).count("O") / len(mm.group(2)))
print(f"{comp}: n={len(rows)}  full={100*sum(rows)/len(rows):.1f}")
for k in (60, 70, 75, 80, 81, 85, 90):
    if k <= len(rows):
        print(f"   first {k:>3} problems: {100*sum(rows[:k])/k:.1f}")
