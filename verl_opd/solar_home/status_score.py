"""Score a MathArena model from its status files (logs/status/<comp>_local/<cfg>.txt), which survive log overwrites.
usage: status_score.py NAME cfg1[:comp,comp] cfg2[:comp,...]   (default comps: all 7).  Reports incomplete comps."""
import re, sys, os
S = "/home/sya225/opd/matharena/logs/status"
ALL = ["aime/aime_2024_I","aime/aime_2024_II","aime/aime_2025","custom/amc","custom/beyondaime","hmmt/hmmt_feb_2025","hmmt/hmmt_feb_2026"]
name = sys.argv[1]; acc = {}; incomplete = []
for arg in sys.argv[2:]:
    cfg, _, cs = arg.partition(":"); comps = cs.split(",") if cs else ALL
    for c in comps:
        p = f"{S}/{c}_local/{cfg}.txt"
        if not os.path.exists(p): continue
        vals = []; left = 0
        for ln in open(p):
            m = re.match(r"\s*Problem\s+\d+:\s*([OX]+)\s*\((\d+) left\)", ln)
            if m: vals.append(m.group(1).count("O")/len(m.group(1))); left += int(m.group(2))
        if not vals: continue
        acc[c] = 100*sum(vals)/len(vals)
        if left: incomplete.append(f"{c}:{left} left")
row = [f"{acc[c]:.1f}" if c in acc else "--" for c in ALL]
vals = [acc[c] for c in ALL if c in acc]
print(f"{name:<28} " + " / ".join(row) + f"  -> mean-{len(vals)} {sum(vals)/len(vals):.1f}" + (f"   INCOMPLETE: {incomplete}" if incomplete else ""))
