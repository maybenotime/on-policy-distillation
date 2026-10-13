import re, sys, csv
FIELDS=["actor/probe/rho","actor/probe/overlap16","actor/probe/fork_overlap","actor/probe/dH",
        "actor/distillation/abs_loss","actor/entropy","actor/grad_norm","response_length/mean",
        "select/w_max","select/eff_frac","select/abs_k1_weighted","select/abs_k1_unweighted"]
log,out=sys.argv[1],sys.argv[2]; rows=[]
for line in open(log,errors="ignore"):
    m=re.search(r"training/global_step:(\d+)",line)
    if not m or "actor/probe/rho" not in line: continue
    r={"step":int(m.group(1))}
    for f in FIELDS:
        mm=re.search(re.escape(f)+r":(?:np\.float64\()?(-?[\d.eE+-]+)",line); r[f]=float(mm.group(1)) if mm else ""
    rows.append(r)
rows.sort(key=lambda r:r["step"])
with open(out,"w",newline="") as fh:
    w=csv.DictWriter(fh,["step"]+FIELDS); w.writeheader(); w.writerows(rows)
print(f"{out}: {len(rows)} steps, last={rows[-1]['step'] if rows else '-'}")
