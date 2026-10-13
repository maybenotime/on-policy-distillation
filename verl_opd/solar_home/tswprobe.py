import re
F="/localscratch/sya225/logs/train_q3i17_ts_mismatchW.log"
FIELDS=["select/w_max","select/eff_frac","select/abs_k1_weighted","select/abs_k1_unweighted",
        "actor/probe/rho","actor/probe/overlap16","actor/distillation/abs_loss","actor/entropy",
        "actor/grad_norm","response_length/mean"]
rows={}
for line in open(F,errors="ignore"):
    m=re.search(r"training/global_step:(\d+)",line)
    if not m or "select/eff_frac" not in line: continue
    r={}
    for f in FIELDS:
        mm=re.search(re.escape(f)+r":(?:np\.float64\()?(-?[\d.eE+-]+)",line)
        if mm: r[f]=float(mm.group(1))
    rows[int(m.group(1))]=r
if not rows: raise SystemExit("no steps yet")
print(f"{'step':>5}{'w_max':>7}{'eff':>6}{'k1_w':>7}{'k1_u':>7}{'ratio':>7}{'rho':>7}{'ov16':>7}{'H':>7}{'grad':>7}{'len':>6}")
for s in sorted(rows):
    if s%5 and s!=max(rows): continue
    r=rows[s]; g=lambda k:r.get(k,float('nan'))
    print(f"{s:>5}{g('select/w_max'):>7.2f}{g('select/eff_frac'):>6.2f}{g('select/abs_k1_weighted'):>7.3f}{g('select/abs_k1_unweighted'):>7.3f}"
          f"{g('select/abs_k1_weighted')/max(g('select/abs_k1_unweighted'),1e-9):>7.2f}{g('actor/probe/rho'):>7.3f}{g('actor/probe/overlap16'):>7.3f}"
          f"{g('actor/entropy'):>7.3f}{g('actor/grad_norm'):>7.2f}{g('response_length/mean'):>6.0f}")
