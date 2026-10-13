import re
RUNS={"flat (DS, target=JustRL mean H)":"/localscratch/sya225/logs/train_ds15b_tpStu_flat.log"}
import os
for name,F in RUNS.items():
    if not os.path.exists(F): print("missing",F); continue
    acc={}
    for line in open(F,errors="ignore"):
        m=re.search(r"training/global_step:(\d+)",line)
        if not m or "actor/tp/H_A" not in line: continue
        def g(f):
            mm=re.search(re.escape(f)+r":(?:np\.float64\()?(-?[\d.eE+-]+)",line)
            return float(mm.group(1)) if mm else float("nan")
        acc[int(m.group(1))]=(g("actor/tp/H_target"),g("actor/tp/H_tilde"),g("actor/tp/unmet_frac"),
                              g("actor/tp/T_at_bound_frac"))
    ks=sorted(acc)[-30:]
    tg=sum(acc[k][0] for k in ks)/len(ks); ti=sum(acc[k][1] for k in ks)/len(ks)
    un=sum(acc[k][2] for k in ks)/len(ks); bd=sum(acc[k][3] for k in ks)/len(ks)
    print(f"{name}\n  last-30-step mean: target {tg:.4f}  achieved {ti:.4f}  ratio {ti/tg:.3f}"
          f"  unmet {100*un:.1f}%  at-bound {100*bd:.2f}%")
