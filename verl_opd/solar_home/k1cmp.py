import re, sys, glob
import numpy as np
FIELDS = ["actor/distillation/abs_loss", "actor/probe/overlap16", "actor/probe/sampled_in_topk",
          "actor/probe/rho", "actor/probe/dH", "actor/probe/student_entropy",
          "actor/probe/teacher_entropy_topk", "response_length/mean", "actor/grad_norm"]
def parse(path, nstep=10):
    rows = {f: [] for f in FIELDS}
    for line in open(path, errors="ignore"):
        if "step:" not in line or "actor/distillation/abs_loss" not in line:
            continue
        m = re.search(r"step:(\d+)", line)
        if not m or int(m.group(1)) > nstep:
            continue
        for f in FIELDS:
            mm = re.search(re.escape(f) + r":(?:np\.float64\()?(-?[\d.eE+]+)", line)
            if mm:
                try: rows[f].append(float(mm.group(1)))
                except ValueError: pass
    return rows
for p in sys.argv[1:]:
    r = parse(p)
    tag = p.split("train_")[-1].replace(".log", "")
    n = len(r["actor/distillation/abs_loss"])
    out = [f"{tag:<34} steps={n}"]
    for f in FIELDS:
        v = r[f]
        out.append(f"  {f.split('/')[-1]:<22}{np.mean(v):.4f}" if v else f"  {f.split('/')[-1]:<22}--")
    print("\n".join(out)); print()
