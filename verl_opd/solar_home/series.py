"""Dump a verl training log's per-step metrics to CSV."""
import re, sys, csv
FIELDS = ["actor/probe/rho", "actor/tp/rho_s_B", "actor/tp/rho_s_A",
          "actor/probe/overlap16", "actor/tp/overlap16_sB", "actor/tp/overlap16_sA",
          "actor/distillation/abs_loss", "actor/tp/abs_k1_A",
          "actor/entropy", "actor/probe/student_entropy", "actor/probe/dH",
          "actor/probe/fork_overlap", "actor/grad_norm", "response_length/mean",
          "actor/probe/sampled_in_topk", "actor/tp/rho_tilde_B", "actor/tp/T_mean"]
log, out = sys.argv[1], sys.argv[2]
rows = []
for line in open(log, errors="ignore"):
    if "training/global_step" not in line:
        continue
    m = re.search(r"training/global_step:(\d+)", line)
    if not m:
        continue
    r = {"step": int(m.group(1))}
    for f in FIELDS:
        mm = re.search(re.escape(f) + r":(?:np\.float64\()?(-?[\d.eE+]+)", line)
        r[f] = float(mm.group(1)) if mm else ""
    rows.append(r)
rows.sort(key=lambda r: r["step"])
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, ["step"] + FIELDS); w.writeheader(); w.writerows(rows)
print(f"{out}: {len(rows)} steps, last={rows[-1]['step'] if rows else '-'}")
