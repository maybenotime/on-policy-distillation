"""Extract per-step metrics from a verl OPD training log into CSV."""
import re, sys, csv
log, out = sys.argv[1], sys.argv[2]
keys = ["actor/probe/overlap16","actor/probe/rho","actor/probe/fork_overlap","actor/probe/dH","actor/probe/student_entropy",
        "actor/probe/teacher_entropy_topk","actor/probe/sampled_in_topk","actor/probe/teacher_mass_topk","actor/distillation/abs_loss",
        "actor/distillation/loss","actor/entropy","actor/grad_norm","response_length/mean","response_length/clip_ratio","timing_s/step"]
pat = re.compile(r"(\S+):(?:np\.float64\()?([-0-9.e]+)\)?")
rows = {}
for line in open(log, errors="ignore"):
    m = re.search(r" step:(\d+) - ", line)
    if not m: continue
    step = int(m.group(1)); d = dict(pat.findall(line))
    rows[step] = [step] + [d.get(k, "") for k in keys]
with open(out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["step"] + [k.split("/")[-1] if k.startswith("actor/probe") else k.replace("/", "_") for k in keys])
    for s in sorted(rows): w.writerow(rows[s])
print("wrote", out, len(rows), "steps")
