"""Like log2csv.py but tolerant: handles np.float64(...) values, '(TaskRunner pid=..)' prefixes, and keeps the LAST occurrence of each step
(so a restarted run overrides the crashed one).  usage: log2csv2.py <log> <out.csv>"""
import re, sys, csv
log, out = sys.argv[1], sys.argv[2]
keys = {"actor/probe/overlap16":"overlap16","actor/probe/rho":"rho","actor/probe/fork_overlap":"fork_overlap","actor/probe/dH":"dH",
        "actor/probe/student_entropy":"student_entropy","actor/probe/teacher_entropy_topk":"teacher_entropy_topk","actor/probe/sampled_in_topk":"sampled_in_topk",
        "actor/probe/teacher_mass_topk":"teacher_mass_topk","actor/distillation/abs_loss":"actor_distillation_abs_loss","actor/distillation/loss":"actor_distillation_loss",
        "actor/entropy":"actor_entropy","actor/grad_norm":"actor_grad_norm","response_length/mean":"response_length_mean","response_length/clip_ratio":"response_length_clip_ratio","timing_s/step":"timing_s_step"}
rows = {}
pat = re.compile(r"(?:^|\s)step:(\d+)\s")
num = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
for line in open(log, errors="replace"):
    m = pat.search(line)
    if not m or " - " not in line: continue
    st = int(m.group(1)); d = {}
    for k, name in keys.items():
        mm = re.search(re.escape(k) + r":(?:np\.float64\()?([-+0-9.eE]+)", line)
        if mm: d[name] = mm.group(1)
    if d: rows[st] = d
cols = ["step"] + list(keys.values())
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for st in sorted(rows): w.writerow({"step": st, **rows[st]})
print("wrote", out, len(rows), "steps")
