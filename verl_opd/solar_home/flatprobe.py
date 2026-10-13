import re
F="/localscratch/sya225/logs/train_ds15b_tpStu_flat.log"
FIELDS=["actor/tp/H_A","actor/tp/H_target","actor/tp/H_tilde","actor/tp/T_mean","actor/tp/logT_mean",
        "actor/tp/unmet_frac","actor/tp/rho_s_A","actor/tp/rho_s_B","actor/tp/overlap16_sA",
        "actor/tp/overlap16_sB","actor/tp/abs_k1_A","actor/entropy","actor/probe/sampled_in_topk",
        "response_length/mean"]
rows={}
for line in open(F,errors="ignore"):
    m=re.search(r"training/global_step:(\d+)",line)
    if not m or "actor/tp/H_A" not in line: continue
    r={}
    for f in FIELDS:
        mm=re.search(re.escape(f)+r":(?:np\.float64\()?(-?[\d.eE+-]+)",line)
        if mm: r[f]=float(mm.group(1))
    rows[int(m.group(1))]=r
print(f"{'step':>5}{'H_A':>7}{'H_tgt':>7}{'H_til':>7}{'logT':>7}{'unmet':>7}{'rho_sA':>8}{'rho_sB':>8}{'ov_sB':>7}{'|k1|':>7}{'H_stu':>7}{'len':>7}")
for s in sorted(rows):
    if s % 5 and s != max(rows): continue
    r=rows[s]; g=lambda k: r.get(k,float('nan'))
    print(f"{s:>5}{g('actor/tp/H_A'):>7.3f}{g('actor/tp/H_target'):>7.3f}{g('actor/tp/H_tilde'):>7.3f}"
          f"{g('actor/tp/logT_mean'):>7.3f}{g('actor/tp/unmet_frac'):>7.3f}{g('actor/tp/rho_s_A'):>8.4f}"
          f"{g('actor/tp/rho_s_B'):>8.4f}{g('actor/tp/overlap16_sB'):>7.3f}{g('actor/tp/abs_k1_A'):>7.3f}"
          f"{g('actor/entropy'):>7.3f}{g('response_length/mean'):>7.0f}")
