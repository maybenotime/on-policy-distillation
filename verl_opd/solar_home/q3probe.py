import re, sys
F="/localscratch/sya225/logs/train_q3i17_tpStu_shapeM500.log"
FIELDS=["actor/tp/H_A","actor/tp/H_target","actor/tp/T_mean","actor/tp/logT_mean","actor/tp/rho_s_B",
        "actor/tp/overlap16_sB","actor/tp/abs_k1_A","actor/entropy","actor/probe/sampled_in_topk",
        "response_length/mean"]
text=open(F,errors="ignore").read().splitlines()
# only the segment after the LAST "######## START" marker — the log is appended across restarts
# the chain script's START banner goes to the driver log, not here, so segment on the LAST step-1 metrics line
ones=[i for i,l in enumerate(text) if re.search(r"training/global_step:1(?!\d)",l) and "actor/tp/H_A" in l]
seg=text[ones[-1]:] if ones else text
print(f"[segment: {len(ones)} run(s) found by their step-1 line; reading from the last one]")
rows={}
for line in seg:
    m=re.search(r"training/global_step:(\d+)",line)
    if not m or "actor/tp/H_A" not in line: continue
    r={}
    for f in FIELDS:
        mm=re.search(re.escape(f)+r":(?:np\.float64\()?(-?[\d.eE+-]+)",line)
        if mm: r[f]=float(mm.group(1))
    rows[int(m.group(1))]=r
if not rows: sys.exit("no steps yet in the current run")
print(f"{'step':>5}{'H_A':>7}{'H_tgt':>7}{'T':>7}{'logT':>7}{'rho_sB':>8}{'ov_sB':>7}{'|k1|':>7}{'H_stu':>7}{'inTopK':>8}{'len':>7}")
for s in sorted(rows):
    r=rows[s]; g=lambda k: r.get(k,float('nan'))
    print(f"{s:>5}{g('actor/tp/H_A'):>7.3f}{g('actor/tp/H_target'):>7.3f}{g('actor/tp/T_mean'):>7.3f}"
          f"{g('actor/tp/logT_mean'):>7.3f}{g('actor/tp/rho_s_B'):>8.4f}{g('actor/tp/overlap16_sB'):>7.3f}"
          f"{g('actor/tp/abs_k1_A'):>7.3f}{g('actor/entropy'):>7.3f}{g('actor/probe/sampled_in_topk'):>8.3f}"
          f"{g('response_length/mean'):>7.0f}")
