import re
F="/localscratch/sya225/logs/train_ds15b_tpStu_flat.log"
rows={}
for line in open(F,errors="ignore"):
    m=re.search(r"training/global_step:(\d+)",line)
    if not m or "actor/tp/H_A" not in line: continue
    def g(f):
        mm=re.search(re.escape(f)+r":(?:np\.float64\()?(-?[\d.eE+-]+)",line)
        return float(mm.group(1)) if mm else float("nan")
    rows[int(m.group(1))]=(g("actor/entropy"),g("actor/tp/H_target"),g("actor/tp/H_A"),g("actor/tp/H_tilde"))
print(f"{'step':>5}{'H_student':>11}{'H_JustRL':>10}{'gap':>8}{'H_baseA':>9}{'H_target_achieved':>18}")
for s in sorted(rows):
    if s % 10 and s != max(rows): continue
    hs,ht,ha,hd = rows[s]
    print(f"{s:>5}{hs:>11.3f}{ht:>10.3f}{ht-hs:>+8.3f}{ha:>9.3f}{hd:>18.3f}")
