"""Prediction task (outline 4.2): pre-OPD probes vs realised OPD gain, Spearman (+bootstrap CI) and teacher-selection regret.
Pre-probes: base-student rollouts (1000 DAPO prompts, 4k, T0.7), same protocol for both students; Qwen3-4B row is the
online step-1 value (no offline probe).  Gain = mean-6 at step 200 of the DAPO-17k run (base: DS 25.5, Qwen3 13.0).
Teacher accuracy = teacher's own six-benchmark mean under the student's eval protocol (log 2.6)."""
import numpy as np
rng = np.random.default_rng(0)
# student, teacher, rho, FO, overlap16, dH, teacher_acc, teacher_params(B), gain
P = [
 ("DS","JustRL-1.5B",        .853,73.9,71.3,.197,40.1,1.5, 8.1),
 ("DS","Nemotron-1.5B",      .812,69.9,69.5,.412,38.0,1.5,11.0),
 ("DS","R1-Distill-7B",      .930,82.6,73.3,.086,43.0,7.0, 0.2),
 ("DS","Skywork-OR1-7B",     .864,75.5,69.6,.079,51.6,7.0, 3.5),
 ("DS","Light-R1-7B",        .866,75.4,68.1,.027,45.4,7.0, 0.7),
 ("Q3","Qwen3-4B-RL-Math",   .771,75.8,70.3,.017,40.3,4.0,15.3),
 ("Q3","Qwen3-4B-Code-1200", .837,79.4,72.0,.100,32.8,4.0, 8.8),
 ("Q3","Qwen3-8B-GRPO-MATH", .865,81.6,72.7,.051,31.6,8.0, 5.1),
 ("Q3","Qwen3-8B",           .889,82.8,73.2,.083,26.1,8.0, 2.3),
 ("Q3","Qwen3-4B",           .895,82.8,74.5,.046,22.7,4.0, 0.5),
]
name = ["pre-ρ (lower=better)","ForkOverlap (lower)","overlap@16 (lower)","ΔH (higher)","teacher acc (higher)","teacher size (higher)"]
sign = [-1,-1,-1,+1,+1,+1]   # direction in which the predictor says "better teacher"
cols = [2,3,4,5,6,7]
gain = np.array([p[8] for p in P]); stu = np.array([p[0] for p in P])
def spear(x,y):
    rx=np.argsort(np.argsort(x)); ry=np.argsort(np.argsort(y)); return np.corrcoef(rx,ry)[0,1]
def spear_ci(x,y,B=5000):
    n=len(x); v=[]
    for _ in range(B):
        i=rng.integers(0,n,n)
        if len(set(x[i]))<3 or len(set(y[i]))<3: continue
        v.append(spear(x[i],y[i]))
    return np.percentile(v,[2.5,97.5])
print("| predictor | Spearman (all 11) | 95% CI | DS (6) | Qwen3 (5) | regret DS | regret Qwen3 | mean regret | top-1 hits |")
print("|---|---|---|---|---|---|---|---|---|")
for nm,sg,c in zip(name,sign,cols):
    x=np.array([p[c] for p in P])*sg
    s_all=spear(x,gain); lo,hi=spear_ci(x,gain)
    out=[nm,f"{s_all:+.2f}",f"[{lo:+.2f}, {hi:+.2f}]"]; regs=[]; hits=0
    for st in ("DS","Q3"):
        m=stu==st; xs=x[m]; gs=gain[m]
        out.append(f"{spear(xs,gs):+.2f}")
        pick=np.argmax(xs); reg=gs.max()-gs[pick]; regs.append(reg); hits+=int(reg==0)
    out += [f"{regs[0]:.1f}",f"{regs[1]:.1f}",f"{np.mean(regs):.1f}",f"{hits}/2"]
    print("| "+" | ".join(out)+" |")
# direction: does the predictor separate 'helps' (gain>=3) from 'does not' (gain<3) ? AUC over the pooled 11
print()
lab=(gain>=3).astype(int)
for nm,sg,c in zip(name,sign,cols):
    x=np.array([p[c] for p in P])*sg
    pos=x[lab==1]; neg=x[lab==0]; auc=np.mean([(a>b)+0.5*(a==b) for a in pos for b in neg])
    print(f"{nm:24s} AUC(helps vs not, pooled) = {auc:.2f}")
print("\nper-pairing table:")
print("| student | teacher | pre-ρ | FO | ov@16 | ΔH | teacher acc | Δ mean-6 (s200) |"); print("|---|---|---|---|---|---|---|---|")
for p in P: print(f"| {p[0]} | {p[1]} | {p[2]:.3f} | {p[3]:.1f} | {p[4]:.1f} | {p[5]:.3f} | {p[6]:.1f} | {p[8]:+.1f} |")
