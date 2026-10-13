import sys, re
KEYS = ["step", "actor/tp/rho_s_B", "actor/tp/rho_s_A", "actor/tp/rho_tilde_B", "actor/probe/rho", "actor/probe/overlap16", "response_length/mean"]
for line in sys.stdin:
    if " step:" not in line:
        continue
    d = {}
    for k in KEYS:
        m = re.search(re.escape(k) + r":(?:np\.float64\()?([0-9.]+)", line)
        if m:
            d[k] = m.group(1)
    if len(d) == len(KEYS):
        print(" ".join(d[k] for k in KEYS))
