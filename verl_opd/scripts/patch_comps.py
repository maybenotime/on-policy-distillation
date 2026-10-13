import re
p='/home/sya225/opd/scripts/eval_v5.sh'; s=open(p).read()
m = re.search(r"    --comp aime/aime_2024_I.*?custom/beyondaime \\\n", s, re.S)
assert m, 'comp block not found'
new = '    --comp ${COMPS:-aime/aime_2024_I aime/aime_2024_II aime/aime_2025 hmmt/hmmt_feb_2025 hmmt/hmmt_feb_2026 custom/amc custom/math500 custom/beyondaime} \\\n'
s = s[:m.start()] + new + s[m.end():]; open(p,'w').write(s); print('patched eval_v5')
p='/home/sya225/opd/scripts/eval_par.sh'; s=open(p).read()
if 'COMPS=' not in s:
    s = s.replace('export MAX_TOKENS=16384 CONCURRENCY=100 GPU_UTIL=0.9 MATH_ONLY=1\n',
      'export MAX_TOKENS=16384 CONCURRENCY=100 GPU_UTIL=0.9 MATH_ONLY=1\nexport COMPS="aime/aime_2024_I aime/aime_2024_II aime/aime_2025 hmmt/hmmt_feb_2025 hmmt/hmmt_feb_2026 custom/amc custom/beyondaime"   # math500 dropped\n', 1)
    open(p,'w').write(s)
print('patched eval_par')
