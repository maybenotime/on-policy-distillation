import pandas as pd, re, collections
tr = pd.read_parquet('/home/USER/opd/verl_data/ot3_math30k/train.parquet')
P = tr['prompt'].apply(lambda p: p[0]['content']).astype(str).tolist()
kw = {'latex $': r'\$', 'probability': r'probab', 'geometry': r'triangle|circle|angle|geometr|polygon', 'calculus': r'integral|derivative|\\int|limit', 'algebra/equation': r'equation|solve for|polynomial|factor', 'number theory': r'modulo|pmod|divisib|prime|remainder|gcd|lcm', 'combinatorics': r'how many ways|permutation|combination|arrange', 'sequence/series': r'sequence|series', 'word problem ($/people/time)': r'\$\d|dollar|hours|minutes|people|students|apples|price', 'complex/matrix/vector': r'complex number|matrix|vector', 'function': r'\bf\(x\)|function'}
for k, v in kw.items():
    r = re.compile(v, re.I); print(f'{k:30s} {sum(bool(r.search(p)) for p in P)/len(P)*100:5.1f}%')
print('first 2 words:', collections.Counter(' '.join(p.split()[:2]) for p in P).most_common(10))
ends = collections.Counter(re.sub(r'\s+', ' ', p.strip()[-60:]) for p in P)
print('tail phrases:', [(k[-45:], n) for k, n in ends.most_common(6)])
for i in range(8): print('---', P[i][:350])
