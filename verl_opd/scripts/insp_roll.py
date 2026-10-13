import json, sys, collections
from transformers import AutoTokenizer
rows=[json.loads(l) for l in open(sys.argv[1])]; tok=AutoTokenizer.from_pretrained(sys.argv[2])
L=sorted(len(r['resp_ids']) for r in rows)
print('n',len(rows),'len p10/p50/p90', L[len(L)//10], L[len(L)//2], L[9*len(L)//10], 'short(<25)', sum(l<25 for l in L), 'trunc', sum(l>=4096 for l in L))
for i in [0,1,2,5,8]:
    r=rows[i]; t=tok.decode(r['resp_ids']); print('---', len(r['resp_ids']), repr(t[:160]), '...', repr(t[-120:]))
# repetition among truncated
rep=0; n=0
for r in rows:
    if len(r['resp_ids'])>=4096:
        n+=1; ids=r['resp_ids']; w=[tuple(ids[j:j+50]) for j in range(0,len(ids)-50,50)]; rep+= (1-len(set(w))/len(w))>0.2
print('truncated', n, 'repetitive', rep)
