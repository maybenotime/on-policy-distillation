import json, sys
recs=[json.loads(l) for l in open(sys.argv[1])]
r=sorted(recs,key=lambda r:r['len'])[len(recs)//2]
t=r['text']; print('len',r['len'],'chars',len(t))
for kw in ['Wait','Let me','verify','check','Actually','However','```','\n\n']:
    print(repr(kw), t.count(kw))
print('---HEAD---'); print(t[:1000]); print('---MID---'); print(t[len(t)//2:len(t)//2+1000]); print('---TAIL---'); print(t[-1200:])
