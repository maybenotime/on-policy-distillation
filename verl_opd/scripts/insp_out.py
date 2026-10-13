"""Inspect matharena outputs of a model config: response lengths, repetition, answer presence."""
import glob, json, sys, os, zstandard as zstd
R="/home/sya225/opd/matharena/outputs"; cfg=sys.argv[1]
files=sorted(glob.glob(f"{R}/aime/aime_2024_I/local/{cfg}/*.json.zst"))[:4]
for fp in files:
    j=json.loads(zstd.ZstdDecompressor().stream_reader(open(fp,"rb")).read())
    keys=list(j.keys())
    msgs=j.get("messages", j.get("messages_history"))
    print("==", os.path.basename(fp), "keys:", keys[:8], "correct:", j.get("correct"), "cost:", j.get("cost"))
    # find the assistant texts
    texts=[]
    def walk(x):
        if isinstance(x, dict):
            if x.get("role")=="assistant" and isinstance(x.get("content"), str): texts.append(x["content"])
            for v in x.values(): walk(v)
        elif isinstance(x, list):
            for v in x: walk(v)
    walk(msgs)
    for t in texts[:2]:
        w=t.split(); wins=[" ".join(w[i:i+30]) for i in range(0,max(1,len(w)-30),30)]; rep=1-len(set(wins))/max(1,len(wins))
        print(f"  len_chars={len(t)} words={len(w)} rep30={rep:.2f} head={t[:120]!r} tail={t[-150:]!r}")
