import json,glob,sys,statistics as st
import zstandard as zstd
root='/home/sya225/opd/matharena/outputs'
comps=['aime/aime_2024_I','aime/aime_2024_II','aime/aime_2025','custom/amc','custom/beyondaime','hmmt/hmmt_feb_2025','hmmt/hmmt_feb_2026']
def load(f):
    if f.endswith('.zst'):
        with open(f,'rb') as fh: return json.loads(zstd.ZstdDecompressor().decompress(fh.read(), max_output_size=1<<31))
    return json.load(open(f))
def stats(model):
    toks=[];corr=[];mx=0
    for c in comps:
        for f in glob.glob(f'{root}/{c}/local/{model}/*'):
            try: d=load(f)
            except Exception: continue
            dc=d.get('detailed_costs') or []; cr=d.get('correct') or []
            for i,x in enumerate(dc):
                ot=x.get('output_tokens',0); toks.append(ot); mx=max(mx,ot)
                corr.append(bool(cr[i]) if i<len(cr) else False)
    if not toks: return f'{model}: no runs'
    cap=mx; tr=sum(t>=cap-8 for t in toks)/len(toks)
    tc=[c for t,c in zip(toks,corr) if t<cap-8]
    return (f'{model:28s} runs={len(toks):5d} acc={st.mean(corr):.3f} mean_tok={st.mean(toks):6.0f} median={st.median(toks):6.0f} '
            f'cap={cap} trunc={tr:.3f} acc|not_trunc={st.mean(tc) if tc else float("nan"):.3f}')
for m in sys.argv[1:]: print(stats(m))
