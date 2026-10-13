"""Partial MathArena aggregation: per-model, per-competition accuracy over finished problems (n samples each)."""
import glob, json, os, sys, zstandard as zstd, io
R = "/home/sya225/opd/matharena/outputs"
comps = ["aime/aime_2024_I", "aime/aime_2024_II", "aime/aime_2025", "custom/amc", "custom/beyondaime", "hmmt/hmmt_feb_2025", "hmmt/hmmt_feb_2026", "custom/math500"]
models = sys.argv[1:]
def load(p):
    with open(p, "rb") as f:
        return json.loads(zstd.ZstdDecompressor().stream_reader(f).read())
print(f"{'model':32s}" + "".join(f"{c.split('/')[-1]:>14s}" for c in comps))
for m in models:
    row = []
    for c in comps:
        d = os.path.join(R, c, "local", m)
        files = glob.glob(d + "/*.json.zst")
        if not files: row.append("-"); continue
        cor = tot = 0
        for fp in files:
            j = load(fp)
            for s in j.get("messages_history", j.get("samples", [])) if isinstance(j, dict) else j:
                pass
            # matharena stores per-problem: "correct": [bool per sample]
            cs = j.get("correct", None) if isinstance(j, dict) else None
            if cs is None: continue
            cor += sum(bool(x) for x in cs); tot += len(cs)
        row.append(f"{100*cor/tot:5.1f}({len(files)})" if tot else f"?({len(files)})")
    print(f"{m:32s}" + "".join(f"{x:>14s}" for x in row))
