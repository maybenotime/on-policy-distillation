"""Aggregate MathArena outputs: avg pass@1 (mean over the 4 samples, then over
problems) per competition per model. Each <problem>.json.zst already carries the
graded `pass_at_1` and the per-sample `correct` list."""
import glob, json, os, zstandard

ROOT = "/home/sya225/opd/matharena/outputs"
MODELS = ["qwen3_8b", "qwen3_4b", "opd_4b", "qwen3_1p7b", "opd_1p7b", "opd_1p7b_from14b", "opd_1.7b_from4b", "opd_1.7b_fromRLCR4B", "opd_1.7b_fromK4Bcode1200", "opd_1.7b_fromK4Bcode300_16k"]
COMPS = [("aime", "aime_2024_I"), ("aime", "aime_2024_II"), ("aime", "aime_2025"),
         ("hmmt", "hmmt_feb_2025"), ("hmmt", "hmmt_feb_2026"),
         ("custom", "amc"), ("custom", "math500"), ("custom", "beyondaime")]
dec = zstandard.ZstdDecompressor()

tbl = {}
for fam, comp in COMPS:
    for m in MODELS:
        files = glob.glob(os.path.join(ROOT, fam, comp, "local", m, "*.json.zst"))
        if not files:
            continue
        ps, n = [], 0
        for f in files:
            d = json.loads(dec.decompress(open(f, "rb").read()))
            ps.append(d.get("pass_at_1", 0.0) or 0.0)
            n = max(n, len(d.get("correct") or []))
        tbl[(comp, m)] = (100.0 * sum(ps) / len(ps), len(ps), n)

hdr = f"{'competition':<16}" + "".join(f"{m:>12}" for m in MODELS)
print(hdr); print("-" * len(hdr))
for _, comp in COMPS:
    row = f"{comp:<16}"
    for m in MODELS:
        v = tbl.get((comp, m))
        row += f"{v[0]:>11.1f} " if v else f"{'-':>12}"
    print(row)
print()
for _, comp in COMPS:
    v = tbl.get((comp, MODELS[0]))
    if v: print(f"  {comp}: {v[1]} problems x n={v[2]}")
