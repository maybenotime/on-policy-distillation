#!/usr/bin/env python3
"""Build a 7k-row FMD subset (rows where Qwen2.5-7B AND 14B both got correct)
for OPSD training. The opsd_plugin's SingleTeacherPreprocessor needs the full
FMD row schema (per-teacher `.generation` and `.correctness_math_verify`
columns), so we filter the full FMD file by index — NOT the swift-format
prompts file in data/prompts_math_swift_q2both_correct_7k.jsonl which lacks
the teacher columns.

Inputs:
  - FMD jsonl   (default /scratch/sya225/data/final_math_dataset_updated.jsonl on Vulcan,
                /localhome/sya225/Projects/Self-Distillation/data/final_math_dataset_updated.jsonl locally)
  - Index file  data/subset_q2both_correct_7k_idx.json (list of 7000 indices into FMD)

Output:
  Filtered jsonl (one FMD row per line, in the order of the index list).

Usage:
  python scripts/prep_opsd_q2both7k.py [fmd_path] [output_path]
"""
import json
import sys
from pathlib import Path

DEFAULT_FMD = '/localhome/sya225/Projects/Self-Distillation/data/final_math_dataset_updated.jsonl'
DEFAULT_IDX = 'data/subset_q2both_correct_7k_idx.json'
DEFAULT_OUT = 'data/fmd_q2both_correct_7k.jsonl'


def main(fmd_path: Path, out_path: Path, idx_path: Path = Path(DEFAULT_IDX)):
    with idx_path.open() as f:
        indices = sorted(set(json.load(f)))
    print(f'loaded {len(indices)} indices from {idx_path}')

    idx_set = set(indices)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    skipped_missing = 0
    skipped_bad_pair = 0

    # First pass: read rows at the wanted indices.
    rows_by_idx = {}
    with fmd_path.open() as fin:
        for i, line in enumerate(fin):
            if i in idx_set:
                rows_by_idx[i] = line
            if len(rows_by_idx) == len(idx_set):
                break

    # Second pass: write in index order, sanity-checking both teachers correct.
    with out_path.open('w') as fout:
        for i in indices:
            line = rows_by_idx.get(i)
            if line is None:
                skipped_missing += 1
                continue
            row = json.loads(line)
            q7 = row.get('Qwen2.5-7B-Instruct', {})
            q14 = row.get('Qwen2.5-14B-Instruct', {})
            if not (q7.get('correctness_math_verify') and q14.get('correctness_math_verify')):
                skipped_bad_pair += 1
                continue
            fout.write(line if line.endswith('\n') else line + '\n')
            kept += 1

    print(f'wrote {kept} rows to {out_path}')
    if skipped_missing:
        print(f'  (skipped {skipped_missing} indices not found in FMD)')
    if skipped_bad_pair:
        print(f'  (skipped {skipped_bad_pair} rows where both-7B-14B-correct sanity check failed)')


if __name__ == '__main__':
    fmd = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_FMD)
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(DEFAULT_OUT)
    main(fmd, out)
