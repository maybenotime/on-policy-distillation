#!/usr/bin/env python3
"""Build a GRPO-ready jsonl from final_math_dataset_updated.jsonl.

GRPO with built-in ms-swift `accuracy` reward (`MathAccuracy` in
`swift/rewards/orm.py`) expects each row to have:
  - `messages`: a list of chat messages ending with a user turn
  - `solution`: ground-truth answer string; the reward function parses
    `\\boxed{...}` from the model's completion and verifies it against
    `solution` via `math_verify`.

This script takes the first N rows of
  /localhome/sya225/Projects/Self-Distillation/data/final_math_dataset_updated.jsonl
(which is 1:1 aligned by index with data/prompts_math_swift.jsonl), keeps
only the user turn from `messages`, and wraps the `answer` field as
`\\boxed{<answer>}` so it round-trips through math_verify cleanly.

Usage:
  python scripts/prep_grpo_data.py 10000 data/prompts_math_grpo_10k.jsonl
"""
import json
import sys
from pathlib import Path

FMD_PATH = Path('/localhome/sya225/Projects/Self-Distillation/data/final_math_dataset_updated.jsonl')


def main(n: int, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    dropped = 0
    with FMD_PATH.open() as fin, out_path.open('w') as fout:
        for i, line in enumerate(fin):
            if kept >= n:
                break
            row = json.loads(line)
            answer = (row.get('answer') or '').strip()
            if not answer:
                dropped += 1
                continue
            # Keep only the user turn — GRPO generates the assistant turn.
            user_msg = next(
                (m for m in row['messages'] if m.get('role') == 'user'),
                None,
            )
            if user_msg is None:
                dropped += 1
                continue
            out_row = {
                'messages': [{'role': 'user', 'content': user_msg['content']}],
                'solution': f'\\boxed{{{answer}}}',
            }
            fout.write(json.dumps(out_row, ensure_ascii=False) + '\n')
            kept += 1
    print(f'wrote {kept} rows to {out_path} (dropped {dropped} rows missing answer/user turn)')


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('data/prompts_math_grpo_10k.jsonl')
    main(n, out_path)
