"""Build aligned 5k answer-conditioned data for LI/AC-OPSD experiments.

The original OPSD experiments use math_all_teachers_correct.jsonl, which only
contains 840 filtered problems. This builder aligns prompts_math_swift.jsonl and
prompts_math_with_answers.jsonl so Chapter 5 online variants can run on the same
first-5k problem set used by AT-SFT and AC-AT-SFT.
"""

import argparse
import json
from pathlib import Path
from typing import Any


def extract_user_problem(row: dict[str, Any]) -> str:
    for message in row.get("messages", []):
        if message.get("role") == "user":
            return str(message.get("content", "")).strip()
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=5000)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with (
        args.prompts.open(encoding="utf-8") as prompt_file,
        args.answers.open(encoding="utf-8") as answer_file,
        args.output.open("w", encoding="utf-8") as output_file,
    ):
        for idx, (prompt_line, answer_line) in enumerate(zip(prompt_file, answer_file), start=1):
            if args.limit is not None and written >= args.limit:
                break

            prompt_row = json.loads(prompt_line)
            answer_row = json.loads(answer_line)

            problem = extract_user_problem(prompt_row)
            answer_problem = str(answer_row.get("prompt", "")).strip()
            answer = str(answer_row.get("answer", "")).strip()

            if not problem:
                raise ValueError(f"Empty prompt at aligned row {idx}.")
            if problem != answer_problem:
                raise ValueError(
                    "Prompt mismatch between prompts and answers files at "
                    f"aligned row {idx}. The files must be in the same order."
                )
            if not answer:
                raise ValueError(f"Empty answer at aligned row {idx}.")

            record = {
                "messages": [{"role": "user", "content": problem}],
                "reference_answer": answer,
            }
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} aligned answer-conditioned rows to {args.output}")


if __name__ == "__main__":
    main()
