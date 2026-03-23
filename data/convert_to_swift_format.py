"""
Convert prompts_math_with_answers.jsonl to SWIFT messages format.

Input format:
  {"prompt": "...", "answer": "..."}

Output format (SWIFT messages):
  {"messages": [{"role": "user", "content": "..."}]}

For on-policy GKD (seq_kd=false), the teacher generates responses during
training, so we only need user-role messages in the dataset.
"""

import json
import argparse
from pathlib import Path


def convert(input_path: str, output_path: str, include_answer: bool = False):
    input_path = Path(input_path)
    output_path = Path(output_path)

    count = 0
    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            prompt = item["prompt"]

            messages = [{"role": "user", "content": prompt}]

            if include_answer and "answer" in item:
                messages.append({"role": "assistant", "content": item["answer"]})

            out = {"messages": messages}
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            count += 1

    print(f"Converted {count} samples -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="prompts_math_with_answers.jsonl")
    parser.add_argument("--output", default="prompts_math_swift.jsonl")
    parser.add_argument(
        "--include_answer",
        action="store_true",
        help="Include the ground-truth answer as assistant message. "
             "For on-policy GKD (seq_kd=false) this is NOT needed since "
             "the teacher generates responses online.",
    )
    args = parser.parse_args()
    convert(args.input, args.output, args.include_answer)
