"""
Sample 100 prompts from each of three datasets for OOD analysis.

Outputs three JSONL files with {"prompt": ..., "answer": ...} format.

Usage:
    python source/sample_datasets.py [--seed 42] [--num_samples 100]
"""

import argparse
import json
import random
import re
from pathlib import Path


def extract_gsm8k_answer(answer_text: str) -> str:
    """Extract the final numeric answer after #### in GSM8K format."""
    match = re.search(r"####\s*(.+)", answer_text)
    if match:
        return match.group(1).strip()
    return answer_text.strip().split("\n")[-1].strip()


def sample_gsm8k(num_samples: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(num_samples, len(ds)))

    samples = []
    for idx in indices:
        item = ds[idx]
        samples.append({
            "prompt": item["question"],
            "answer": extract_gsm8k_answer(item["answer"]),
        })
    return samples


def sample_math(num_samples: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("hendrydong/hendrycks_math", split="train")
    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(num_samples, len(ds)))

    samples = []
    for idx in indices:
        item = ds[idx]
        answer = item.get("answer") or item.get("solution", "")
        samples.append({
            "prompt": item["problem"],
            "answer": answer,
        })
    return samples


def sample_train(num_samples: int, seed: int, path: str) -> list[dict]:
    with open(path) as f:
        all_items = [json.loads(line) for line in f if line.strip()]

    rng = random.Random(seed)
    chosen = rng.sample(all_items, min(num_samples, len(all_items)))

    return [{"prompt": item["prompt"], "answer": item["answer"]} for item in chosen]


def save_jsonl(samples: list[dict], output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Saved {len(samples)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--train_path", default="data/prompts_math_with_answers.jsonl")
    parser.add_argument("--output_dir", default="data")
    args = parser.parse_args()

    print(f"Sampling {args.num_samples} from each dataset (seed={args.seed})")

    print("\n[1/3] GSM8K train ...")
    gsm8k_samples = sample_gsm8k(args.num_samples, args.seed)
    save_jsonl(gsm8k_samples, f"{args.output_dir}/ood_gsm8k_100.jsonl")

    print("\n[2/3] MATH train (hendrydong/hendrycks_math) ...")
    math_samples = sample_math(args.num_samples, args.seed)
    save_jsonl(math_samples, f"{args.output_dir}/ood_math_100.jsonl")

    print("\n[3/3] Current training set ...")
    train_samples = sample_train(args.num_samples, args.seed, args.train_path)
    save_jsonl(train_samples, f"{args.output_dir}/ood_train_100.jsonl")

    print("\nDone.")


if __name__ == "__main__":
    main()
