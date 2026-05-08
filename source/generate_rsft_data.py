#!/usr/bin/env python3
"""Generate RSFT training data from student's own correct rollouts.

For each problem in math_all_teachers_correct.jsonl, generate N rollouts
with the 0.5B student model, keep the first correct one, and save as
ms-swift SFT-format JSONL (messages with assistant turn).

Usage (from repo root):
    python source/generate_rsft_data.py \
        --data_path data/math_all_teachers_correct.jsonl \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --output_path data/rsft_student_correct.jsonl
"""

import argparse
import json
import os
import sys

from math_verify import parse, verify
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

PREDICTION_CONFIGS = [LatexExtractionConfig(), ExprExtractionConfig()]


def is_correct(prediction: str, gold_answer: str) -> bool:
    gold_configs = (
        [LatexExtractionConfig()]
        if ("$" in gold_answer or "\\" in gold_answer)
        else [ExprExtractionConfig()]
    )
    try:
        gold_parsed = parse(gold_answer, extraction_config=gold_configs)
        pred_parsed = parse(prediction, extraction_config=PREDICTION_CONFIGS)
        return bool(verify(gold_parsed, pred_parsed))
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/math_all_teachers_correct.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output_path", default="data/rsft_student_correct.jsonl")
    parser.add_argument("--n_samples", type=int, default=8,
                        help="rollouts per problem (higher = more coverage)")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    args = parser.parse_args()

    # ── Load problems ─────────────────────────────────────────────────────────
    problems = []
    with open(args.data_path) as f:
        for line in f:
            row = json.loads(line)
            problem = row.get("messages", [{}])[0].get("content", row.get("problem", ""))
            answer = row.get("answer", "")
            if problem and answer:
                problems.append({"problem": problem, "answer": answer})
    print(f"Loaded {len(problems)} problems from {args.data_path}")

    # ── Build chat prompts ────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = []
    for p in problems:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": p["problem"]},
        ]
        prompts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )

    # ── Init vLLM and generate ────────────────────────────────────────────────
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_tokens + 512,
    )
    sampling_params = SamplingParams(
        n=args.n_samples,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    print(f"Generating {args.n_samples} rollouts × {len(prompts)} problems ...")
    outputs = llm.generate(prompts, sampling_params)

    # ── Filter correct rollouts ───────────────────────────────────────────────
    results = []
    covered = 0
    total_candidates = 0

    for output, p in zip(outputs, problems):
        found = False
        for candidate in output.outputs:
            total_candidates += 1
            if is_correct(candidate.text, p["answer"]):
                results.append({
                    "messages": [
                        {"role": "system",    "content": SYSTEM_PROMPT},
                        {"role": "user",      "content": p["problem"]},
                        {"role": "assistant", "content": candidate.text},
                    ]
                })
                covered += 1
                found = True
                break  # take first correct rollout per problem

    print(
        f"\nResults: {total_candidates} candidates generated, "
        f"{covered}/{len(problems)} problems have ≥1 correct rollout "
        f"({100*covered/len(problems):.1f}%)"
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(results)} SFT samples → {args.output_path}")


if __name__ == "__main__":
    main()
