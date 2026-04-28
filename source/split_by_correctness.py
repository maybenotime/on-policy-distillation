"""
Evaluate student and teacher on GSM8K training set and label each prompt
with correctness flags.

Outputs a single JSONL file where each line contains:
  - messages: ms-swift chat format
  - answer: ground truth
  - student_correct: bool
  - teacher_correct: bool

Downstream scripts can filter/group by these flags as needed (e.g. 2x2 design).

Usage:
    python source/split_by_correctness.py \
        --student_model Qwen/Qwen2.5-0.5B-Instruct \
        --teacher_model Qwen/Qwen2.5-3B-Instruct \
        --output_path data/gsm8k_train_correctness.jsonl
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from math_evaluator import MathSolutionEvaluator


def format_chat_prompt(prompt: str, tokenizer) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def extract_gsm8k_answer(answer_str: str) -> str:
    if "####" in answer_str:
        return answer_str.split("####")[-1].strip()
    return answer_str.strip()


def evaluate_completions(completions: list[str], gold_answers: list[str]) -> list[bool]:
    evaluator = MathSolutionEvaluator()
    results = []
    for comp, gold in zip(completions, gold_answers):
        try:
            result = evaluator.evaluate_single_solution(comp, gold)
            results.append(result["final_correct"])
        except Exception:
            results.append(False)
    return results


def generate_with_vllm(model_name: str, prompts: list[str],
                       max_new_tokens: int, temperature: float,
                       gpu_memory_utilization: float = 0.85) -> list[str]:
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=8192,
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        top_p=1.0,
    )
    outputs = llm.generate(prompts, sampling_params)
    completions = [o.outputs[0].text for o in outputs]

    del llm
    torch.cuda.empty_cache()
    return completions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--teacher_model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output_path", default="data/gsm8k_train_correctness.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    args = parser.parse_args()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load GSM8K training set
    print("Loading GSM8K training set...")
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    prompts = [row["question"] for row in dataset]
    gold_answers = [extract_gsm8k_answer(row["answer"]) for row in dataset]
    print(f"Loaded {len(prompts)} prompts")

    # Student evaluation
    print(f"\n[Phase 1] Generating student completions ({args.student_model})...")
    student_tokenizer = AutoTokenizer.from_pretrained(args.student_model)
    student_formatted = [format_chat_prompt(p, student_tokenizer) for p in prompts]
    student_completions = generate_with_vllm(
        args.student_model, student_formatted,
        args.max_new_tokens, args.temperature,
        args.gpu_memory_utilization,
    )
    student_correct = evaluate_completions(student_completions, gold_answers)
    n_sc = sum(student_correct)
    print(f"Student correct: {n_sc}/{len(prompts)} ({100*n_sc/len(prompts):.1f}%)")

    # Teacher evaluation
    print(f"\n[Phase 2] Generating teacher completions ({args.teacher_model})...")
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    teacher_formatted = [format_chat_prompt(p, teacher_tokenizer) for p in prompts]
    teacher_completions = generate_with_vllm(
        args.teacher_model, teacher_formatted,
        args.max_new_tokens, args.temperature,
        args.gpu_memory_utilization,
    )
    teacher_correct = evaluate_completions(teacher_completions, gold_answers)
    n_tc = sum(teacher_correct)
    print(f"Teacher correct: {n_tc}/{len(prompts)} ({100*n_tc/len(prompts):.1f}%)")

    # Save labeled data
    with open(output_path, "w") as f:
        for i in range(len(prompts)):
            record = {
                "messages": [{"role": "user", "content": prompts[i]}],
                "answer": gold_answers[i],
                "student_correct": student_correct[i],
                "teacher_correct": teacher_correct[i],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Print 2x2 summary
    st, sf, tt, tf = 0, 0, 0, 0
    for sc, tc in zip(student_correct, teacher_correct):
        if sc and tc: st += 1
        elif sc and not tc: sf += 1
        elif not sc and tc: tt += 1
        else: tf += 1

    print(f"\n{'':>20} {'T✓':>8} {'T✗':>8} {'Total':>8}")
    print(f"{'S✓':>20} {st:>8} {sf:>8} {st+sf:>8}")
    print(f"{'S✗':>20} {tt:>8} {tf:>8} {tt+tf:>8}")
    print(f"{'Total':>20} {st+tt:>8} {sf+tf:>8} {len(prompts):>8}")
    print(f"\nSaved {len(prompts)} labeled records to {output_path}")


if __name__ == "__main__":
    main()
