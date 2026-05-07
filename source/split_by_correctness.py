"""
Evaluate student and teacher on a dataset and label each prompt
with correctness flags.

Supports:
  --dataset gsm8k          : openai/gsm8k train split
  --dataset math           : hendrycks/MATH train split (all subjects)
  --dataset custom         : custom JSONL (requires --data_path)
                             Each line must have: {"messages": [{"role":"user","content":"..."}], "answer": "..."}
                             or: {"prompt": "...", "answer": "..."}

Options:
  --max_samples N          : limit to first N samples (for dry-run testing)
  --data_path PATH         : path to JSONL when --dataset custom

Usage:
    python source/split_by_correctness.py \
        --dataset gsm8k \
        --student_model Qwen/Qwen2.5-0.5B-Instruct \
        --teacher_model Qwen/Qwen2.5-3B-Instruct

    python source/split_by_correctness.py \
        --dataset math \
        --max_samples 20 \
        --student_model Qwen/Qwen2.5-0.5B-Instruct \
        --teacher_model Qwen/Qwen2.5-14B-Instruct

    python source/split_by_correctness.py \
        --dataset custom \
        --data_path ../data/prompts_math_swift.jsonl \
        --student_model Qwen/Qwen2.5-0.5B-Instruct \
        --teacher_model Qwen/Qwen2.5-14B-Instruct
"""

import argparse
import json
import re
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


def extract_math_answer(solution: str) -> str:
    match = re.search(r'\\boxed\{(.+)\}', solution)
    if match:
        return match.group(1)
    return solution.strip()


def load_data(dataset_name: str, data_path: str = None, max_samples: int = None):
    if dataset_name == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split="train")
        prompts = [row["question"] for row in dataset]
        gold_answers = [extract_gsm8k_answer(row["answer"]) for row in dataset]
    elif dataset_name == "math":
        subjects = ['algebra', 'counting_and_probability', 'geometry',
                    'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
        prompts, gold_answers = [], []
        for subj in subjects:
            ds = load_dataset("EleutherAI/hendrycks_math", subj, split="train")
            prompts.extend([row["problem"] for row in ds])
            gold_answers.extend([extract_math_answer(row["solution"]) for row in ds])
    elif dataset_name == "custom":
        assert data_path, "--data_path is required when --dataset custom"
        with open(data_path) as f:
            items = [json.loads(l) for l in f if l.strip()]
        prompts = [
            item["messages"][0]["content"] if "messages" in item else item["prompt"]
            for item in items
        ]
        gold_answers = [item["answer"] for item in items]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose gsm8k / math / custom")

    if max_samples is not None and max_samples < len(prompts):
        prompts = prompts[:max_samples]
        gold_answers = gold_answers[:max_samples]
        print(f"[max_samples] Trimmed to {max_samples} prompts")

    return prompts, gold_answers


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
    parser.add_argument("--dataset", default="gsm8k", help="gsm8k | math | custom")
    parser.add_argument("--data_path", default=None, help="JSONL path when --dataset custom")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit to first N prompts (dry-run)")
    parser.add_argument("--student_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--teacher_model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    args = parser.parse_args()

    if args.output_path is None:
        student_tag = args.student_model.split("/")[-1]
        teacher_tag = args.teacher_model.split("/")[-1]
        args.output_path = f"data/{args.dataset}_train_correctness_{student_tag}_{teacher_tag}.jsonl"

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.dataset} training set...")
    prompts, gold_answers = load_data(args.dataset, args.data_path, args.max_samples)
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
