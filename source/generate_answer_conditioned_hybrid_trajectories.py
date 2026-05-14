import argparse
import gc
import json
from pathlib import Path

import torch
from vllm import LLM, SamplingParams


QWEN_STUDENT_PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "You are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    "{problem}<|im_end|>\n"
    "<|im_start|>assistant\n"
    "<|thought|>\n"
)

QWEN_TEACHER_PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "You are a helpful math assistant. Reason step by step and put the final answer within \\boxed{{}}."
    "<|im_end|>\n"
    "<|im_start|>user\n"
    "Problem:\n{problem}\n\n"
    "The correct final answer is:\n{answer}\n\n"
    "A student has started the solution as follows:\n{prefix}\n\n"
    "Continue from the student's reasoning when possible. If the prefix contains an error, correct it naturally. "
    "Derive the final answer step by step and put the final answer within \\boxed{{}}."
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
    "<|thought|>\n"
    "{prefix}"
)


def cleanup() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def extract_user_problem(row: dict) -> str:
    for message in row.get("messages", []):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


def load_problem_answer_pairs(prompts_path: Path, answers_path: Path, limit: int | None) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with prompts_path.open(encoding="utf-8") as prompt_file, answers_path.open(encoding="utf-8") as answer_file:
        for prompt_line, answer_line in zip(prompt_file, answer_file):
            prompt_row = json.loads(prompt_line)
            answer_row = json.loads(answer_line)

            problem = extract_user_problem(prompt_row)
            answer_problem = answer_row.get("prompt", "")
            if problem != answer_problem:
                raise ValueError(
                    "Prompt mismatch between prompts and answers files at "
                    f"sample {len(pairs)}. The files must share the same order."
                )

            answer = answer_row.get("answer", "").strip()
            if not problem or not answer:
                raise ValueError(f"Empty problem or answer at sample {len(pairs)}.")

            pairs.append((problem, answer))
            if limit is not None and len(pairs) >= limit:
                break

    return pairs


def generate_student_prefixes(
    model_name: str,
    problems: list[str],
    ratio: float,
    max_total_len: int,
    gpu_memory_utilization: float,
) -> list[str]:
    print(f"--- Phase 1: Generating student prefixes ({model_name}, ratio={ratio}) ---")
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=max_total_len)

    prompts = [QWEN_STUDENT_PROMPT_TEMPLATE.format(problem=problem) for problem in problems]
    outputs = llm.generate(prompts, sampling_params)

    prefixes: list[str] = []
    for output in outputs:
        full_text = output.outputs[0].text
        char_trunc = max(1, int(len(full_text) * ratio))
        prefixes.append(full_text[:char_trunc])

    del llm
    cleanup()
    return prefixes


def teacher_completion(
    model_name: str,
    pairs: list[tuple[str, str]],
    prefixes: list[str],
    output_file: Path,
    max_tokens: int,
    gpu_memory_utilization: float,
) -> None:
    print(f"--- Phase 2: Answer-conditioned teacher completion ({model_name}) ---")
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    sampling_params = SamplingParams(temperature=0, max_tokens=max_tokens)

    prompts = [
        QWEN_TEACHER_PROMPT_TEMPLATE.format(problem=problem, answer=answer, prefix=prefix)
        for (problem, answer), prefix in zip(pairs, prefixes)
    ]
    outputs = llm.generate(prompts, sampling_params)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as file:
        for (problem, _), prefix, output in zip(pairs, prefixes, outputs):
            teacher_suffix = output.outputs[0].text
            full_cot = prefix + teacher_suffix
            record = {
                "messages": [
                    {"role": "user", "content": problem},
                    {"role": "assistant", "content": f"<|thought|>\n{full_cot}"},
                ]
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(pairs)} answer-conditioned trajectories to {output_file}")
    del llm
    cleanup()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, required=True, help="Path to prompts_math_swift.jsonl")
    parser.add_argument("--answers", type=Path, required=True, help="Path to prompts_math_with_answers.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="Path to save AC-AT jsonl")
    parser.add_argument("--student", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--teacher", type=str, default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--ratio", type=float, default=0.5, help="Truncation ratio, e.g. 0.3/0.5/0.7")
    parser.add_argument("--limit", type=int, default=5000, help="Number of aligned problem-answer pairs to use")
    parser.add_argument("--student_max_tokens", type=int, default=512)
    parser.add_argument("--teacher_max_tokens", type=int, default=1024)
    parser.add_argument("--student_gpu_memory_utilization", type=float, default=0.4)
    parser.add_argument("--teacher_gpu_memory_utilization", type=float, default=0.8)
    args = parser.parse_args()

    pairs = load_problem_answer_pairs(args.prompts, args.answers, args.limit)
    print(f"Loaded {len(pairs)} aligned problem-answer pairs.")

    prefixes = generate_student_prefixes(
        args.student,
        [problem for problem, _ in pairs],
        args.ratio,
        args.student_max_tokens,
        args.student_gpu_memory_utilization,
    )
    teacher_completion(
        args.teacher,
        pairs,
        prefixes,
        args.output,
        args.teacher_max_tokens,
        args.teacher_gpu_memory_utilization,
    )


if __name__ == "__main__":
    main()
