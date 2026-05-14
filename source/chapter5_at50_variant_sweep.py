import argparse
import gc
import json
import re
from pathlib import Path
from statistics import mean, median


STUDENT_PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "You are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    "{problem}<|im_end|>\n"
    "<|im_start|>assistant\n"
    "<|thought|>\n"
)


def cleanup() -> None:
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        return


def extract_user_problem(row: dict) -> str:
    for message in row.get("messages", []):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


def extract_assistant(row: dict) -> str:
    for message in row.get("messages", []):
        if message.get("role") == "assistant":
            return message.get("content", "")
    return ""


def strip_thought(text: str) -> str:
    return re.sub(r"^\s*<\|thought\|>\s*", "", text).strip()


def load_problem_answer_pairs(prompts_path: Path, answers_path: Path, limit: int) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with prompts_path.open(encoding="utf-8") as prompt_file, answers_path.open(encoding="utf-8") as answer_file:
        for prompt_line, answer_line in zip(prompt_file, answer_file):
            prompt_row = json.loads(prompt_line)
            answer_row = json.loads(answer_line)
            problem = extract_user_problem(prompt_row)
            answer_problem = answer_row.get("prompt", "")
            if problem != answer_problem:
                raise ValueError(f"Prompt/answer mismatch at sample {len(pairs)}")
            answer = answer_row.get("answer", "").strip()
            if not problem or not answer:
                raise ValueError(f"Empty problem or answer at sample {len(pairs)}")
            pairs.append((problem, answer))
            if len(pairs) >= limit:
                break
    return pairs


def load_drafts(draft_path: Path, pairs: list[tuple[str, str]]) -> list[str]:
    drafts: list[str] = []
    with draft_path.open(encoding="utf-8") as file:
        for idx, line in enumerate(file):
            if idx >= len(pairs):
                break
            row = json.loads(line)
            problem = extract_user_problem(row)
            if problem != pairs[idx][0]:
                raise ValueError(f"Draft problem mismatch at sample {idx}")
            drafts.append(strip_thought(extract_assistant(row)))
    if len(drafts) != len(pairs):
        raise ValueError(f"Need {len(pairs)} drafts, got {len(drafts)}")
    return drafts


def last_boxed_end(text: str) -> int | None:
    starts = [match.start() for match in re.finditer(r"\\boxed\s*\{", text)]
    if not starts:
        return None
    start = starts[-1]
    brace_start = text.find("{", start)
    depth = 0
    for idx in range(brace_start, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return idx + 1
    return None


def remove_duplicate_lines(text: str) -> str:
    cleaned: list[str] = []
    previous = ""
    repeat_count = 0
    for line in text.splitlines():
        normalized = re.sub(r"\s+", " ", line).strip()
        if normalized and normalized == previous:
            repeat_count += 1
            if repeat_count >= 1:
                continue
        else:
            repeat_count = 0
        cleaned.append(line)
        previous = normalized
    return "\n".join(cleaned)


def has_repeated_tail(text: str) -> bool:
    compact = re.sub(r"\s+", " ", text[-1800:]).strip()
    for size in (60, 90, 120):
        chunks = [compact[i : i + size] for i in range(0, max(0, len(compact) - size), size)]
        counts: dict[str, int] = {}
        for chunk in chunks:
            if len(chunk.strip()) < size * 0.8:
                continue
            counts[chunk] = counts.get(chunk, 0) + 1
            if counts[chunk] >= 3:
                return True
    return False


def looks_truncated(text: str) -> bool:
    tail = text.strip()[-160:]
    if re.search(r"(=|\+|-|,|\\frac\{|\\sqrt\{|\\left|\(|\[|\{)\s*$", tail):
        return True
    return "\\boxed" not in text and not tail.endswith((".", "。", "!", "?"))


def cleanup_solution(text: str, answer: str, force_box: bool = True) -> str:
    cleaned = text.replace("<|answer|>", "").replace("<|assistant|>", "")
    cleaned = re.sub(r"<\|im_[^>]+?\|>", "", cleaned)
    cleaned = remove_duplicate_lines(cleaned).strip()

    boxed_end = last_boxed_end(cleaned)
    if boxed_end is not None:
        cleaned = cleaned[:boxed_end].strip()
    elif force_box:
        cleaned = cleaned.rstrip()
        cleaned += f"\n\nTherefore, the final answer is \\(\\boxed{{{answer}}}\\)."

    return cleaned.strip()


def make_record(problem: str, solution: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": problem},
            {"role": "assistant", "content": f"<|thought|>\n{solution.strip()}"},
        ]
    }


def save_records(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate_student_prefixes(
    model_name: str,
    problems: list[str],
    ratio: float,
    max_tokens: int,
    gpu_memory_utilization: float,
) -> list[str]:
    from vllm import LLM, SamplingParams

    print(f"Generating student prefixes: model={model_name}, ratio={ratio}")
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=max_tokens)
    prompts = [STUDENT_PROMPT_TEMPLATE.format(problem=problem) for problem in problems]
    outputs = llm.generate(prompts, sampling_params)
    prefixes = []
    for output in outputs:
        full_text = output.outputs[0].text
        char_trunc = max(1, int(len(full_text) * ratio))
        prefixes.append(full_text[:char_trunc])
    del llm
    cleanup()
    return prefixes


def enhanced_continue_prompt(problem: str, answer: str, prefix: str) -> str:
    return (
        "<|im_start|>system\n"
        "You are a careful math teacher. Produce concise, non-repetitive reasoning and end with the exact boxed answer."
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Problem:\n{problem}\n\n"
        f"Correct final answer:\n{answer}\n\n"
        f"Student prefix:\n{prefix}\n\n"
        "Continue the solution only if the prefix is consistent with the correct answer. "
        "If the prefix is wrong or repetitive, silently correct it and write a clean concise continuation. "
        "Use at most 8 reasoning steps. Do not repeat equivalent equations, cases, or sentences. "
        "If the derivation starts to repeat, stop and give the final answer. "
        f"End exactly with:\n\\boxed{{{answer}}}\n"
        "Do not write anything after the final boxed answer."
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<|thought|>\n"
        f"{prefix}"
    )


def repair_prompt(problem: str, answer: str, draft: str) -> str:
    return (
        "<|im_start|>system\n"
        "You are a careful math solution editor. Rewrite flawed draft solutions into concise correct solutions."
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Problem:\n{problem}\n\n"
        f"Correct final answer:\n{answer}\n\n"
        f"Draft solution:\n{draft}\n\n"
        "The draft may contain repetition, truncation, wrong turns, or a missing final boxed answer. "
        "Rewrite it into a clean solution. Preserve useful correct ideas from the draft, but remove repeated, unfinished, "
        "or incorrect parts. Do not mention the draft or these instructions. Use at most 8 reasoning steps. "
        f"End exactly with:\n\\boxed{{{answer}}}\n"
        "Do not write anything after the final boxed answer."
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<|thought|>\n"
    )


def batch_generate(
    model_name: str,
    prompts: list[str],
    max_tokens: int,
    gpu_memory_utilization: float,
) -> list[str]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    sampling_params = SamplingParams(temperature=0, max_tokens=max_tokens)
    outputs = llm.generate(prompts, sampling_params)
    texts = [output.outputs[0].text for output in outputs]
    del llm
    cleanup()
    return texts


def run_prompt_enhanced(args: argparse.Namespace, pairs: list[tuple[str, str]]) -> list[dict]:
    problems = [problem for problem, _ in pairs]
    prefixes = generate_student_prefixes(
        args.student,
        problems,
        args.ratio,
        args.student_max_tokens,
        args.student_gpu_memory_utilization,
    )
    prompts = [
        enhanced_continue_prompt(problem, answer, prefix)
        for (problem, answer), prefix in zip(pairs, prefixes)
    ]
    suffixes = batch_generate(
        args.teacher,
        prompts,
        args.teacher_max_tokens,
        args.teacher_gpu_memory_utilization,
    )
    records = []
    for (problem, answer), prefix, suffix in zip(pairs, prefixes, suffixes):
        solution = cleanup_solution(prefix + suffix, answer, force_box=False)
        records.append(make_record(problem, solution))
    return records


def run_postprocess_only(pairs: list[tuple[str, str]], drafts: list[str]) -> list[dict]:
    records = []
    for (problem, answer), draft in zip(pairs, drafts):
        solution = cleanup_solution(draft, answer, force_box=True)
        records.append(make_record(problem, solution))
    return records


def run_teacher_repair(args: argparse.Namespace, pairs: list[tuple[str, str]], drafts: list[str]) -> list[dict]:
    prompts = [repair_prompt(problem, answer, draft) for (problem, answer), draft in zip(pairs, drafts)]
    rewrites = batch_generate(
        args.teacher,
        prompts,
        args.repair_max_tokens,
        args.teacher_gpu_memory_utilization,
    )
    records = []
    for (problem, answer), rewrite in zip(pairs, rewrites):
        solution = cleanup_solution(rewrite, answer, force_box=True)
        records.append(make_record(problem, solution))
    return records


def summarize_records(name: str, records: list[dict], pairs: list[tuple[str, str]]) -> dict:
    answers = [answer for _, answer in pairs]
    texts = [extract_assistant(record) for record in records]
    lengths = [len(text) for text in texts]
    boxed = sum("\\boxed" in text for text in texts)
    exact = 0
    for text, answer in zip(texts, answers):
        compact_text = re.sub(r"\s+", "", text)
        compact_answer = re.sub(r"\s+", "", answer)
        if f"\\boxed{{{compact_answer}}}" in compact_text:
            exact += 1
    return {
        "name": name,
        "rows": len(records),
        "boxed": boxed,
        "boxed_rate": boxed / len(records) if records else 0,
        "exact_gold_boxed": exact,
        "exact_gold_boxed_rate": exact / len(records) if records else 0,
        "special_token": sum("<|answer|>" in text or "<|assistant|>" in text or "<|im_" in text for text in texts),
        "repeated_tail": sum(has_repeated_tail(text) for text in texts),
        "truncated_heuristic": sum(looks_truncated(text) for text in texts),
        "len_mean": mean(lengths) if lengths else 0,
        "len_median": median(lengths) if lengths else 0,
        "len_min": min(lengths) if lengths else 0,
        "len_max": max(lengths) if lengths else 0,
    }


def write_summary(output_dir: Path, summaries: list[dict]) -> None:
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summaries, file, ensure_ascii=False, indent=2)
    with (output_dir / "summary.md").open("w", encoding="utf-8") as file:
        file.write("| variant | rows | boxed | exact gold boxed | special | repeated tail | truncated | mean len | median len |\n")
        file.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in summaries:
            file.write(
                f"| {row['name']} | {row['rows']} | {row['boxed']} ({row['boxed_rate']:.1%}) "
                f"| {row['exact_gold_boxed']} ({row['exact_gold_boxed_rate']:.1%}) "
                f"| {row['special_token']} | {row['repeated_tail']} | {row['truncated_heuristic']} "
                f"| {row['len_mean']:.1f} | {row['len_median']:.1f} |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True, help="Existing AT-50 first5k draft data")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--student", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--teacher", type=str, default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--student_max_tokens", type=int, default=512)
    parser.add_argument("--teacher_max_tokens", type=int, default=1536)
    parser.add_argument("--repair_max_tokens", type=int, default=1536)
    parser.add_argument("--student_gpu_memory_utilization", type=float, default=0.4)
    parser.add_argument("--teacher_gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["prompt_enhanced", "postprocess_only", "teacher_repair"],
        choices=["prompt_enhanced", "postprocess_only", "teacher_repair"],
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs = load_problem_answer_pairs(args.prompts, args.answers, args.limit)
    drafts = load_drafts(args.draft, pairs)
    print(f"Loaded {len(pairs)} aligned AT-50 samples.")

    outputs: dict[str, list[dict]] = {}
    if "prompt_enhanced" in args.variants:
        print("Running variant: prompt_enhanced")
        outputs["prompt_enhanced"] = run_prompt_enhanced(args, pairs)
    if "postprocess_only" in args.variants:
        print("Running variant: postprocess_only")
        outputs["postprocess_only"] = run_postprocess_only(pairs, drafts)
    if "teacher_repair" in args.variants:
        print("Running variant: teacher_repair")
        outputs["teacher_repair"] = run_teacher_repair(args, pairs, drafts)

    summaries = []
    for name, records in outputs.items():
        output_path = args.output_dir / f"at50_{name}_{args.limit}.jsonl"
        save_records(output_path, records)
        summaries.append(summarize_records(name, records, pairs))
        print(f"Saved {name}: {output_path}")

    write_summary(args.output_dir, summaries)
    print(f"Saved summaries to {args.output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
