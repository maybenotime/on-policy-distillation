"""
OOD Analysis: Measuring teacher feedback degradation on student trajectories.

Three metrics are computed along the token positions of student-generated sequences:
  1. Cumulative PPL: teacher's perplexity on the student trajectory up to each position
  2. Teacher Completion Accuracy: at sampled positions, teacher generates N completions
     and we measure the fraction that reach the correct answer
  3. Teacher Entropy: entropy of the teacher's output distribution at each position

Usage:
    python source/ood_analysis.py \
        --student_model Qwen/Qwen2.5-3B-Instruct \
        --teacher_model Qwen/Qwen2.5-7B-Instruct \
        --prompts_file data/prompts_math_with_answers_correct_100.jsonl \
        --output_path results/ood_analysis.pkl \
        --num_rollouts 10 \
        --num_sample_positions 10
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import torch

from token_reward_analysis import (
    load_model_and_tokenizer,
    apply_chat_template_if_needed,
    generate_completions,
    get_per_token_logps_and_entropy,
)
from math_evaluator import MathSolutionEvaluator


def compute_cumulative_ppl(logps: torch.Tensor) -> torch.Tensor:
    """Given per-token log probs (T,), return cumulative PPL at each position."""
    logps_f32 = logps.float()
    cumsum = torch.cumsum(logps_f32, dim=0)
    positions = torch.arange(1, len(logps_f32) + 1, dtype=torch.float32)
    return torch.exp(-cumsum / positions)


def build_teacher_prefixes(
    prompts_text: list[str],
    completions_text: list[str],
    tokenizer,
    num_sample_positions: int,
    completion_ids_list: list[list[int]],
) -> tuple[list[list[str]], list[list[float]], list[list[int]]]:
    """
    For each rollout, build prefixes at sampled positions for teacher completion.

    Returns:
        all_prefixes: list of (list of prefix strings), one inner list per sample
        all_fractions: list of (list of fractions), one inner list per sample
        all_positions: list of (list of token positions), one inner list per sample
    """
    fractions = [i / num_sample_positions for i in range(num_sample_positions)]

    all_prefixes = []
    all_fractions = []
    all_positions = []

    for idx, (prompt_text, comp_text, comp_ids) in enumerate(
        zip(prompts_text, completions_text, completion_ids_list)
    ):
        T = len(comp_ids)
        prefixes = []
        positions = []

        for frac in fractions:
            pos = int(frac * T)
            positions.append(pos)
            if pos == 0:
                prefixes.append(prompt_text)
            else:
                partial_completion = tokenizer.decode(comp_ids[:pos], skip_special_tokens=True)
                prefixes.append(prompt_text + partial_completion)

        all_prefixes.append(prefixes)
        all_fractions.append(fractions)
        all_positions.append(positions)

    return all_prefixes, all_fractions, all_positions


def run_teacher_completions_vllm(
    teacher_model_name: str,
    all_prefixes: list[list[str]],
    answers: list[str],
    num_rollouts: int,
    max_new_tokens: int,
    temperature: float,
    gpu_memory_utilization: float,
    dtype: str,
) -> list[list[float]]:
    """
    Use vLLM to generate teacher completions at each prefix and evaluate accuracy.

    Returns:
        all_accuracies: list of (list of accuracy floats), one inner list per sample
    """
    from vllm import LLM, SamplingParams

    num_samples = len(all_prefixes)
    num_positions = len(all_prefixes[0])

    flat_prefixes = []
    flat_indices = []
    for sample_idx in range(num_samples):
        for pos_idx in range(num_positions):
            flat_prefixes.append(all_prefixes[sample_idx][pos_idx])
            flat_indices.append((sample_idx, pos_idx))

    total_prompts = len(flat_prefixes)
    expanded_prefixes = []
    for prefix in flat_prefixes:
        expanded_prefixes.extend([prefix] * num_rollouts)

    print(f"    Total vLLM generations: {total_prompts} prefixes x {num_rollouts} rollouts = {total_prompts * num_rollouts}")

    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(teacher_model_name, trust_remote_code=True)
    max_prefix_tokens = max(
        len(_tok.encode(p, add_special_tokens=False))
        for prefixes in all_prefixes for p in prefixes
    )
    del _tok
    estimated_max_len = max_prefix_tokens + max_new_tokens + 256

    llm = LLM(
        model=teacher_model_name,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=estimated_max_len,
        trust_remote_code=True,
    )

    if temperature == 0:
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=max_new_tokens,
            n=1,
        )
    else:
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            n=1,
        )

    print("    Running vLLM inference ...")
    outputs = llm.generate(expanded_prefixes, sampling_params)

    evaluator = MathSolutionEvaluator()

    all_accuracies = [[0.0] * num_positions for _ in range(num_samples)]

    for gen_idx, output in enumerate(outputs):
        flat_idx = gen_idx // num_rollouts
        rollout_idx = gen_idx % num_rollouts
        sample_idx, pos_idx = flat_indices[flat_idx]

        generated_text = output.outputs[0].text
        result = evaluator.evaluate_single_solution(generated_text, answers[sample_idx])
        if result["final_correct"]:
            all_accuracies[sample_idx][pos_idx] += 1.0 / num_rollouts

    del llm
    torch.cuda.empty_cache()

    return all_accuracies


def run_teacher_completions_hf(
    teacher_model_name: str,
    all_prefixes: list[list[str]],
    answers: list[str],
    num_rollouts: int,
    max_new_tokens: int,
    temperature: float,
    dtype,
    device: str,
) -> list[list[float]]:
    """Fallback: use HuggingFace generate for teacher completions (slower)."""
    model, tokenizer = load_model_and_tokenizer(teacher_model_name, dtype=dtype, device=device)

    evaluator = MathSolutionEvaluator()
    num_samples = len(all_prefixes)
    num_positions = len(all_prefixes[0])
    all_accuracies = [[0.0] * num_positions for _ in range(num_samples)]

    total = num_samples * num_positions * num_rollouts
    count = 0

    for sample_idx in range(num_samples):
        for pos_idx in range(num_positions):
            prefix = all_prefixes[sample_idx][pos_idx]
            correct_count = 0

            for _ in range(num_rollouts):
                encoded = tokenizer(
                    prefix,
                    return_tensors="pt",
                    truncation=True,
                    add_special_tokens=False,
                ).to(device)

                output_ids = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else 1.0,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.pad_token_id,
                )

                generated_text = tokenizer.decode(
                    output_ids[0, encoded["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
                result = evaluator.evaluate_single_solution(generated_text, answers[sample_idx])
                if result["final_correct"]:
                    correct_count += 1

                count += 1
                if count % 100 == 0:
                    print(f"    Progress: {count}/{total}")

            all_accuracies[sample_idx][pos_idx] = correct_count / num_rollouts

    del model
    torch.cuda.empty_cache()

    return all_accuracies


def run_analysis(
    student_model_name: str,
    teacher_model_name: str,
    prompts: list[str],
    answers: list[str],
    output_path: str,
    max_new_tokens: int = 8192,
    temperature: float = 1.0,
    top_p: float = 1.0,
    num_rollouts: int = 10,
    num_sample_positions: int = 10,
    batch_size: int = 4,
    use_vllm: bool = True,
    vllm_gpu_memory_utilization: float = 0.9,
    dtype=torch.bfloat16,
    device: str = "cuda",
):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # ---- Phase 1: Student rollout generation ----
    print(f"[Phase 1] Loading student model: {student_model_name}")
    student_model, student_tokenizer = load_model_and_tokenizer(
        student_model_name, dtype=dtype, device=device
    )

    formatted_prompts = apply_chat_template_if_needed(prompts, student_tokenizer)

    print(f"[Phase 1] Generating {len(prompts)} rollouts from student ...")
    prompt_ids_list, completion_ids_list = generate_completions(
        student_model,
        student_tokenizer,
        formatted_prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        batch_size=batch_size,
    )

    completions_text = [
        student_tokenizer.decode(ids, skip_special_tokens=True)
        for ids in completion_ids_list
    ]

    evaluator = MathSolutionEvaluator()
    is_correct_list = [
        evaluator.evaluate_single_solution(comp, ans)["final_correct"]
        for comp, ans in zip(completions_text, answers)
    ]
    n_correct = sum(is_correct_list)
    print(f"[Phase 1] Correctness: {n_correct}/{len(is_correct_list)}")

    # Build formatted prompt text for prefix construction (with chat template applied)
    formatted_prompts_text = formatted_prompts

    del student_model
    torch.cuda.empty_cache()

    # ---- Phase 2: Teacher forward pass (PPL + Entropy) ----
    print(f"\n[Phase 2] Loading teacher model (HF): {teacher_model_name}")
    teacher_model, teacher_tokenizer = load_model_and_tokenizer(
        teacher_model_name, dtype=dtype, device=device
    )

    teacher_formatted_prompts = apply_chat_template_if_needed(prompts, teacher_tokenizer)

    assert student_tokenizer.get_vocab() == teacher_tokenizer.get_vocab(), (
        f"Tokenizer mismatch: teacher '{teacher_model_name}' has a different vocabulary "
        f"from the student '{student_model_name}'. Cannot share completion token IDs."
    )

    teacher_prompt_ids_list = []
    for p in teacher_formatted_prompts:
        enc = teacher_tokenizer(p, add_special_tokens=False)
        teacher_prompt_ids_list.append(enc["input_ids"])

    print("[Phase 2] Computing teacher log-probs and entropy ...")
    teacher_logps_list, teacher_entropy_list = get_per_token_logps_and_entropy(
        teacher_model,
        teacher_tokenizer,
        teacher_prompt_ids_list,
        completion_ids_list,
        temperature=1.0,
        batch_size=batch_size,
    )

    cumulative_ppl_list = [compute_cumulative_ppl(logps) for logps in teacher_logps_list]

    del teacher_model
    torch.cuda.empty_cache()

    # ---- Phase 3: Teacher completion accuracy ----
    print(f"\n[Phase 3] Building prefixes for teacher completion ({num_sample_positions} positions per rollout) ...")

    # Use teacher tokenizer for decoding partial completions
    all_prefixes, all_fractions, all_positions = build_teacher_prefixes(
        teacher_formatted_prompts,
        completions_text,
        teacher_tokenizer,
        num_sample_positions,
        completion_ids_list,
    )

    dtype_str = {torch.bfloat16: "bfloat16", torch.float16: "float16", torch.float32: "float32"}[dtype]

    if use_vllm:
        print(f"[Phase 3] Running teacher completions with vLLM ...")
        all_accuracies = run_teacher_completions_vllm(
            teacher_model_name,
            all_prefixes,
            answers,
            num_rollouts=num_rollouts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            gpu_memory_utilization=vllm_gpu_memory_utilization,
            dtype=dtype_str,
        )
    else:
        print(f"[Phase 3] Running teacher completions with HuggingFace (slow) ...")
        all_accuracies = run_teacher_completions_hf(
            teacher_model_name,
            all_prefixes,
            answers,
            num_rollouts=num_rollouts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            dtype=dtype,
            device=device,
        )

    # ---- Assemble and save results ----
    records = []
    for i in range(len(prompts)):
        record = {
            "prompt": prompts[i],
            "formatted_prompt": formatted_prompts_text[i],
            "completion": completions_text[i],
            "prompt_token_ids": prompt_ids_list[i],
            "completion_token_ids": completion_ids_list[i],
            "is_correct": is_correct_list[i],
            "teacher_logps": teacher_logps_list[i],
            "teacher_entropy": teacher_entropy_list[i],
            "cumulative_ppl": cumulative_ppl_list[i],
            "sample_positions": all_positions[i],
            "sample_fractions": all_fractions[i],
            "completion_accuracy": all_accuracies[i],
        }
        records.append(record)

    metadata = {
        "student_model": student_model_name,
        "teacher_model": teacher_model_name,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "num_rollouts": num_rollouts,
        "num_sample_positions": num_sample_positions,
        "num_samples": len(prompts),
    }

    output = {"metadata": metadata, "records": records}
    with open(output_path, "wb") as f:
        pickle.dump(output, f)

    print(f"\nSaved {len(records)} records to {output_path}")
    return output


def main():
    parser = argparse.ArgumentParser(description="OOD analysis: teacher feedback degradation on student trajectories")
    parser.add_argument("--student_model", required=True)
    parser.add_argument("--teacher_model", required=True)
    parser.add_argument("--prompts_file", required=True)
    parser.add_argument("--output_path", default="results/ood_analysis.pkl")
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_rollouts", type=int, default=10)
    parser.add_argument("--num_sample_positions", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--use_vllm", action="store_true", default=False)
    parser.add_argument("--no_vllm", action="store_true", default=False)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # Load prompts and answers
    with open(args.prompts_file) as f:
        prompts, raw_answers = [], []
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if isinstance(obj, str):
                prompts.append(obj)
                raw_answers.append(None)
            else:
                prompts.append(obj["prompt"])
                raw_answers.append(obj.get("answer"))

    if any(a is None for a in raw_answers):
        print("ERROR: All prompts must have ground-truth answers for OOD analysis.", file=sys.stderr)
        sys.exit(1)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    use_vllm = args.use_vllm and not args.no_vllm

    run_analysis(
        student_model_name=args.student_model,
        teacher_model_name=args.teacher_model,
        prompts=prompts,
        answers=raw_answers,
        output_path=args.output_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        num_rollouts=args.num_rollouts,
        num_sample_positions=args.num_sample_positions,
        batch_size=args.batch_size,
        use_vllm=use_vllm,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        dtype=dtype_map[args.dtype],
        device=args.device,
    )


if __name__ == "__main__":
    main()
