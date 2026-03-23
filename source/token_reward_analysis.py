"""
Token-level reward analysis for on-policy distillation.

For each prompt, generates a rollout from the student model, then computes:
  - per-token log probs from student and each teacher
  - per-token reward: log p_teacher(y_t) - log p_student(y_t)
  - per-token entropy from the student's distribution

Results are saved as a pickle file ready for visualization.

Usage:
    python token_reward_analysis.py \
        --student_model Qwen/Qwen2.5-0.5B-Instruct \
        --teacher_models Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-72B-Instruct \
        --prompts "Solve: 2+2" "What is the capital of France?" \
        --output_path results/token_rewards.pkl \
        --max_new_tokens 512 \
        --temperature 1.0 \
        --batch_size 4
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Optional

from math_evaluator import MathSolutionEvaluator

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name: str, dtype=torch.bfloat16, device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device,
    )
    model.eval()
    return model, tokenizer


def apply_chat_template_if_needed(prompts: list[str], tokenizer) -> list[str]:
    """Wrap plain strings in a minimal chat template if the tokenizer supports it."""
    if tokenizer.chat_template is None:
        return prompts
    formatted = []
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        formatted.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )
    return formatted


# ---------------------------------------------------------------------------
# Rollout: generate completions from student
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_completions(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 1.0,
    batch_size: int = 4,
) -> tuple[list[list[int]], list[list[int]]]:
    """
    Returns:
        prompt_ids_list  : list of token-id lists (one per prompt)
        completion_ids_list: list of token-id lists (one per prompt)
    """
    prompt_ids_list = []
    completion_ids_list = []

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        encoded = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)

        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=gen_config,
        )  # (B, prompt_len + completion_len)

        for i in range(len(batch_prompts)):
            prompt_len = attention_mask[i].sum().item()
            prompt_token_ids = input_ids[i, -prompt_len:].tolist()
            completion_token_ids = output_ids[i, input_ids.shape[1] :].tolist()
            # Strip trailing pad and EOS tokens (they may differ in ID, e.g. Qwen2.5)
            special_ids = {tokenizer.pad_token_id, tokenizer.eos_token_id}
            while completion_token_ids and completion_token_ids[-1] in special_ids:
                completion_token_ids.pop()
            prompt_ids_list.append(prompt_token_ids)
            completion_ids_list.append(completion_token_ids)

    return prompt_ids_list, completion_ids_list


# ---------------------------------------------------------------------------
# Forward pass: per-token log probs and entropy
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_per_token_logps_and_entropy(
    model,
    tokenizer,
    prompt_ids_list: list[list[int]],
    completion_ids_list: list[list[int]],
    temperature: float = 1.0,
    batch_size: int = 4,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Run a forward pass for each (prompt + completion) pair and return:
        logps_list   : list of 1-D tensors, shape (completion_len,)
                       – log p_model(y_t | context) for each completion token
        entropy_list : list of 1-D tensors, shape (completion_len,)
                       – H[p_model(· | context)] at each completion position

    Batching is done by padding to the same length within each mini-batch.
    """
    device = model.device
    logps_list: list[torch.Tensor] = []
    entropy_list: list[torch.Tensor] = []

    for start in range(0, len(prompt_ids_list), batch_size):
        batch_p = prompt_ids_list[start : start + batch_size]
        batch_c = completion_ids_list[start : start + batch_size]

        # Build full sequences (prompt + completion) and track completion lengths
        sequences = [p + c for p, c in zip(batch_p, batch_c)]
        comp_lens = [len(c) for c in batch_c]
        max_len = max(len(s) for s in sequences)

        # Left-pad sequences so they all have the same length
        pad_id = tokenizer.pad_token_id
        input_ids = torch.tensor(
            [[pad_id] * (max_len - len(s)) + s for s in sequences],
            dtype=torch.long,
            device=device,
        )
        attention_mask = (input_ids != pad_id).long()

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        # logits[i, t] predicts token at position t+1 → shift left by 1
        logits = logits[:, :-1, :]  # (B, L-1, V)
        if temperature != 1.0:
            logits = logits / temperature

        log_probs = F.log_softmax(logits, dim=-1)  # (B, L-1, V)

        for i, (seq, comp_len) in enumerate(zip(sequences, comp_lens)):
            full_len = len(seq)
            # The completion tokens occupy positions [full_len-comp_len .. full_len-1] in seq
            # In the padded tensor they start at: max_len - full_len + (full_len - comp_len)
            #                                   = max_len - comp_len
            # After the ±1 logit shift the indices are the same (shifted target):
            comp_start = max_len - comp_len  # index in padded seq where completion starts
            # logits[:, t, :] predicts position t+1, so for token at position comp_start
            # we need logits[:, comp_start-1, :]
            token_ids = input_ids[i, comp_start:]  # completion token ids, shape (comp_len,)
            # logit positions for each completion token: comp_start-1 .. max_len-2
            lp_slice = log_probs[i, comp_start - 1 : max_len - 1, :]  # (comp_len, V)

            # Selected log prob for the actual token
            selected_logps = lp_slice.gather(1, token_ids.unsqueeze(1)).squeeze(1)  # (comp_len,)

            # Entropy H = -sum_v p(v) * log p(v)
            probs = lp_slice.exp()
            entropies = -(probs * lp_slice).sum(dim=-1)  # (comp_len,)

            logps_list.append(selected_logps.cpu())
            entropy_list.append(entropies.cpu())

    return logps_list, entropy_list


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------

def run_analysis(
    student_model_name: str,
    teacher_model_names: list[str],
    prompts: list[str],
    output_path: str,
    answers: Optional[list[str]] = None,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 1.0,
    batch_size: int = 4,
    dtype=torch.bfloat16,
    device: str = "cuda",
):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load student, apply chat template, generate rollouts
    # ------------------------------------------------------------------
    print(f"[1/3] Loading student model: {student_model_name}")
    student_model, student_tokenizer = load_model_and_tokenizer(
        student_model_name, dtype=dtype, device=device
    )

    formatted_prompts = apply_chat_template_if_needed(prompts, student_tokenizer)

    print(f"[1/3] Generating {len(prompts)} rollouts from student …")
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

    # Evaluate correctness against ground-truth answers (if provided)
    if answers is not None:
        evaluator = MathSolutionEvaluator()
        is_correct_list = [
            evaluator.evaluate_single_solution(comp, ans)["final_correct"]
            for comp, ans in zip(completions_text, answers)
        ]
        n_correct = sum(is_correct_list)
        print(f"    Correctness: {n_correct}/{len(is_correct_list)} correct")
    else:
        is_correct_list = [None] * len(completions_text)

    # ------------------------------------------------------------------
    # Step 2: Compute student log probs and entropy
    # ------------------------------------------------------------------
    print("[2/3] Computing student log-probs and entropy …")
    student_logps_list, student_entropy_list = get_per_token_logps_and_entropy(
        student_model,
        student_tokenizer,
        prompt_ids_list,
        completion_ids_list,
        temperature=temperature,
        batch_size=batch_size,
    )

    # Free student model memory before loading teachers
    del student_model
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Step 3: For each teacher, compute log probs and per-token rewards
    # ------------------------------------------------------------------
    teacher_results: dict[str, list[torch.Tensor]] = {}

    for teacher_name in teacher_model_names:
        print(f"[3/3] Loading teacher model: {teacher_name}")
        teacher_model, teacher_tokenizer = load_model_and_tokenizer(
            teacher_name, dtype=dtype, device=device
        )

        # Re-format prompts with teacher tokenizer (may have a different chat template)
        teacher_formatted_prompts = apply_chat_template_if_needed(prompts, teacher_tokenizer)

        # Re-tokenize the same prompts with teacher tokenizer to get prompt_ids
        teacher_prompt_ids_list = []
        for p in teacher_formatted_prompts:
            enc = teacher_tokenizer(p, add_special_tokens=False)
            teacher_prompt_ids_list.append(enc["input_ids"])

        # Assert teacher and student share the same vocabulary (token ID → string mapping)
        # Note: we cannot verify alignment by re-tokenizing the completion text because
        # autoregressive generation produces token sequences that are NOT necessarily the
        # canonical BPE tokenization of the decoded string (e.g. ["Dev","i"] vs ["Devi"]).
        # Vocab equality is the correct and sufficient check for shared token ID space.
        assert student_tokenizer.get_vocab() == teacher_tokenizer.get_vocab(), (
            f"Tokenizer misalign: teacher '{teacher_name}' has a different vocabulary from "
            f"the student '{student_model_name}'. Cannot share completion token IDs."
        )

        # Reuse student completion ids directly (same vocab confirmed above)
        print(f"    Computing teacher log-probs for {teacher_name} …")
        teacher_logps_list, _ = get_per_token_logps_and_entropy(
            teacher_model,
            teacher_tokenizer,
            teacher_prompt_ids_list,
            completion_ids_list,
            temperature=temperature,
            batch_size=batch_size,
        )

        # Per-token reward: log p_teacher(y_t) - log p_student(y_t)
        rewards_list = [t_logps - s_logps for s_logps, t_logps in zip(student_logps_list, teacher_logps_list)]

        teacher_results[teacher_name] = rewards_list

        del teacher_model
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Step 4: Assemble and save results
    # ------------------------------------------------------------------
    records = []
    for i, prompt in enumerate(prompts):
        record = {
            "prompt": prompt,
            "formatted_prompt": formatted_prompts[i],
            "completion": completions_text[i],
            "prompt_token_ids": prompt_ids_list[i],
            "completion_token_ids": completion_ids_list[i],
            # Student quantities
            "student_logps": student_logps_list[i],     # shape: (T,)
            "student_entropy": student_entropy_list[i],  # shape: (T,)
            # Per-teacher rewards: log p_teacher - log p_student
            "teacher_rewards": {
                teacher_name: teacher_results[teacher_name][i]
                for teacher_name in teacher_model_names
            },
            "is_correct": is_correct_list[i],  # bool or None if no answer provided
        }
        records.append(record)

    metadata = {
        "student_model": student_model_name,
        "teacher_models": teacher_model_names,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "num_samples": len(prompts),
    }

    output = {"metadata": metadata, "records": records}
    with open(output_path, "wb") as f:
        pickle.dump(output, f)

    print(f"\nSaved {len(records)} records to {output_path}")
    return output


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Token-level reward analysis for on-policy distillation")
    parser.add_argument("--student_model", required=True, help="Student model name or path")
    parser.add_argument(
        "--teacher_models",
        nargs="+",
        required=True,
        help="One or more teacher model names or paths",
    )
    parser.add_argument(
        "--prompts",
        nargs="*",
        default=None,
        help="Prompts to use (inline). Mutually exclusive with --prompts_file.",
    )
    parser.add_argument(
        "--prompts_file",
        default=None,
        help="Path to prompts file. Supports two formats: "
             "(1) .jsonl – one JSON-encoded string per line; "
             "(2) plain .txt – one prompt per line (no internal newlines).",
    )
    parser.add_argument(
        "--output_path",
        default="results/token_rewards.pkl",
        help="Where to save the pickle output.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # Collect prompts (and optional answers)
    answers = None
    if args.prompts_file is not None:
        with open(args.prompts_file) as f:
            if args.prompts_file.endswith(".jsonl"):
                prompts, raw_answers = [], []
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    if isinstance(obj, str):          # legacy: plain prompt string
                        prompts.append(obj)
                        raw_answers.append(None)
                    else:                             # new: {"prompt": ..., "answer": ...}
                        prompts.append(obj["prompt"])
                        raw_answers.append(obj.get("answer"))
                if any(a is not None for a in raw_answers):
                    answers = raw_answers
            else:
                prompts = [line.strip() for line in f if line.strip()]
    elif args.prompts is not None:
        prompts = args.prompts
    else:
        raise ValueError("Provide either --prompts or --prompts_file")

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    run_analysis(
        student_model_name=args.student_model,
        teacher_model_names=args.teacher_models,
        prompts=prompts,
        output_path=args.output_path,
        answers=answers,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        batch_size=args.batch_size,
        dtype=dtype_map[args.dtype],
        device=args.device,
    )


if __name__ == "__main__":
    main()
