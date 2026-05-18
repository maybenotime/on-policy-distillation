#!/usr/bin/env python
"""Per-model accuracy on (prompt, answer)-format jsonl.

Generates n_rollouts completions per prompt with vLLM (default 1 = single
rollout), parses the boxed answer via math_verify, and saves an
analysis-friendly JSON:

{
  "metadata": {model, n_prompts, temperature, top_p, max_tokens, data_file,
               data_range, timestamp},
  "accuracy": float,
  "n_correct": int,
  "records": [{idx, is_correct, extracted_answer, ground_truth_answer,
               completion}, ...]
}

For n_rollouts > 1, each record's is_correct = any rollout correct (pass@n),
and we still keep all completions under a `completions` list field.
"""
from __future__ import annotations
import argparse
import datetime
import json
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompts_jsonl", required=True)
    p.add_argument("--n_prompts", type=int, default=10000)
    p.add_argument("--start_idx", type=int, default=0,
                   help="Skip first N rows of prompts_jsonl. Output `idx` will be global (start_idx+local).")
    p.add_argument("--n_rollouts", type=int, default=1)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--max_tokens", type=int, default=8192)
    p.add_argument("--gpu_mem_util", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=16384)
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--data_parallel_size", type=int, default=1)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--enable_thinking", choices=["true", "false", "none"], default="none",
                   help="Forwarded to tokenizer.apply_chat_template. Use 'false' for Qwen3 non-thinking.")
    args = p.parse_args()

    print(f"[acc] model={args.model}", flush=True)
    print(f"[acc] n_prompts={args.n_prompts} n_rollouts={args.n_rollouts} "
          f"T={args.temperature} top_p={args.top_p} max_tokens={args.max_tokens}",
          flush=True)

    with open(args.prompts_jsonl) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    rows = rows[args.start_idx : args.start_idx + args.n_prompts]
    prompts = [r["prompt"] for r in rows]
    answers = [str(r["answer"]) for r in rows]
    print(f"[acc] loaded {len(rows)} (prompt, answer) pairs", flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    chat_kwargs = {}
    if args.enable_thinking in ("true", "false"):
        chat_kwargs["enable_thinking"] = (args.enable_thinking == "true")
    formatted = [
        tok.apply_chat_template(
            [{"role": "user", "content": pr + "\nPlease reason step by step, and put your final answer within \\boxed{}."}],
            tokenize=False,
            add_generation_prompt=True,
            **chat_kwargs,
        )
        for pr in prompts
    ]

    from vllm import LLM, SamplingParams
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        seed=args.seed,
        tensor_parallel_size=args.tensor_parallel_size,
        data_parallel_size=args.data_parallel_size,
    )
    sp = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=args.n_rollouts,
        seed=args.seed,
    )

    print(f"[acc] vLLM generating {len(formatted)} × {args.n_rollouts} ...", flush=True)
    t0 = time.time()
    outputs = llm.generate(formatted, sp)
    print(f"[acc] generation took {time.time()-t0:.1f}s", flush=True)

    sys.path.insert(0, str(Path(__file__).parent))
    from math_evaluator import MathSolutionEvaluator
    evaluator = MathSolutionEvaluator()

    records = []
    n_correct = 0
    t1 = time.time()
    for i, out in enumerate(outputs):
        ans = answers[i]
        completions = [r.text for r in out.outputs]
        # Evaluate each rollout
        per_rollout = []
        any_correct = False
        extracted_first = ""
        for txt in completions:
            r = evaluator.evaluate_single_solution(txt, ans)
            ok = bool(r.get("final_correct"))
            ext = r.get("predicted_answers")
            ext_str = ", ".join(ext) if isinstance(ext, list) else (str(ext) if ext else "")
            per_rollout.append({"is_correct": ok, "extracted_answer": ext_str})
            if not extracted_first:
                extracted_first = ext_str
            any_correct = any_correct or ok
        if any_correct:
            n_correct += 1

        if args.n_rollouts == 1:
            records.append({
                "idx": args.start_idx + i,
                "is_correct": per_rollout[0]["is_correct"],
                "extracted_answer": per_rollout[0]["extracted_answer"],
                "ground_truth_answer": ans,
                "completion": completions[0],
            })
        else:
            records.append({
                "idx": args.start_idx + i,
                "is_correct_any": any_correct,
                "per_rollout": per_rollout,
                "ground_truth_answer": ans,
                "completions": completions,
            })

        if (i + 1) % 500 == 0:
            print(f"[acc] scored {i+1}/{len(outputs)} acc-so-far={n_correct/(i+1):.4f} ({time.time()-t1:.0f}s)", flush=True)

    N = len(records)
    accuracy = n_correct / N

    print(f"\n=== {args.model} ===", flush=True)
    print(f"n_prompts={N}, n_rollouts={args.n_rollouts}, T={args.temperature}", flush=True)
    print(f"accuracy = {n_correct}/{N} = {accuracy:.4f}", flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(
            {
                "metadata": {
                    "model": args.model,
                    "n_prompts": N,
                    "n_rollouts": args.n_rollouts,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_tokens": args.max_tokens,
                    "data_file": Path(args.prompts_jsonl).name,
                    "data_range": [args.start_idx, args.start_idx + args.n_prompts],
                    "seed": args.seed,
                    "timestamp": datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                },
                "accuracy": accuracy,
                "n_correct": n_correct,
                "records": records,
            },
            f,
            indent=1,
        )
    print(f"\n[acc] saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
