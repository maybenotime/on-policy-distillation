#!/usr/bin/env python
"""Generate HumanEval+ / LiveCodeBench-v6 completions with the Qwen3
NON-THINKING chat template, so eval matches the distillation setup exactly.

Runs in a Python-3.12 venv (verl_opd) because the LiveCodeBench venv is 3.11 and
its vllm 0.27.1 dies at engine startup with
`TypeError: type 'array.array' is not subscriptable`.

We import ONLY `lcb_runner.benchmarks.code_generation` for the dataset (needs
PYTHONPATH + cwd = LiveCodeBench, which reads data files relative to cwd).
The prompt template and code extractor are inlined verbatim from
lcb_runner/prompts/code_generation.py and lcb_runner/utils/extraction_utils.py,
because importing `lcb_runner.prompts.*` pulls `from anthropic import
HUMAN_PROMPT`, a constant removed in modern anthropic SDKs.

Scoring stays on the official paths:
  LCB  -> lcb_runner.runner.custom_evaluator  (LiveCodeBench venv)
  HE+  -> evalplus.evaluate --samples          (evalplus_env)
"""
import argparse, json, os, re
from pathlib import Path

# --- verbatim from lcb_runner/prompts/code_generation.py -------------------
SYSTEM_MESSAGE_GENERIC = (
    "You are an expert Python programmer. You will be given a question (problem specification) "
    "and will generate a correct Python program that matches the specification and passes all tests."
)
FORMATTING_MESSAGE_WITH_STARTER_CODE = (
    "You will use the following starter code to write the solution to the problem "
    "and enclose your code within delimiters."
)
FORMATTING_WITHOUT_STARTER_CODE = (
    "Read the inputs from stdin solve the problem and write the answer to stdout "
    "(do not directly test on the sample inputs). Enclose your code within delimiters as follows. "
    "Ensure that when the python program runs, it reads the inputs, runs the algorithm and "
    "writes output to STDOUT."
)


def generic_question_template(question):
    prompt = f"### Question:\n{question.question_content}\n\n"
    if question.starter_code:
        prompt += f"### Format: {FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{question.starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt


def extract_code(model_output: str) -> str:
    """verbatim from lcb_runner/utils/extraction_utils.py (non-CodeLLaMa branch)"""
    outputlines = model_output.split("\n")
    indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    if len(indexlines) < 2:
        return ""
    return "\n".join(outputlines[indexlines[-2] + 1 : indexlines[-1]])
# ---------------------------------------------------------------------------


def build_lcb(release, start_date):
    from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
    probs = load_code_generation_dataset(release_version=release, start_date=start_date)
    return [{"key": str(q.question_id),
             "messages": [{"role": "system", "content": SYSTEM_MESSAGE_GENERIC},
                          {"role": "user", "content": generic_question_template(q)}]}
            for q in probs]


def build_humanevalplus():
    probs = json.load(open("/home/sya225/opd/bench/humanevalplus_problems.json"))
    instr = ("Complete the following Python function. Return the COMPLETE function "
             "implementation inside a single ```python code block. No tests, no explanations.")
    return [{"key": tid,
             "messages": [{"role": "system", "content": "You are an expert Python programmer."},
                          {"role": "user", "content": f"{instr}\n\n```python\n{p['prompt']}```"}]}
            for tid, p in probs.items()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bench", required=True, choices=["lcb_v6", "humanevalplus"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--max_tokens", type=int, default=6144)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu_mem_util", type=float, default=0.85)
    ap.add_argument("--release", default="release_v6")
    ap.add_argument("--start_date", default="2025-01-01")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    items = build_lcb(args.release, args.start_date) if args.bench == "lcb_v6" else build_humanevalplus()
    print(f"[gen] bench={args.bench} problems={len(items)} model={args.model}", flush=True)

    prompts = [tok.apply_chat_template(it["messages"], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
               for it in items]
    assert "<think>" in prompts[0], "expected Qwen3 pre-filled empty <think></think> block"
    print("[gen] prompt tail:", repr(prompts[0][-55:]), flush=True)

    llm = LLM(model=args.model, tensor_parallel_size=args.tp, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_mem_util,
              max_model_len=args.max_tokens + 4096, enforce_eager=True)
    outs = llm.generate(prompts, SamplingParams(temperature=args.temperature, top_p=1.0,
                                                max_tokens=args.max_tokens, n=args.n))

    tag = args.tag or os.path.basename(args.model.rstrip("/"))
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    if args.bench == "lcb_v6":
        recs = [{"question_id": it["key"], "code_list": [extract_code(c.text) for c in o.outputs]}
                for it, o in zip(items, outs)]
        f = out_dir / f"lcb_v6_{tag}.json"
        f.write_text(json.dumps(recs))
        empty = sum(1 for r in recs if not r["code_list"][0].strip())
    else:
        f = out_dir / f"humanevalplus_{tag}.jsonl"
        empty = 0
        with f.open("w") as fh:
            for it, o in zip(items, outs):
                txt = o.outputs[0].text
                m = re.findall(r"```(?:python)?\n(.*?)```", txt, re.DOTALL)
                sol = m[-1] if m else txt
                if not sol.strip():
                    empty += 1
                fh.write(json.dumps({"task_id": it["key"], "solution": sol}) + "\n")
    print(f"[gen] wrote {f}  ({len(items)} problems, {empty} empty extractions)", flush=True)


if __name__ == "__main__":
    main()
