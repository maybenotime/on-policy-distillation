#!/usr/bin/env python
"""Generate HumanEval+ / LiveCodeBench-v6 completions with the Qwen3
NON-THINKING chat template, so eval matches the distillation setup.

Runs in code_eval_env. Uses lcb_runner's OWN prompt builder and code extractor,
so prompts/extraction match the official harness exactly; only the chat wrapper
differs (LCB's own Qwen branch hardcodes a Qwen1.5-72B tokenizer path and never
passes enable_thinking, which is why we wrap the messages ourselves).

Requires: cwd=LiveCodeBench (lcb_runner reads few-shot data relative to cwd),
PYTHONPATH=LiveCodeBench, PATH including the venv bin (vllm shells out to ninja).
"""
import argparse, json, os, re
from pathlib import Path


def build_lcb(release, start_date):
    from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
    from lcb_runner.prompts.code_generation import (
        PromptConstants, get_generic_question_template_answer)
    probs = load_code_generation_dataset(release_version=release, start_date=start_date)
    return [{"key": str(q.question_id),
             "messages": [{"role": "system", "content": PromptConstants.SYSTEM_MESSAGE_GENERIC},
                          {"role": "user", "content": get_generic_question_template_answer(q)}]}
            for q in probs]


def build_humanevalplus():
    from evalplus.data import get_human_eval_plus
    probs = get_human_eval_plus()
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
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=-1)
    ap.add_argument("--min_p", type=float, default=0.0)
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
              max_model_len=min(args.max_tokens + 4096, 40960),
              enforce_eager=False,
              compilation_config={"cudagraph_mode": "PIECEWISE"})
    outs = llm.generate(prompts, SamplingParams(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, min_p=args.min_p,
                                                max_tokens=args.max_tokens, n=args.n))

    tag = args.tag or os.path.basename(args.model.rstrip("/"))
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    if args.bench == "lcb_v6":
        from lcb_runner.utils.extraction_utils import extract_code
        from lcb_runner.lm_styles import LMStyle
        recs = [{"question_id": it["key"],
                 "code_list": [extract_code(c.text, LMStyle.CodeQwenInstruct) for c in o.outputs]}
                for it, o in zip(items, outs)]
        f = out_dir / f"lcb_v6_{tag}.json"
        f.write_text(json.dumps(recs))
        # Keep the raw completions: extract_code takes the LAST fenced block, and
        # models that append a usage demo get that demo scored instead of their
        # solution. Saving raw lets the extractor be revised (and the size of the
        # artefact measured) without paying for regeneration.
        raw = [{"question_id": it["key"], "raw_list": [c.text for c in o.outputs]}
               for it, o in zip(items, outs)]
        (out_dir / f"lcb_v6_{tag}_raw.json").write_text(json.dumps(raw))
        empty = sum(1 for r in recs if not r["code_list"][0].strip())
    else:
        f = out_dir / f"humanevalplus_{tag}.jsonl"
        fraw = out_dir / f"humanevalplus_{tag}_raw.jsonl"
        empty = nodef = 0
        with f.open("w") as fh, fraw.open("w") as fr:
            for it, o in zip(items, outs):
                # Emit EVERY sample, not just outputs[0]: evalplus groups rows by
                # task_id and averages, so one row per sample is what turns --n 4
                # into a real avg pass@1 instead of a single noisy draw.
                for c in o.outputs:
                    txt = c.text
                    m = re.findall(r"```(?:python)?\n(.*?)```", txt, re.DOTALL)
                    # Take the last block that DEFINES something, not simply the
                    # last block. Models often append a usage demo as a trailing
                    # block; blindly taking m[-1] scored that demo as the answer
                    # (33% of samples on the 14B-teacher checkpoint had no `def`
                    # at all, costing ~17 pass@1 points of pure artefact).
                    cand = [b for b in m if "def " in b or "class " in b]
                    sol = cand[-1] if cand else (m[-1] if m else txt)
                    if not sol.strip():
                        empty += 1
                    if "def " not in sol:
                        nodef += 1
                    fh.write(json.dumps({"task_id": it["key"], "solution": sol}) + "\n")
                    # keep the raw completion so the extractor can be revised
                    # later without paying for regeneration
                    fr.write(json.dumps({"task_id": it["key"], "raw": txt}) + "\n")
        print(f"[gen] no-def samples: {nodef}", flush=True)
    print(f"[gen] wrote {f}  ({len(items)} problems, {empty} empty extractions)", flush=True)


if __name__ == "__main__":
    main()
