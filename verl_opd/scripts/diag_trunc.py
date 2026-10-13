#!/usr/bin/env python
"""Is the 14% empty-extraction rate on LCB v6 caused by max_tokens truncation?

Regenerates LCB v6 and reports, per model, the finish_reason split and whether
an unclosed ``` fence coincides with length-truncation. If truncation drives it,
the baseline LCB numbers are understated and the teacher/student inversion may
be an artefact rather than a real ordering.
"""
import argparse, json
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--max_tokens", type=int, default=6144)
a = ap.parse_args()

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
from lcb_runner.prompts.code_generation import PromptConstants, get_generic_question_template_answer
from lcb_runner.utils.extraction_utils import extract_code
from lcb_runner.lm_styles import LMStyle

probs = load_code_generation_dataset(release_version="release_v6", start_date="2025-01-01")
tok = AutoTokenizer.from_pretrained(a.model)
prompts = [tok.apply_chat_template(
    [{"role": "system", "content": PromptConstants.SYSTEM_MESSAGE_GENERIC},
     {"role": "user", "content": get_generic_question_template_answer(q)}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False) for q in probs]

llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=0.85,
          max_model_len=a.max_tokens + 4096, enforce_eager=True)
outs = llm.generate(prompts, SamplingParams(temperature=0.0, top_p=1.0, max_tokens=a.max_tokens))

fin, empty_by_fin, toks = Counter(), Counter(), []
for o in outs:
    c = o.outputs[0]
    fin[c.finish_reason] += 1
    toks.append(len(c.token_ids))
    if not extract_code(c.text, LMStyle.CodeQwenInstruct).strip():
        empty_by_fin[c.finish_reason] += 1

toks.sort()
print(f"\n=== {a.model}  max_tokens={a.max_tokens}  n={len(outs)} ===")
print("finish_reason:", dict(fin))
print("empty extraction by finish_reason:", dict(empty_by_fin))
print(f"output tokens: p50={toks[len(toks)//2]} p90={toks[int(len(toks)*.9)]} max={toks[-1]}")
print(f"TRUNCATED share: {fin.get('length',0)/len(outs):.1%}")
