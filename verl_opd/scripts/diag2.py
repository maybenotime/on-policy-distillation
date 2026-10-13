#!/usr/bin/env python
"""Is the LCB length-truncation caused by greedy decoding degeneracy?

Both Qwen3 models ship generation_config.json with do_sample=true,
temperature=0.6, top_k=20, top_p=0.95 -- our eval used temperature=0.0, which the
Qwen3 card explicitly warns can cause endless repetition. Raising max_tokens
6144 -> 16384 changed the truncation count by exactly zero, which is the
signature of a decoding loop rather than answers that need more room.

Reports the finish_reason split at the given sampling params, plus a repetition
score for truncated outputs (longest tail fraction covered by a repeated 40-char
block) and a sample tail so the failure mode is visible rather than inferred.
"""
import argparse, json
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--max_tokens", type=int, default=6144)
ap.add_argument("--temperature", type=float, default=0.6)
ap.add_argument("--top_p", type=float, default=0.95)
ap.add_argument("--top_k", type=int, default=20)
ap.add_argument("--dump", type=int, default=2)
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
outs = llm.generate(prompts, SamplingParams(temperature=a.temperature, top_p=a.top_p,
                                            top_k=a.top_k, max_tokens=a.max_tokens))

def repetition_score(s: str, block: int = 40) -> float:
    """Fraction of the last 4000 chars covered by a block that repeats >=3 times."""
    tail = s[-4000:]
    if len(tail) < block * 3:
        return 0.0
    best = 0
    for start in range(0, len(tail) - block, max(1, block // 2)):
        pat = tail[start:start + block]
        c = tail.count(pat)
        if c >= 3:
            best = max(best, c * block)
    return min(1.0, best / len(tail))

fin, empty_by_fin, toks, reps = Counter(), Counter(), [], []
dumped = 0
for o in outs:
    c = o.outputs[0]
    fin[c.finish_reason] += 1
    toks.append(len(c.token_ids))
    if c.finish_reason == "length":
        reps.append(repetition_score(c.text))
        if dumped < a.dump:
            print(f"\n----- TRUNCATED SAMPLE {dumped} (last 400 chars) -----")
            print(repr(c.text[-400:]))
            dumped += 1
    if not extract_code(c.text, LMStyle.CodeQwenInstruct).strip():
        empty_by_fin[c.finish_reason] += 1

toks.sort()
print(f"\n=== {a.model} T={a.temperature} top_p={a.top_p} top_k={a.top_k} "
      f"max_tokens={a.max_tokens} n={len(outs)} ===")
print("finish_reason:", dict(fin))
print("empty extraction by finish_reason:", dict(empty_by_fin))
print(f"output tokens: p50={toks[len(toks)//2]} p90={toks[int(len(toks)*.9)]} max={toks[-1]}")
print(f"TRUNCATED share: {fin.get('length',0)/len(outs):.1%}")
if reps:
    reps.sort()
    print(f"repetition score of truncated (0=none,1=fully looped): "
          f"p50={reps[len(reps)//2]:.2f} mean={sum(reps)/len(reps):.2f}")
