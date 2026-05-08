"""
Teacher reflection probe.

Two complementary experiments on INCORRECT student trajectories from an
existing OOD analysis pkl:

  A) LOGITS: single teacher forward pass per trajectory; extract
     P(reflection_token | prefix_t) at every position t along the
     completion. Tells us how the teacher's probability of switching to
     a reflection token evolves as the trajectory progresses.

  B) SAMPLING: at K position fractions along each trajectory, sample N
     teacher continuations and count which contain a reflection token in
     the first M tokens, and which finally produce the correct answer.
     Provides causal-style 4-cell contingency (reflection ⨯ correctness)
     and concrete recovery case studies.

Reuses utilities from token_reward_analysis.py and math_evaluator.py.

Usage:
    python source/teacher_reflection_probe.py \\
        --input_pkl results/ood_math_100_Qwen3-1.7B_Qwen3-8B.pkl \\
        --teacher_model Qwen/Qwen3-8B \\
        --output_path results/reflection_probe_Qwen3-1.7B_Qwen3-8B.pkl \\
        --mode both --enable_thinking false
"""

import argparse
import pickle
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from token_reward_analysis import load_model_and_tokenizer
from math_evaluator import MathSolutionEvaluator


REFLECTION_KEYWORDS = [
    "wait", "Wait",
    "alternatively", "Alternatively",
    "actually", "Actually",
    "instead", "Instead",
    "hmm", "Hmm",
    "reconsider", "Reconsider",
    "but wait", "But wait",
]

# Substring patterns for sampling-side detection (case insensitive, word boundary).
REFLECTION_PATTERNS = [
    r"\bwait\b", r"\balternatively\b", r"\bactually\b",
    r"\binstead\b", r"\bhmm\b", r"\breconsider\b",
    r"\bre-?examine\b", r"\brethink\b",
]


def collect_kw_token_ids(tokenizer) -> dict[str, list[int]]:
    """Return {kw -> [first_token_ids of variants]} for next-token logit lookup.

    For each keyword we add 4 variants — bare, leading-space, capitalized,
    leading-space + capitalized — and keep just the FIRST token id of each
    (multi-token kws use first-token activation as a lower bound).
    """
    bases = ["wait", "alternatively", "actually", "instead",
             "hmm", "reconsider", "rethink"]
    out = {}
    for base in bases:
        ids = set()
        for variant in [base, " " + base, base.capitalize(), " " + base.capitalize()]:
            tok = tokenizer.encode(variant, add_special_tokens=False)
            if tok:
                ids.add(tok[0])
        out[base] = sorted(ids)
    return out


def relative_bin(values: np.ndarray, num_bins: int = 100) -> np.ndarray:
    """Bin a length-T array into num_bins relative-position buckets (mean per bin)."""
    T = len(values)
    if T == 0:
        return np.full(num_bins, np.nan)
    out = np.full(num_bins, np.nan)
    for b in range(num_bins):
        lo = int(b * T / num_bins)
        hi = int((b + 1) * T / num_bins)
        if hi <= lo:
            hi = lo + 1
        seg = values[lo:hi]
        if len(seg):
            out[b] = float(np.nanmean(seg))
    return out


# ---------------------------------------------------------------------------
# Experiment A: per-position next-token reflection prob via HF forward pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_logits_probe(teacher_model, teacher_tokenizer, records,
                     kw_token_ids: dict[str, list[int]],
                     num_bins: int = 100,
                     dtype=torch.bfloat16):
    """For each (incorrect) record, do one teacher forward pass over
    [prompt_ids + completion_ids], and at every position t (over the
    COMPLETION) extract log P(reflection_kw | prefix up to t).
    """
    device = next(teacher_model.parameters()).device
    pad_id = teacher_tokenizer.pad_token_id or teacher_tokenizer.eos_token_id

    out_records = []
    for ridx, r in enumerate(records):
        prompt_ids = r["prompt_token_ids"]
        completion_ids = r["completion_token_ids"]
        if not completion_ids or len(completion_ids) < 2:
            continue
        full_ids = list(prompt_ids) + list(completion_ids)
        x = torch.tensor([full_ids], dtype=torch.long, device=device)

        logits = teacher_model(input_ids=x).logits[0]   # [seq, vocab]
        logprobs = torch.log_softmax(logits.float(), dim=-1)

        # Per-token next-token logprob aligned with completion tokens.
        # logits at position p predicts token at position p+1.
        # Position of the FIRST completion token in `full_ids` is len(prompt_ids).
        # So predicted-by-prefix-of-length-t corresponds to logits[len(prompt_ids)-1+t]
        # for t in 0..len(completion_ids)-1.
        T = len(completion_ids)
        base = len(prompt_ids) - 1  # logits[base + t] predicts token (base + t + 1)
        kw_logp = {kw: np.zeros(T, dtype=np.float32) for kw in kw_token_ids}
        for t in range(T):
            row = logprobs[base + t]  # shape [vocab]
            for kw, ids in kw_token_ids.items():
                if not ids:
                    kw_logp[kw][t] = -np.inf
                else:
                    # P(any variant of kw as next token) = sum of probs of variants
                    lps = row[ids]
                    # logsumexp
                    m = lps.max()
                    kw_logp[kw][t] = float((m + torch.log(torch.exp(lps - m).sum())).cpu())

        # Bin to relative positions for aggregation
        binned = {kw: relative_bin(np.exp(kw_logp[kw]), num_bins) for kw in kw_logp}

        out_records.append({
            "trajectory_idx": ridx,
            "is_correct": r["is_correct"],
            "completion_len_tokens": T,
            "kw_logprobs_raw": kw_logp,    # {kw: array [T]}
            "kw_probs_binned": binned,     # {kw: array [num_bins]}
        })

        if ridx % 5 == 0:
            print(f"    [logits] {ridx + 1}/{len(records)} done", flush=True)

    return out_records


# ---------------------------------------------------------------------------
# Experiment B: prefix sampling with vLLM + reflection / correctness audit
# ---------------------------------------------------------------------------

def run_sampling_probe(teacher_model_name, records, answers, position_fractions,
                       num_samples, max_new_tokens, temperature,
                       gpu_memory_utilization, dtype, teacher_tokenizer,
                       reflection_first_n_tokens: int = 100):
    """For each record, at each position fraction, sample N teacher
    continuations from prompt+completion[:pos]. Record:
      - whether the sample contains a reflection keyword in first K tokens
      - whether the sample's final answer is correct (math_verify)
      - the first 200 chars of each sample (for case studies)
    """
    from vllm import LLM, SamplingParams

    # Build all prefixes up front to make a single batched vLLM call.
    flat_prefixes = []
    flat_meta = []  # (record_idx, pos_idx, sample_idx)

    record_prefixes_by_pos = []  # list of [pos_idx -> prefix_text]
    for ridx, r in enumerate(records):
        completion_ids = r["completion_token_ids"]
        if not completion_ids:
            record_prefixes_by_pos.append([])
            continue
        per_pos = []
        for pidx, frac in enumerate(position_fractions):
            cut = max(1, int(frac * len(completion_ids)))
            prefix_ids = list(r["prompt_token_ids"]) + list(completion_ids[:cut])
            prefix_text = teacher_tokenizer.decode(prefix_ids, skip_special_tokens=False)
            per_pos.append(prefix_text)
            for sidx in range(num_samples):
                flat_prefixes.append(prefix_text)
                flat_meta.append((ridx, pidx, sidx))
        record_prefixes_by_pos.append(per_pos)

    print(f"  [sampling] generating {len(flat_prefixes)} samples ({len(records)} records "
          f"× {len(position_fractions)} positions × {num_samples} samples)")

    # vLLM does its own model loading
    max_prefix_tokens = max(
        len(teacher_tokenizer.encode(p, add_special_tokens=False))
        for p in flat_prefixes
    )
    estimated_max_len = max_prefix_tokens + max_new_tokens + 256

    llm = LLM(
        model=teacher_model_name,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=estimated_max_len,
        trust_remote_code=True,
    )
    sp = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        n=1,
    )
    print("  [sampling] running vLLM ...")
    outputs = llm.generate(flat_prefixes, sp)

    evaluator = MathSolutionEvaluator()
    refl_re = re.compile("|".join(REFLECTION_PATTERNS), flags=re.IGNORECASE)

    # Aggregate per (record, pos, sample)
    out = {}  # (ridx, pidx, sidx) -> dict
    for o, meta in zip(outputs, flat_meta):
        ridx, pidx, sidx = meta
        gen_text = o.outputs[0].text
        # First-K-token region for reflection check
        first_ids = teacher_tokenizer.encode(gen_text, add_special_tokens=False)[:reflection_first_n_tokens]
        first_text = teacher_tokenizer.decode(first_ids, skip_special_tokens=True)
        has_refl = bool(refl_re.search(first_text))

        # Correctness on the full new completion
        eval_res = evaluator.evaluate_single_solution(gen_text, answers[ridx])
        is_correct = bool(eval_res.get("final_correct", False))

        out[(ridx, pidx, sidx)] = {
            "has_reflection": has_refl,
            "is_correct_after_recovery": is_correct,
            "first_chars": gen_text[:200],
            "full_text_len": len(gen_text),
        }

    # Pack per-record summary
    per_record = []
    for ridx, r in enumerate(records):
        per_pos = []
        for pidx, frac in enumerate(position_fractions):
            samples = []
            for sidx in range(num_samples):
                key = (ridx, pidx, sidx)
                if key in out:
                    samples.append(out[key])
            per_pos.append({
                "position_fraction": frac,
                "samples": samples,
            })
        per_record.append({
            "trajectory_idx": ridx,
            "is_correct_original": r["is_correct"],
            "completion_len_tokens": len(r["completion_token_ids"]),
            "positions": per_pos,
        })
    return per_record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pkl", required=True,
                        help="Existing OOD analysis pkl produced by ood_analysis.py")
    parser.add_argument("--teacher_model", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--mode", choices=["logits", "sampling", "both"], default="both")
    parser.add_argument("--filter", choices=["incorrect", "correct", "all"], default="incorrect",
                        help="Which student trajectories to probe")
    # logits
    parser.add_argument("--num_bins", type=int, default=100)
    # sampling
    parser.add_argument("--position_fractions", type=float, nargs="+",
                        default=[0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.85)
    # misc
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--enable_thinking", default="none", choices=["true", "false", "none"])
    args = parser.parse_args()

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading OOD pkl: {args.input_pkl}")
    with open(args.input_pkl, "rb") as f:
        data = pickle.load(f)
    metadata = data["metadata"]
    all_records = data["records"]

    if args.filter == "incorrect":
        records = [r for r in all_records if not r["is_correct"]]
    elif args.filter == "correct":
        records = [r for r in all_records if r["is_correct"]]
    else:
        records = all_records
    print(f"Filter='{args.filter}': {len(records)}/{len(all_records)} trajectories")

    # gold answers — need to re-extract from input pkl. Some pkls keep
    # answers separately; we encode gold_answer per record via metadata
    # if present; otherwise we look at neighboring fields. For OOD pkl
    # we don't store the gold answer, so we need to reload from prompts file.
    # Workaround: reload the prompts file path stored implicitly — no,
    # not stored. So caller must pass --gold_answers_pkl or we recompute
    # by re-reading the original prompts JSONL.
    # For our case the input file is data/ood_math_100.jsonl; we infer
    # by matching prompt text. Cheap & robust.

    # Reload gold answers from the original prompts file
    prompts_file = "data/ood_math_100.jsonl"
    import json
    prompt_to_answer = {}
    if Path(prompts_file).exists():
        with open(prompts_file) as f:
            for line in f:
                row = json.loads(line)
                prompt_to_answer[row["prompt"]] = row["answer"]
    answers = []
    missing = 0
    for r in records:
        a = prompt_to_answer.get(r["prompt"])
        if a is None:
            missing += 1
            a = ""
        answers.append(a)
    if missing:
        print(f"WARNING: {missing} prompts could not be matched to gold answers")

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    enable_thinking_map = {"true": True, "false": False, "none": None}
    enable_thinking = enable_thinking_map[args.enable_thinking]

    output = {
        "metadata": {
            "input_pkl": args.input_pkl,
            "teacher_model": args.teacher_model,
            "student_model": metadata.get("student_model"),
            "filter": args.filter,
            "num_records": len(records),
            "mode": args.mode,
            "position_fractions": args.position_fractions,
            "num_samples": args.num_samples,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "enable_thinking": enable_thinking,
            "reflection_keywords": REFLECTION_KEYWORDS,
            "reflection_patterns": REFLECTION_PATTERNS,
        },
    }

    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)

    if args.mode in ("logits", "both"):
        print(f"\n[Phase A] Logits probe (teacher forward pass per trajectory) ...")
        kw_token_ids = collect_kw_token_ids(teacher_tokenizer)
        for kw, ids in kw_token_ids.items():
            decoded = [repr(teacher_tokenizer.decode([i])) for i in ids]
            print(f"  {kw}: ids={ids} ({decoded})")
        teacher_model, _ = load_model_and_tokenizer(args.teacher_model, dtype=dtype, device="cuda")
        teacher_model.eval()
        logits_records = run_logits_probe(teacher_model, teacher_tokenizer, records,
                                          kw_token_ids=kw_token_ids,
                                          num_bins=args.num_bins, dtype=dtype)
        output["logits_records"] = logits_records
        output["kw_token_ids"] = kw_token_ids
        del teacher_model
        torch.cuda.empty_cache()
    if args.mode in ("sampling", "both"):
        print(f"\n[Phase B] Sampling probe (vLLM teacher continuations) ...")
        sampling_records = run_sampling_probe(
            teacher_model_name=args.teacher_model,
            records=records,
            answers=answers,
            position_fractions=args.position_fractions,
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            dtype=args.dtype,
            teacher_tokenizer=teacher_tokenizer,
        )
        output["sampling_records"] = sampling_records

    with open(args.output_path, "wb") as f:
        pickle.dump(output, f)
    print(f"\nSaved {args.output_path}")


if __name__ == "__main__":
    main()
