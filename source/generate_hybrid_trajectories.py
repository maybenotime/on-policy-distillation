import json
import os
import argparse
import torch
import gc
from vllm import LLM, SamplingParams
from tqdm import tqdm

# Constants for chat templates
QWEN_PROMPT_TEMPLATE = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n<|thought|>\n"

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def generate_student_prefixes(model_name, problems, ratio, max_total_len=512):
    """Phase 1: Student generates the full CoT to determine length, then we keep the prefix."""
    print(f"--- Phase 1: Generating prefixes with Student ({model_name}) ---")
    llm = LLM(model=model_name, trust_remote_code=True, tensor_parallel_size=1, gpu_memory_utilization=0.4)
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=max_total_len)
    
    prompts = [QWEN_PROMPT_TEMPLATE.format(problem=p) for p in problems]
    outputs = llm.generate(prompts, sampling_params)
    
    prefixes = []
    for output in outputs:
        full_text = output.outputs[0].text
        token_ids = output.outputs[0].token_ids
        
        # Calculate truncation point based on ratio
        trunc_idx = max(1, int(len(token_ids) * ratio))
        # Since token_ids -> text mapping is complex with multi-byte chars, 
        # we'll use character ratio of the generated text as a robust engineering proxy.
        char_trunc = max(1, int(len(full_text) * ratio))
        prefixes.append(full_text[:char_trunc])
        
    del llm
    cleanup()
    return prefixes

def teacher_completion(model_name, problems, prefixes, output_file):
    """Phase 2: Teacher completes the reasoning from the student's prefix."""
    print(f"--- Phase 2: Completing trajectories with Teacher ({model_name}) ---")
    # Using TP=1 for A40 (48GB) to fit 14B easily.
    llm = LLM(model=model_name, trust_remote_code=True, tensor_parallel_size=1, gpu_memory_utilization=0.8)
    sampling_params = SamplingParams(temperature=0, max_tokens=1024) # Greedy for teacher
    
    # The prompt for the teacher is: Problem + Student Prefix
    combined_prompts = []
    for p, pre in zip(problems, prefixes):
        combined_prompts.append(QWEN_PROMPT_TEMPLATE.format(problem=p) + pre)
        
    outputs = llm.generate(combined_prompts, sampling_params)
    
    final_data = []
    for i, output in enumerate(outputs):
        teacher_suffix = output.outputs[0].text
        full_cot = prefixes[i] + teacher_suffix
        
        # Convert to swift SFT format (messages)
        record = {
            "messages": [
                {"role": "user", "content": problems[i]},
                {"role": "assistant", "content": f"<|thought|>\n{full_cot}"}
            ]
        }
        final_data.append(record)
        
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in final_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"Successfully saved {len(final_data)} mixed trajectories to {output_file}")
    del llm
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to prompts_math_swift.jsonl")
    parser.add_argument("--output", type=str, required=True, help="Path to save hybrid.jsonl")
    parser.add_argument("--student", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--teacher", type=str, default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--ratio", type=float, default=0.3, help="Truncation ratio (0.0 to 1.0)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for dryrun")
    args = parser.parse_args()

    # Load problems
    problems = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # Extract user message
            for msg in data['messages']:
                if msg['role'] == 'user':
                    problems.append(msg['content'])
                    break
            if args.limit and len(problems) >= args.limit:
                break

    # Run pipeline
    prefixes = generate_student_prefixes(args.student, problems, args.ratio)
    teacher_completion(args.teacher, problems, prefixes, args.output)
