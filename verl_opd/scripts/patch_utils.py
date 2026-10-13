"""Patch verl/workers/rollout/vllm_rollout/utils.py::extract_prompt_logprobs:
with env VERL_TEACHER_KEEP_SAMPLED=1 and top-k mode, append one extra column holding the ACTUAL (sampled) token's
id and exact log-prob, so downstream losses can use the exact teacher log-prob even when the token is outside the top-k."""
import sys
p = sys.argv[1]; s = open(p).read()
if "VERL_TEACHER_KEEP_SAMPLED" in s:
    print("already patched"); sys.exit(0)
old = '''            prompt_ids = [None] * num_prompt_logprobs
            prompt_logprobs = [None] * num_prompt_logprobs
            # We get either top-k logprobs or top-k plus the sampled logprob (if sampled token is not in top-k)
            assert len(logprobs_dict) in [num_prompt_logprobs, num_prompt_logprobs + 1], len(logprobs_dict)
            for token_id_str, token_logprob in logprobs_dict.items():
                rank = token_logprob.rank
                if rank > num_prompt_logprobs:
                    continue  # the sampled token is not in the top-k
                logprob = token_logprob.logprob
                prompt_ids[rank - 1] = int(token_id_str)
                prompt_logprobs[rank - 1] = logprob
            prompt_logprobs_ls.append(prompt_logprobs)
            prompt_ids_ls.append(prompt_ids)
'''
new = '''            prompt_ids = [None] * (num_prompt_logprobs + keep_sampled)
            prompt_logprobs = [None] * (num_prompt_logprobs + keep_sampled)
            # We get either top-k logprobs or top-k plus the sampled logprob (if sampled token is not in top-k)
            assert len(logprobs_dict) in [num_prompt_logprobs, num_prompt_logprobs + 1], len(logprobs_dict)
            for token_id_str, token_logprob in logprobs_dict.items():
                rank = token_logprob.rank
                if rank > num_prompt_logprobs:
                    continue  # the sampled token is not in the top-k
                logprob = token_logprob.logprob
                prompt_ids[rank - 1] = int(token_id_str)
                prompt_logprobs[rank - 1] = logprob
            if keep_sampled:
                # extra last column: the actual token of this position and its exact log-prob (always present in the dict)
                actual_id = int(actual_ids[pos])
                prompt_ids[num_prompt_logprobs] = actual_id
                prompt_logprobs[num_prompt_logprobs] = logprobs_dict[actual_id].logprob
            prompt_logprobs_ls.append(prompt_logprobs)
            prompt_ids_ls.append(prompt_ids)
'''
assert old in s, "block not found"
s = s.replace(old, new, 1)
old2 = '''    prompt_logprobs_ls, prompt_ids_ls = [], []
    # NOTE: logprob of first prompt token is None.
    for logprobs_dict in output.prompt_logprobs[1:]:
'''
new2 = '''    prompt_logprobs_ls, prompt_ids_ls = [], []
    # local patch (k1_probe): keep the sampled token's exact log-prob as an extra column in top-k mode
    keep_sampled = int(os.getenv("VERL_TEACHER_KEEP_SAMPLED", "0") == "1" and (num_prompt_logprobs or 0) > 0)
    actual_ids = list(output.prompt_token_ids)
    # NOTE: logprob of first prompt token is None.
    for pos, logprobs_dict in enumerate(output.prompt_logprobs[1:], start=1):
'''
assert old2 in s, "header not found"
s = s.replace(old2, new2, 1)
old3 = '''    prompt_logprobs_ls.append([0.0] * max(num_prompt_logprobs, 1))
    prompt_ids_ls.append([0] * max(num_prompt_logprobs, 1))
'''
new3 = '''    prompt_logprobs_ls.append([0.0] * (max(num_prompt_logprobs, 1) + keep_sampled))
    prompt_ids_ls.append([0] * (max(num_prompt_logprobs, 1) + keep_sampled))
'''
assert old3 in s, "dummy row not found"
s = s.replace(old3, new3, 1)
if "\nimport os\n" not in s:
    s = s.replace("import ", "import os\nimport ", 1)
open(p, "w").write(s); print("patched", p)
