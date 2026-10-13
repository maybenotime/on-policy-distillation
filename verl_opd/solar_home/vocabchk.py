import json, glob, os
HF=os.environ.get("HF_HOME","/localscratch/sya225/hf_cache")
want={"Qwen3-1.7B":"models--Qwen--Qwen3-1.7B",
      "Qwen3-4B-Math-Step500":"models--Keven16--Qwen3-4B-Non-Thinking-RL-Math-Step500",
      "DS-R1-Distill-1.5B":"models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B",
      "JustRL-1.5B":"models--hbx--JustRL-DeepSeek-1.5B",
      "R1-Distill-7B":"models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B",
      "Qwen3-4B":"models--Qwen--Qwen3-4B"}
for name,d in want.items():
    cfgs=glob.glob(f"{HF}/hub/{d}/snapshots/*/config.json")
    if not cfgs: print(f"{name:<24} (not cached here)"); continue
    c=json.load(open(cfgs[0]))
    print(f"{name:<24} vocab={c.get('vocab_size')}  hidden={c.get('hidden_size')}  layers={c.get('num_hidden_layers')}  tie={c.get('tie_word_embeddings')}")
