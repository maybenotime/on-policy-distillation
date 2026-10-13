"""Forward Qwen3-8B (HF, bf16) on one GPU and report NaN/inf in logits + top-5 at the last position."""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
m = sys.argv[1]; dev = "cuda:0"
tok = AutoTokenizer.from_pretrained(m); model = AutoModelForCausalLM.from_pretrained(m, dtype=torch.bfloat16).to(dev).eval()
ids = tok.apply_chat_template([{"role":"user","content":"Compute 17*23. Answer: "}], add_generation_prompt=True, enable_thinking=False, return_tensors="pt", return_dict=True)["input_ids"].to(dev)
with torch.inference_mode():
    lg = model(ids, use_cache=False).logits[0].float()
print("GPU", torch.cuda.get_device_name(0), "logits nan:", torch.isnan(lg).sum().item(), "inf:", torch.isinf(lg).sum().item(), "max|logit|:", lg.abs().max().item())
lp = torch.log_softmax(lg[-1], -1); top = torch.topk(lp, 5)
print("last-pos top5:", [(tok.decode([int(i)]), round(float(v),3)) for v, i in zip(top.values, top.indices)], "entropy:", float(-(lp.exp()*lp).sum()))
