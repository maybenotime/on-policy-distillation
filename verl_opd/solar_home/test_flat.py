"""Offline check of the new 'flat' mode: same B forward as 'shape', target = B's per-sequence mean entropy."""
import torch, math, sys
sys.path.insert(0, "/home/sya225/opd/envs/verl_opd/lib/python3.12/site-packages")
from verl.trainer.distillation.fsdp.tp_core import transplant_sequence, load_teacher, entropy_from_logits
dev = "cuda"
A = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"; B = "hbx/JustRL-DeepSeek-1.5B"
ma, Wa = load_teacher(A, dev); mb, Wb = load_teacher(B, dev)
torch.manual_seed(0)
ids = torch.randint(1000, 5000, (600,), device=dev)
for mode in ("shape", "flat"):
    o = transplant_sequence(ids, ma, Wa, mode, model_b=mb, W_b=Wb, chunk=256)
    ht, tg, lt = o["H_tilde"], o["H_target"], o["logT"]
    print(f"{mode:>6}: H_A {o['H_A'].mean():.4f}  H_target mean {tg.mean():.4f} std {tg.std():.4f}  "
          f"H_tilde mean {ht.mean():.4f}  |H_tilde-H_target| max {(ht-tg).abs().max():.2e}  "
          f"logT p10/p90 {lt.quantile(0.1):.3f}/{lt.quantile(0.9):.3f}")
o_s = transplant_sequence(ids, ma, Wa, "shape", model_b=mb, W_b=Wb, chunk=256)
o_f = transplant_sequence(ids, ma, Wa, "flat",  model_b=mb, W_b=Wb, chunk=256)
print(f"top16_B identical (both read B): {torch.equal(o_s['top16_B'], o_f['top16_B'])}")
print(f"flat target is constant: {o_f['H_target'].std().item():.2e}  == shape target mean: "
      f"{abs(o_f['H_target'][0].item() - o_s['H_target'].mean().item()):.2e}")
print(f"top16_A preserved in flat (argmax unchanged by temperature): "
      f"{torch.equal(o_s['top16_A'], o_f['top16_A'])}")
