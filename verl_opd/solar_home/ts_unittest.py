import os, sys, torch
sys.path.insert(0, "/home/sya225/opd/envs/verl_opd/lib/python3.12/site-packages")
from verl.trainer.distillation import token_select as ts
torch.manual_seed(0); B, L = 3, 40
mask = torch.ones(B, L, dtype=torch.bool); mask[1, 30:] = False
hs = torch.rand(B, L); ht = hs + 0.3 * torch.rand(B, L)
loss = torch.randn(B, L)
for mode in ["mismatch", "entropy", "random", "both_committed", "mismatch_w", "mismatch_w2"]:
    os.environ["TOKEN_SELECT"] = mode; os.environ["TOKEN_FRAC"] = "0.3"; m = {}
    out = ts.apply_token_selection(loss.clone(), hs, ht, mask, m)
    assert all(isinstance(v, float) for v in m.values()), {k: type(v) for k, v in m.items()}
    ratio = out[mask].abs().mean().item() / loss[mask].abs().mean().item()
    print(f"{mode:<15} scale={ratio:.3f}  " + "  ".join(f"{k.split('/')[1]}={v:.3f}" for k, v in m.items()))
print("all metrics numeric: OK")
