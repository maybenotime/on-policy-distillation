"""How often does the per-position temperature FLIP the sign of the sampled-token advantage?
A_1 = log q_A(y) - log p_s(y)  (no transplant)   vs   A_T = log q~(y) - log p_s(y)  (A's logits at the solved T_t).
Measured on the base Qwen3-1.7B student's own DAPO rollouts, at response positions."""
import json, sys, torch, numpy as np
sys.path.insert(0, "/home/sya225/opd/envs/verl_opd/lib/python3.12/site-packages")
from verl.trainer.distillation.fsdp.tp_core import load_teacher, transplant_sequence
from transformers import AutoModelForCausalLM
dev = "cuda"
STU = "Qwen/Qwen3-1.7B"; M5 = "Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500"; Q4 = "Qwen/Qwen3-4B"
recs = [json.loads(l) for l in open("/home/sya225/opd/fork/roll_q3i17-dapo.jsonl")][:200]
stu = AutoModelForCausalLM.from_pretrained(STU, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
ma, Wa = load_teacher(M5, dev); mb, Wb = load_teacher(Q4, dev); ms, Ws = load_teacher(STU, dev)
CONF = [("A=M500 prefs, shape<-4B     (run 4)", ma, Wa, mb, Wb),
        ("A=4B   prefs, shape<-M500   (run 3)", mb, Wb, ma, Wa),
        ("A=student,    shape<-M500   (shape-only)", ms, Ws, ma, Wa)]
with torch.inference_mode():
    for name, A, WA, B, WB in CONF:
        n_all = n_flip = n_negpos = n_posneg = 0; dmag = []; amax_flip = amax_all = 0; a1s = []; ats = []
        for r in recs:
            p, s = r["prompt_ids"], r["resp_ids"]
            if len(s) < 5: continue
            ids = torch.tensor(p + s, device=dev)
            lo, hi = len(p) - 1, len(p) + len(s) - 1
            lp = torch.log_softmax(stu(ids.unsqueeze(0), use_cache=False).logits[0, lo:hi].float(), -1)
            y = ids[lo + 1:hi + 1]
            logp_s = lp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
            o = transplant_sequence(ids, A, WA, "shape", model_b=B, W_b=WB, chunk=512)
            A1 = o["logq_A"][lo:hi] - logp_s; AT = o["logq_tilde"][lo:hi] - logp_s
            is_amax = (o["top16_A"][lo:hi, 0] == y)
            flip = (torch.sign(A1) != torch.sign(AT)) & (A1.abs() > 1e-6)
            n_all += A1.numel(); n_flip += int(flip.sum()); n_negpos += int((flip & (A1 < 0)).sum()); n_posneg += int((flip & (A1 > 0)).sum())
            amax_flip += int((flip & is_amax).sum()); amax_all += int(is_amax.sum())
            dmag.append((AT.abs() - A1.abs()).cpu()); a1s.append(A1.abs().cpu()); ats.append(AT.abs().cpu())
        dm = torch.cat(dmag); a1 = torch.cat(a1s); at = torch.cat(ats)
        print(f"{name}: positions={n_all}  mean|A_1|={a1.mean():.3f}  mean|A_T|={at.mean():.3f}  "
              f"sign flips={100*n_flip/n_all:.1f}%  (neg->pos {100*n_negpos/n_all:.1f}%, pos->neg {100*n_posneg/n_all:.1f}%)  "
              f"flips among A-argmax samples={100*amax_flip/max(amax_all,1):.1f}%  |A| change p10/p50/p90={dm.quantile(.1):+.3f}/{dm.quantile(.5):+.3f}/{dm.quantile(.9):+.3f}", flush=True)
