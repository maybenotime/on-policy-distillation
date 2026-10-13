"""Patch verl/trainer/distillation/losses.py: dispatch k1_probe in compute_topk_loss, import the plugin at the end."""
import sys
p = sys.argv[1]; s = open(p).read()
if "k1_probe" in s:
    print("already patched"); sys.exit(0)
anchor = "    match config.strategy:\n        # VeOmni uses FSDP2 internally"
assert anchor in s, "anchor not found"
branch = '''    # --- k1_probe plugin: k1 loss + probes from the teacher top-K (see trainer/distillation/k1_probe.py) ---
    if distillation_config.distillation_loss.loss_mode == "k1_probe":
        from verl.trainer.distillation.fsdp.k1_probe import compute_k1_probe_topk

        outputs = compute_k1_probe_topk(
            student_logits=student_logits,
            teacher_topk_log_probs=data["teacher_logprobs"],
            teacher_topk_ids=data["teacher_ids"],
            config=distillation_config,
            data_format=data_format,
            data=data,
        )
        for k, v in outputs.items():
            assert v.shape == student_logits.shape[:2], f"{k}: {v.shape} vs {student_logits.shape[:2]}"
        return outputs

'''
s = s.replace(anchor, branch + anchor, 1)
s += "\n\n# registers loss_mode 'k1_probe' (local plugin)\nfrom verl.trainer.distillation import k1_probe as _k1_probe  # noqa: E402,F401\n"
open(p, "w").write(s); print("patched", p)
