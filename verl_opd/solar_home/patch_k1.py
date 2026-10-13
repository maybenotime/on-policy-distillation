import re
P="/home/sya225/opd/plugins/k1_probe/k1_probe.py"
s=open(P).read()
assert "token_select" not in s, "already patched"
s=s.replace("from verl.workers.utils.padding import no_padding_2_padding",
            "from verl.workers.utils.padding import no_padding_2_padding\nfrom verl.trainer.distillation.token_select import apply_token_selection",1)
old="""    if model_output.get("entropy") is not None:
        s_ent = no_padding_2_padding(model_output["entropy"], data)
        metrics["probe/student_entropy"] = s_ent[mask].mean().item()
        st = _entropy_shape_stats(s_ent, t_ent, mask)
        if st is not None:
            metrics["probe/rho"], metrics["probe/fork_overlap"], metrics["probe/dH"], metrics["probe/n_seq"] = st
    return losses, metrics


@register_distillation_loss(DistillationLossSettings(names=["rk_topk_probe"], use_topk=True))"""
new="""    s_ent = None
    if model_output.get("entropy") is not None:
        s_ent = no_padding_2_padding(model_output["entropy"], data)
        metrics["probe/student_entropy"] = s_ent[mask].mean().item()
        st = _entropy_shape_stats(s_ent, t_ent, mask)
        if st is not None:
            metrics["probe/rho"], metrics["probe/fork_overlap"], metrics["probe/dH"], metrics["probe/n_seq"] = st
    # TOKEN_SELECT=mismatch|entropy|random|both_committed restricts the loss to a subset of tokens (token_select.py);
    # default 'none' leaves this a no-op.  Applied to the k1_probe path only.
    losses = apply_token_selection(losses, s_ent, t_ent, mask, metrics)
    return losses, metrics


@register_distillation_loss(DistillationLossSettings(names=["rk_topk_probe"], use_topk=True))"""
assert s.count(old)==1, "anchor not unique"
s=s.replace(old,new)
open(P,"w").write(s); print("k1_probe.py patched")
