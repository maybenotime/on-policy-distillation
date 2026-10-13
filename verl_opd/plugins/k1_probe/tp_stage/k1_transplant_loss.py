

@register_distillation_loss(DistillationLossSettings(names=["k1_transplant"], use_topk=True))  # type: ignore[arg-type]
def compute_k1_transplant(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """k1 + PG against the synthetic teacher A ⊕ shape(B) (fsdp/k1_probe.py::compute_transplant): the teacher term of k1 is
    log q~(y) with q~ = softmax(z_A / T_t), H(q~_t) = target entropy (teacher B / student / constant).
    probe/overlap16 and probe/sampled_in_topk still come from the vLLM teacher (= A; the transplant does not change A's top-k),
    probe/rho, probe/fork_overlap, probe/dH, probe/teacher_entropy_topk refer to the synthetic teacher used in the loss,
    tp/* are the transplant diagnostics (temperatures, unmet targets, ρ between all pairs)."""
    assert distillation_config.distillation_loss.use_policy_gradient, "k1_transplant requires use_policy_gradient=True"
    pad = lambda k: no_padding_2_padding(model_output[k], data)  # noqa: E731
    s_lp = pad("log_probs")
    q_t, q_a = pad("tp_logq_tilde"), pad("tp_logq_A")
    mask = data["response_mask"]
    mask = mask.bool().to_padded_tensor(False) if mask.is_nested else mask.bool()
    assert s_lp.shape == q_t.shape == mask.shape, (s_lp.shape, q_t.shape, mask.shape)
    losses = s_lp - q_t                                                       # k1 vs the synthetic teacher
    hA, hB, hT = pad("tp_H_A"), pad("tp_H_target"), pad("tp_H_tilde")
    logT, unmet = pad("tp_logT"), pad("tp_unmet")
    ov, hit, t_ent_vllm = pad("overlap16"), pad("sampled_in_topk"), pad("teacher_entropy")
    lt = logT[mask]
    metrics: dict[str, Any] = {
        "distillation/abs_loss": Metric(AggregationType.MEAN, losses[mask].abs().mean()),
        "tp/abs_k1_A": (s_lp - q_a)[mask].abs().mean().item(),
        "tp/logq_tilde_minus_A": (q_t - q_a)[mask].mean().item(),
        "tp/H_A": hA[mask].mean().item(),
        "tp/H_target": hB[mask].mean().item(),
        "tp/H_tilde": hT[mask].mean().item(),
        "tp/T_mean": lt.exp().mean().item(),
        "tp/logT_mean": lt.mean().item(),
        "tp/logT_p10": torch.quantile(lt, 0.1).item(),
        "tp/logT_p90": torch.quantile(lt, 0.9).item(),
        "tp/unmet_frac": unmet[mask].mean().item(),
        "tp/T_at_bound_frac": (lt.abs() > 3.4).float().mean().item(),
        "tp/cached_seqs": model_output["tp_cached"].max().item(),
        "probe/overlap16": ov[mask].mean().item(),
        "probe/sampled_in_topk": hit[mask].mean().item(),
        "probe/teacher_entropy_topk": hT[mask].mean().item(),
        "probe/teacher_entropy_vllm_topk": t_ent_vllm[mask].mean().item(),
    }
    if model_output.get("entropy") is not None:
        s_ent = pad("entropy")
        metrics["probe/student_entropy"] = s_ent[mask].mean().item()
        pairs = (("rho", s_ent, hT), ("rho_s_A", s_ent, hA), ("rho_s_B", s_ent, hB), ("rho_tilde_B", hT, hB), ("rho_A_B", hA, hB))
        for name, x, y in pairs:
            st = _entropy_shape_stats(x, y, mask)
            if st is None:
                continue
            if name == "rho":
                metrics["probe/rho"], metrics["probe/fork_overlap"], metrics["probe/dH"], metrics["probe/n_seq"] = st
            else:
                metrics[f"tp/{name}"], metrics[f"tp/fork_{name[4:]}"], metrics[f"tp/dH_{name[4:]}"] = st[0], st[1], st[2]
    return losses, metrics
