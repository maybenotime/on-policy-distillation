# k1_probe (final-loss half): k1 loss, identical to loss_mode=k1 with use_policy_gradient=True, plus per-step probes
# computed from the teacher's top-K (+ exact sampled-token log-prob, see fsdp/k1_probe.py; needs VERL_TEACHER_KEEP_SAMPLED=1) and the student's exact entropy (actor.calculate_entropy=True):
#   probe/overlap16, probe/sampled_in_topk, probe/teacher_mass_topk, probe/teacher_entropy_topk, probe/student_entropy,
#   probe/rho (per-sequence Pearson of the two entropy curves), probe/fork_overlap (top-20% entropy positions),
#   probe/dH (|mean H_s - mean H_t|) -- the same definitions as scripts/fork_metrics.py.
from typing import Any

import torch
from tensordict import TensorDict

from verl.trainer.distillation.losses import DistillationLossSettings, register_distillation_loss
from verl.utils.metric import AggregationType, Metric
from verl.workers.config import ActorConfig, DistillationConfig
from verl.workers.utils.padding import no_padding_2_padding
from verl.trainer.distillation.token_select import apply_token_selection

MIN_LEN = 25      # fork_metrics.py rule: floor(0.2 * T) >= 5
FORK_FRAC = 0.2


def _entropy_shape_stats(hs: torch.Tensor, ht: torch.Tensor, mask: torch.Tensor):
    rhos, forks, dhs = [], [], []
    for i in range(hs.shape[0]):
        m = mask[i]
        n = int(m.sum())
        if n < MIN_LEN:
            continue
        a, b = hs[i][m].float(), ht[i][m].float()
        if a.std() == 0 or b.std() == 0:
            continue
        rhos.append(torch.corrcoef(torch.stack([a, b]))[0, 1])
        k = int(FORK_FRAC * n)
        fa, fb = torch.topk(a, k).indices, torch.topk(b, k).indices
        forks.append(torch.isin(fa, fb).float().sum() / k)
        dhs.append((a.mean() - b.mean()).abs())
    if not rhos:
        return None
    return (torch.stack(rhos).mean().item(), torch.stack(forks).mean().item(), torch.stack(dhs).mean().item(), len(rhos))


@register_distillation_loss(DistillationLossSettings(names=["k1_probe"], use_topk=True))  # type: ignore[arg-type]
def compute_k1_probe(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    assert distillation_config.distillation_loss.use_policy_gradient, "k1_probe requires use_policy_gradient=True"
    s_lp = no_padding_2_padding(model_output["log_probs"], data)               # (bsz, resp_len)
    t_lp = no_padding_2_padding(model_output["teacher_sampled_logprob"], data)
    mask = data["response_mask"]
    mask = mask.bool().to_padded_tensor(False) if mask.is_nested else mask.bool()
    assert s_lp.shape == t_lp.shape == mask.shape, (s_lp.shape, t_lp.shape, mask.shape)
    losses = s_lp - t_lp                                                       # k1 = log p_student - log p_teacher

    ov = no_padding_2_padding(model_output["overlap16"], data)
    hit = no_padding_2_padding(model_output["sampled_in_topk"], data)
    t_ent = no_padding_2_padding(model_output["teacher_entropy"], data)
    t_mass = no_padding_2_padding(model_output["teacher_mass"], data)
    idm = no_padding_2_padding(model_output["sampled_id_match"], data)
    metrics: dict[str, Any] = {
        "distillation/abs_loss": Metric(AggregationType.MEAN, losses[mask].abs().mean()),
        "probe/overlap16": ov[mask].mean().item(),
        "probe/sampled_in_topk": hit[mask].mean().item(),
        "probe/sampled_id_match": idm[mask].mean().item(),
        "probe/teacher_mass_topk": t_mass[mask].mean().item(),
        "probe/teacher_entropy_topk": t_ent[mask].mean().item(),
    }
    s_ent = None
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


@register_distillation_loss(DistillationLossSettings(names=["rk_topk_probe"], use_topk=True))  # type: ignore[arg-type]
def compute_rk_topk_probe(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Student-top-16 reverse KL (Rethinking-OPD 'only_stu' + 'student_p'), supervised path (use_policy_gradient=False):
    the per-token loss carries the PG-surrogate gradient (see fsdp/k1_probe.py) and is aggregated by verl's agg_loss.
    Same probes as k1_probe."""
    assert not distillation_config.distillation_loss.use_policy_gradient, "rk_topk_probe requires use_policy_gradient=False"
    losses = no_padding_2_padding(model_output["rk_topk_losses"], data)              # (bsz, resp_len), with grad
    mask = data["response_mask"]
    mask = mask.bool().to_padded_tensor(False) if mask.is_nested else mask.bool()
    assert losses.shape == mask.shape, (losses.shape, mask.shape)
    ov = no_padding_2_padding(model_output["overlap16"], data)
    hit = no_padding_2_padding(model_output["sampled_in_topk"], data)
    idm = no_padding_2_padding(model_output["sampled_id_match"], data)
    t_ent = no_padding_2_padding(model_output["teacher_entropy"], data)
    t_mass = no_padding_2_padding(model_output["teacher_mass"], data)
    s16 = no_padding_2_padding(model_output["stu16_in_t64"], data)
    metrics: dict[str, Any] = {
        "distillation/rk_topk_kl": losses[mask].detach().mean().item(),
        "probe/overlap16": ov[mask].mean().item(),
        "probe/sampled_in_topk": hit[mask].mean().item(),
        "probe/sampled_id_match": idm[mask].mean().item(),
        "probe/stu16_in_t64": s16[mask].mean().item(),
        "probe/teacher_mass_topk": t_mass[mask].mean().item(),
        "probe/teacher_entropy_topk": t_ent[mask].mean().item(),
    }
    if model_output.get("entropy") is not None:
        s_ent = no_padding_2_padding(model_output["entropy"], data)
        metrics["probe/student_entropy"] = s_ent[mask].mean().item()
        st = _entropy_shape_stats(s_ent, t_ent, mask)
        if st is not None:
            metrics["probe/rho"], metrics["probe/fork_overlap"], metrics["probe/dH"], metrics["probe/n_seq"] = st
    return losses, metrics


@register_distillation_loss(DistillationLossSettings(names=["k3_probe"], use_topk=True))  # type: ignore[arg-type]
def compute_k3_probe(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """verl's default recipe (loss_mode=k3, use_policy_gradient=False) + the same per-step probes as k1_probe.
    Per sampled token: u = log q - log p ; k3 = exp(u) - u - 1 (teacher term constant, student log-prob with grad),
    so the gradient is (1 - q/p) * grad log p, identical to verl's built-in k3 path."""
    assert not distillation_config.distillation_loss.use_policy_gradient, "k3_probe requires use_policy_gradient=False"
    s_lp = no_padding_2_padding(model_output["student_sampled_logprob"], data)     # with grad
    t_lp = no_padding_2_padding(model_output["teacher_sampled_logprob"], data).detach()
    mask = data["response_mask"]
    mask = mask.bool().to_padded_tensor(False) if mask.is_nested else mask.bool()
    assert s_lp.shape == t_lp.shape == mask.shape, (s_lp.shape, t_lp.shape, mask.shape)
    u = t_lp - s_lp
    losses = torch.exp(u) - u - 1.0
    ov = no_padding_2_padding(model_output["overlap16"], data)
    hit = no_padding_2_padding(model_output["sampled_in_topk"], data)
    idm = no_padding_2_padding(model_output["sampled_id_match"], data)
    t_ent = no_padding_2_padding(model_output["teacher_entropy"], data)
    t_mass = no_padding_2_padding(model_output["teacher_mass"], data)
    metrics: dict[str, Any] = {
        "distillation/k3": losses[mask].detach().mean().item(),
        "distillation/abs_k1": (s_lp - t_lp)[mask].detach().abs().mean().item(),
        "probe/overlap16": ov[mask].mean().item(),
        "probe/sampled_in_topk": hit[mask].mean().item(),
        "probe/sampled_id_match": idm[mask].mean().item(),
        "probe/teacher_mass_topk": t_mass[mask].mean().item(),
        "probe/teacher_entropy_topk": t_ent[mask].mean().item(),
    }
    if model_output.get("entropy") is not None:
        s_ent = no_padding_2_padding(model_output["entropy"], data)
        metrics["probe/student_entropy"] = s_ent[mask].mean().item()
        st = _entropy_shape_stats(s_ent, t_ent, mask)
        if st is not None:
            metrics["probe/rho"], metrics["probe/fork_overlap"], metrics["probe/dH"], metrics["probe/n_seq"] = st
    return losses, metrics


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
        "tp/overlap16_sA": pad("tp_ovA")[mask].mean().item(),
        "tp/overlap16_sB": pad("tp_ovB")[mask].mean().item(),
        "tp/overlap16_AB": pad("tp_ovAB")[mask].mean().item(),
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
