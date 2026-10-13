# k1_probe (logits-processor half). Loss stays k1 + policy gradient; the teacher's top-K is used only for probes.
# Requires VERL_TEACHER_KEEP_SAMPLED=1 (local verl patch): the teacher tensors then have K+1 columns, the last one
# being the ACTUAL sampled token and its exact log-prob (also when it is outside the top-K).
# Outputs (all (1, total_nnz) float32, aligned with student_logits positions, i.e. position i predicts token i+1):
#   teacher_sampled_logprob : exact teacher log-prob of the sampled token
#   sampled_in_topk         : 1 if the sampled token is inside the teacher's top-K (diagnostic only)
#   sampled_id_match        : 1 if the teacher's actual-token id equals input_ids[i+1] (alignment check)
#   overlap16               : |student top-16 ∩ teacher top-16| / 16
#   teacher_entropy         : entropy of the teacher's top-K-truncated distribution (tail mass lumped as one event)
#   teacher_mass            : sum of the teacher's top-K probabilities
import torch
from tensordict import TensorDict

from verl.utils.ulysses import get_ulysses_sequence_parallel_world_size
from verl.workers.config import DistillationConfig

K_OVERLAP = 16


def _chunked_log_probs_at(logits: torch.Tensor, ids: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """log_softmax(logits).gather(ids) without materializing [N, V]; keeps autograd. logits (1,N,V), ids (1,N,k)."""
    N, V = logits.shape[1], logits.shape[-1]
    fl, fi = logits.reshape(N, V), ids.reshape(N, -1)
    out = []
    for s in range(0, N, chunk):
        c = fl[s : s + chunk].float()
        out.append(torch.gather(c, -1, fi[s : s + chunk]) - torch.logsumexp(c, dim=-1, keepdim=True))
    return torch.cat(out, 0).reshape(1, N, -1)



def compute_k1_probe_topk(
    student_logits: torch.Tensor,          # (1, total_nnz, vocab)
    teacher_topk_log_probs: torch.Tensor,  # nested -> (1, total_nnz, K+1): top-K rank-sorted, then the sampled token
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
    data: TensorDict,
) -> dict[str, torch.Tensor]:
    assert get_ulysses_sequence_parallel_world_size() == 1, "k1_probe: sequence parallel not supported"
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    t_lp_all = teacher_topk_log_probs.values().unsqueeze(0).float()
    t_id_all = teacher_topk_ids.values().unsqueeze(0).long()
    K = config.distillation_loss.topk
    assert t_id_all.shape[-1] == K + 1, (
        f"k1_probe expects K+1={K + 1} teacher columns (VERL_TEACHER_KEEP_SAMPLED=1), got {t_id_all.shape[-1]}"
    )
    assert t_lp_all.shape[:2] == t_id_all.shape[:2] == student_logits.shape[:2], (t_lp_all.shape, student_logits.shape)
    t_lp, t_id = t_lp_all[..., :K], t_id_all[..., :K]              # top-K, rank-sorted
    t_sampled, t_sampled_id = t_lp_all[..., K], t_id_all[..., K]    # exact sampled-token log-prob and its id

    # sampled token at position i is input_ids[i+1]: same roll as the engine's input_ids_rmpad_rolled
    ids = data["input_ids"]
    ids = ids.values() if ids.is_nested else ids.reshape(-1)
    sampled = torch.roll(ids.long(), shifts=-1, dims=0).unsqueeze(0)   # (1, N)
    id_match = (t_sampled_id == sampled).float()                        # mismatches only at the last position of each sequence
    in_topk = (t_id == sampled.unsqueeze(-1)).any(-1).float()

    k = min(K_OVERLAP, K)
    s_top = torch.topk(student_logits, k=k, dim=-1).indices             # top-k of logits == top-k of log-probs
    overlap = (t_id[..., :k].unsqueeze(-1) == s_top.unsqueeze(-2)).any(-1).sum(-1).float() / k

    p = t_lp.exp()
    mass = p.sum(-1)
    tail = (1.0 - mass).clamp_min(1e-8)
    t_ent = -(p * t_lp).sum(-1) - tail * tail.log()

    out = {}
    if config.distillation_loss.loss_mode == "k3_probe":
        # verl's default k3 (supervised backprop) needs the student's log-prob of the sampled token WITH grad;
        # in the top-k path the engine skips model_output["log_probs"] (distillation_only), so compute it here.
        out["student_sampled_logprob"] = _chunked_log_probs_at(student_logits, sampled.unsqueeze(-1)).squeeze(-1)
    if config.distillation_loss.loss_mode == "rk_topk_probe":
        # Rethinking-OPD default (LOG_PROB_TOP_K=16, only_stu, student_p): reverse KL on the STUDENT's top-16,
        # weights = student probs renormalised over the 16 (detached), teacher log-prob looked up in its top-K
        # (floor = log_prob_min_clamp when a student candidate is outside the teacher top-K).
        # Loss value = truncated KL (for logging/clamp); gradient = the PG surrogate sum_k [w_k (lp_k - lq_k)] * grad lp_k,
        # which equals their on-policy top-k policy gradient (ratio = 1, no clipping active).
        s_lp16 = _chunked_log_probs_at(student_logits, s_top)                       # (1, N, k) with grad
        m16 = s_top.unsqueeze(-1) == t_id.unsqueeze(-2)                              # (1, N, k, K)
        found = m16.any(-1)
        floor = config.distillation_loss.log_prob_min_clamp
        floor = -10.0 if floor is None else float(floor)
        lq16 = torch.where(found, (m16 * t_lp.unsqueeze(-2)).sum(-1), torch.full_like(s_lp16, floor)).detach()
        w = torch.softmax(s_lp16.detach(), dim=-1)
        diff = (s_lp16 - lq16).detach()
        kl16 = (w * diff).sum(-1)                                                     # value: truncated reverse KL
        surrogate = ((w * diff) * s_lp16).sum(-1)                                     # grad: PG surrogate
        out["rk_topk_losses"] = kl16 + (surrogate - surrogate.detach())
        out["stu16_in_t64"] = found.float().mean(-1)
    if config.distillation_loss.loss_mode == "k1_transplant":
        out.update(compute_transplant(student_logits, data))
    out.update({
        "teacher_sampled_logprob": t_sampled,
        "sampled_in_topk": in_topk,
        "sampled_id_match": id_match,
        "overlap16": overlap,
        "teacher_entropy": t_ent,
        "teacher_mass": mass,
    })
    return out


# ---- entropy-shape transplant (loss_mode = k1_transplant) -------------------------------------------------------------
# Runs two HF teachers (A = preferences, B = uncertainty profile) on every sequence of the micro-batch, full vocabulary,
# and replaces the teacher term of k1 by log q~_t(y_t) where q~_t = softmax(z^A_t / T_t) with H(q~_t) = H(q^B_t).
# Env: TRANSPLANT_MODE (shape|flat|student|uniform|none), TRANSPLANT_LOGITS_MODEL (A), TRANSPLANT_SHAPE_MODEL (B),
#      TRANSPLANT_GLOBAL_T (uniform mode), TRANSPLANT_CHUNK (positions per logits chunk, default 1024).
import os as _os

_TP_CACHE: dict = {}
_TP_KEYS = ("logq_tilde", "logq_A", "H_A", "H_target", "H_tilde", "logT")
_TP_ID_KEYS = ("top16_A", "top16_B")


def _tp_seq_key(seq: torch.Tensor):
    return (int(seq.shape[0]), hash(tuple(seq[:48].tolist())), hash(tuple(seq[-48:].tolist())))


@torch.no_grad()
def compute_transplant(student_logits: torch.Tensor, data: TensorDict) -> dict[str, torch.Tensor]:
    from verl.trainer.distillation.fsdp.tp_core import load_teacher, transplant_sequence

    mode = _os.getenv("TRANSPLANT_MODE", "shape")
    a_path = _os.environ["TRANSPLANT_LOGITS_MODEL"]
    b_path = _os.getenv("TRANSPLANT_SHAPE_MODEL", "")
    gT = float(_os.getenv("TRANSPLANT_GLOBAL_T", "1.0"))
    chunk = int(_os.getenv("TRANSPLANT_CHUNK", "1024"))
    dev = student_logits.device
    ids_n = data["input_ids"]
    assert ids_n.is_nested, "k1_transplant needs the nested (rmpad) data format"
    ids = ids_n.values().long()
    offs = ids_n.offsets().tolist()
    N = ids.shape[0]
    assert student_logits.shape[1] == N, (student_logits.shape, N)
    m_a, W_a = load_teacher(a_path, dev)
    m_b, W_b = load_teacher(b_path, dev) if mode in ("shape", "flat") else (None, None)
    out = {k: torch.empty(N, device=dev, dtype=torch.float32) for k in _TP_KEYS}
    ids_out = {k: torch.zeros(N, 16, device=dev, dtype=torch.long) for k in _TP_ID_KEYS}
    n_cached = 0
    for s in range(len(offs) - 1):
        a, b = int(offs[s]), int(offs[s + 1])
        if b <= a:
            continue
        seq = ids[a:b]
        key = _tp_seq_key(seq)
        if key in _TP_CACHE:
            for k in _TP_KEYS:
                out[k][a:b] = _TP_CACHE[key][k].to(dev)
            for k in _TP_ID_KEYS:
                ids_out[k][a:b] = _TP_CACHE[key][k].to(dev)
            n_cached += 1
            continue
        r = transplant_sequence(seq, m_a, W_a, mode, m_b, W_b,
                                student_logits=student_logits[0, a:b].detach() if mode == "student" else None,
                                global_T=gT, chunk=chunk)
        for k in _TP_KEYS:
            out[k][a:b] = r[k]
        for k in _TP_ID_KEYS:
            ids_out[k][a:b] = r[k]
        if len(_TP_CACHE) > 2048:
            _TP_CACHE.clear()
        _TP_CACHE[key] = {k: r[k].to("cpu") for k in _TP_KEYS + _TP_ID_KEYS}
    # overlap@16 of the student's top-16 with teacher A's and with the shape source B's (mass-free, ids only)
    s_top = torch.topk(student_logits[0], k=16, dim=-1).indices                    # (N, 16)
    ov = {}
    for k, ids16 in ids_out.items():
        ov["ovA" if k == "top16_A" else "ovB"] = (s_top.unsqueeze(-1) == ids16.unsqueeze(-2)).any(-1).sum(-1).float() / 16.0
    ov["ovAB"] = (ids_out["top16_A"].unsqueeze(-1) == ids_out["top16_B"].unsqueeze(-2)).any(-1).sum(-1).float() / 16.0
    res = {f"tp_{k}": v.unsqueeze(0) for k, v in out.items()}
    res.update({f"tp_{k}": v.unsqueeze(0) for k, v in ov.items()})
    res["tp_unmet"] = ((out["H_tilde"] - out["H_target"]).abs() > 0.01).float().unsqueeze(0)
    res["tp_cached"] = torch.full((1, N), float(n_cached), device=dev)
    return res
