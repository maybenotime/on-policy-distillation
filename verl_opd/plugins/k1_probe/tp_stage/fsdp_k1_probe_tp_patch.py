# ---- entropy-shape transplant (loss_mode = k1_transplant) -------------------------------------------------------------
# Runs two HF teachers (A = preferences, B = uncertainty profile) on every sequence of the micro-batch, full vocabulary,
# and replaces the teacher term of k1 by log q~_t(y_t) where q~_t = softmax(z^A_t / T_t) with H(q~_t) = H(q^B_t).
# Env: TRANSPLANT_MODE (shape|student|uniform|none), TRANSPLANT_LOGITS_MODEL (A), TRANSPLANT_SHAPE_MODEL (B),
#      TRANSPLANT_GLOBAL_T (uniform mode), TRANSPLANT_CHUNK (positions per logits chunk, default 1024).
import os as _os

_TP_CACHE: dict = {}
_TP_KEYS = ("logq_tilde", "logq_A", "H_A", "H_target", "H_tilde", "logT")


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
    m_b, W_b = load_teacher(b_path, dev) if mode == "shape" else (None, None)
    out = {k: torch.empty(N, device=dev, dtype=torch.float32) for k in _TP_KEYS}
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
            n_cached += 1
            continue
        r = transplant_sequence(seq, m_a, W_a, mode, m_b, W_b,
                                student_logits=student_logits[0, a:b].detach() if mode == "student" else None,
                                global_T=gT, chunk=chunk)
        for k in _TP_KEYS:
            out[k][a:b] = r[k]
        if len(_TP_CACHE) > 2048:
            _TP_CACHE.clear()
        _TP_CACHE[key] = {k: r[k].to("cpu") for k in _TP_KEYS}
    res = {f"tp_{k}": v.unsqueeze(0) for k, v in out.items()}
    res["tp_unmet"] = ((out["H_tilde"] - out["H_target"]).abs() > 0.01).float().unsqueeze(0)
    res["tp_cached"] = torch.full((1, N), float(n_cached), device=dev)
    return res
