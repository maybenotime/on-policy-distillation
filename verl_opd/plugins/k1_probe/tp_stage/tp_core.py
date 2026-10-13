# Entropy-shape transplant core (pure torch; used by the k1_transplant loss mode and by offline verification).
#
# Synthetic teacher  A ⊕ shape(B):  per position t, find T_t such that H(softmax(z^A_t / T_t)) = h*_t, where z^A_t are
# teacher A's full-vocabulary logits and h*_t is the target entropy (teacher B's entropy at the same position of the same
# student rollout, or the student's own entropy, or a constant for the uniform-sharpening control).  Dividing logits by a
# positive scalar keeps A's within-position token ranking (argmax / top-k sets) and only changes how peaked it is, so the
# result carries A's preferences with B's uncertainty profile.
#
# Solver: Newton in log T with a bracket (bisection fallback).  dH/dlogT = Var_{p_T}(z) / T^2 > 0.
import math, os
import torch

_MODELS: dict = {}


def load_teacher(path: str, device):
    """HF causal LM (bf16, flash-attn) + fp32 copy of lm_head weight, cached per process."""
    key = (path, str(device))
    if key not in _MODELS:
        from transformers import AutoModelForCausalLM
        attn = os.getenv("TRANSPLANT_ATTN", "flash_attention_2")
        m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation=attn)  # dtype= (torch_dtype is ignored by this transformers -> fp32)
        m = m.to(device=device, dtype=torch.bfloat16).eval()     # force bf16 whatever the worker's default dtype is
        for p in m.parameters():
            p.requires_grad_(False)
        W = m.lm_head.weight.detach().to(device=device, dtype=torch.float32)
        print(f"[tp_core] loaded {path} on {device}: param dtype {next(m.parameters()).dtype}, attn {m.config._attn_implementation}", flush=True)
        _MODELS[key] = (m, W)
    return _MODELS[key]


@torch.no_grad()
def hidden_states(model, ids: torch.Tensor) -> torch.Tensor:
    """ids (L,) long -> last hidden state (L, d) in bf16 (no logits materialised)."""
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model.model(input_ids=ids.unsqueeze(0), use_cache=False)
    return out.last_hidden_state[0].to(torch.bfloat16)


def entropy_from_logits(z: torch.Tensor) -> torch.Tensor:
    lp = torch.log_softmax(z.float(), -1)
    return -(lp.exp() * lp).sum(-1)


@torch.no_grad()
def solve_temperature(z: torch.Tensor, h_target: torch.Tensor, iters: int = 10, lo: float = -3.5, hi: float = 3.5):
    """z (C, V) fp32 logits, h_target (C,) in nats.  Returns logT (C,), log_softmax(z/T) (C, V), H_tilde (C,)."""
    zc = z - z.max(-1, keepdim=True).values
    logT = torch.zeros_like(h_target)
    lo_b = torch.full_like(h_target, lo)
    hi_b = torch.full_like(h_target, hi)
    for _ in range(iters):
        T = logT.exp().unsqueeze(-1)
        lp = torch.log_softmax(zc / T, -1)
        p = lp.exp()
        H = -(p * lp).sum(-1)
        Ez = (p * zc).sum(-1)
        var = ((p * zc * zc).sum(-1) - Ez * Ez).clamp_min(0.0)
        d = (var / (T.squeeze(-1) ** 2)).clamp_min(1e-6)          # dH / dlogT
        too_high = H > h_target                                    # H increases with T -> T too large
        hi_b = torch.where(too_high, logT, hi_b)
        lo_b = torch.where(too_high, lo_b, logT)
        new = logT + (h_target - H) / d
        bad = (~torch.isfinite(new)) | (new <= lo_b) | (new >= hi_b)
        logT = torch.where(bad, 0.5 * (lo_b + hi_b), new)
    T = logT.exp().unsqueeze(-1)
    lp = torch.log_softmax(zc / T, -1)
    H = -(lp.exp() * lp).sum(-1)
    return logT, lp, H


@torch.no_grad()
def transplant_sequence(ids: torch.Tensor, model_a, W_a, mode: str, model_b=None, W_b=None,
                        student_logits: torch.Tensor | None = None, global_T: float = 1.0, chunk: int = 1024,
                        iters: int = 10):
    """One sequence (prompt+response token ids, (L,)).  Position j predicts ids[j+1].
    mode: 'shape'   -> target entropy = teacher B's entropy at the same position
          'flat'    -> target entropy = the MEAN of B's entropy over this sequence, recomputed every step.
                       Matched control for 'shape': same model B forward on the same rollout, same absolute
                       entropy level, only the within-sequence profile removed.  (Unlike 'uniform', the level
                       is re-anchored to B every step instead of a temperature calibrated once offline.)
          'student' -> target entropy = student's entropy (student_logits (L, V) required)
          'uniform' -> fixed global temperature global_T (no solve).  NOTE: with A = the student's own base and
                       global_T > 1 this has no entropy anchor at all — the per-token advantage is
                       (1/T - 1) * z_y + const, which monotonically penalises high-logit tokens and drives the
                       entropy up without bound.  Use 'flat' instead.
          'none'    -> T = 1 (returns A's own log-probs; sanity mode)
    Returns dict of (L,) fp32 tensors: logq_tilde, logq_A, H_A, H_target, H_tilde, logT."""
    L = ids.shape[0]
    dev = ids.device
    V = W_a.shape[0]
    y = torch.roll(ids, -1)                                       # next token; last position is garbage (masked later)
    hs_a = hidden_states(model_a, ids)
    hs_b = hidden_states(model_b, ids) if mode in ("shape", "flat") else None
    out = {k: torch.empty(L, device=dev, dtype=torch.float32) for k in ("logq_tilde", "logq_A", "H_A", "H_target", "H_tilde", "logT")}
    out["top16_A"] = torch.empty(L, 16, device=dev, dtype=torch.long)
    out["top16_B"] = torch.empty(L, 16, device=dev, dtype=torch.long)
    hmax = math.log(V) - 1e-3
    HB = top16B = None
    if hs_b is not None:                                          # one pass over B, reused below
        HB = torch.empty(L, device=dev, dtype=torch.float32)
        top16B = torch.empty(L, 16, device=dev, dtype=torch.long)
        for c in range(0, L, chunk):
            e = min(L, c + chunk)
            zB = hs_b[c:e].float() @ W_b.T
            HB[c:e] = entropy_from_logits(zB)
            top16B[c:e] = torch.topk(zB, 16, dim=-1).indices
            del zB
    HB_mean = HB.mean() if HB is not None else None               # B's mean entropy on THIS rollout
    for c in range(0, L, chunk):
        e = min(L, c + chunk)
        zA = hs_a[c:e].float() @ W_a.T                            # (C, V) fp32
        lpA = torch.log_softmax(zA, -1)
        HA = -(lpA.exp() * lpA).sum(-1)
        out["top16_A"][c:e] = torch.topk(zA, 16, dim=-1).indices
        out["top16_B"][c:e] = out["top16_A"][c:e]          # overridden below in shape mode
        if mode == "shape":
            tgt = HB[c:e]
            out["top16_B"][c:e] = top16B[c:e]
        elif mode == "flat":
            tgt = HB_mean.expand(e - c).clone()
            out["top16_B"][c:e] = top16B[c:e]
        elif mode == "student":
            tgt = entropy_from_logits(student_logits[c:e])
        elif mode == "uniform":
            tgt = None
        elif mode == "none":
            tgt = HA
        else:
            raise ValueError(mode)
        if mode == "uniform":
            lT = torch.full((e - c,), math.log(global_T), device=dev)
            lp = torch.log_softmax(zA / global_T, -1)
            Ht = -(lp.exp() * lp).sum(-1)
            tgt = Ht
        elif mode == "none":
            lT = torch.zeros(e - c, device=dev); lp = lpA; Ht = HA
        else:
            tgt = tgt.clamp(1e-3, hmax)
            lT, lp, Ht = solve_temperature(zA, tgt, iters=iters)
        yy = y[c:e].unsqueeze(-1)
        out["logq_tilde"][c:e] = lp.gather(-1, yy).squeeze(-1)
        out["logq_A"][c:e] = lpA.gather(-1, yy).squeeze(-1)
        out["H_A"][c:e] = HA
        out["H_target"][c:e] = tgt
        out["H_tilde"][c:e] = Ht
        out["logT"][c:e] = lT
        del zA, lpA, lp
    return out


if __name__ == "__main__":
    # self-test of the solver on random logits
    torch.manual_seed(0)
    z = torch.randn(64, 151936) * 3
    z[:, :5] += 8
    H0 = entropy_from_logits(z)
    tgt = torch.rand(64) * 4 + 0.05
    logT, lp, H = solve_temperature(z, tgt)
    err = (H - tgt).abs()
    print(f"H0 mean {H0.mean():.3f}; target mean {tgt.mean():.3f}; max |H-target| {err.max():.2e}; "
          f"unmet(>0.01) {(err > 0.01).float().mean():.3f}; logT range [{logT.min():.2f}, {logT.max():.2f}]")
    assert (err < 1e-3).all(), "solver failed"
    # argmax preserved
    assert (lp.argmax(-1) == z.argmax(-1)).all()
    print("solver OK")
