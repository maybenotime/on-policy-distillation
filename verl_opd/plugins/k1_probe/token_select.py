# Token selection for the k1_probe loss: train only on the tokens a per-sequence criterion picks.
#
#   TOKEN_SELECT = none | mismatch | entropy | random | both_committed | mismatch_w | mismatch_w2   (default none)
#   TOKEN_WCAP   = cap on a soft weight after per-sequence normalisation to mean 1   (default 5)
#   TOKEN_FRAC   = fraction of each sequence's response tokens to keep     (default 0.3; ignored by both_committed)
#   TOKEN_SEED   = seed for the random arm                                 (default 0)
#
# mismatch       : rank the student's and the teacher's per-position entropies within the sequence; a token is
#                  "misplaced" by |rank_s - rank_t| / n; keep the TOKEN_FRAC most misplaced.  This is the token-level
#                  reading of rho: rho is the correlation of the two entropy curves, and it is low exactly where their
#                  orderings disagree.
# entropy        : keep the TOKEN_FRAC highest-student-entropy tokens (the Table-4 / TIP-style baseline).
# random         : keep a random TOKEN_FRAC (budget control).
# both_committed : drop tokens where BOTH entropies are below their own sequence median; keep everything else.
# mismatch_w     : SOFT weight w_t = |rank_s - rank_t|, normalised to mean 1 within the sequence (no token is dropped).
# mismatch_w2    : w_t = (rank_s - rank_t)^2, likewise normalised.  Spearman rho = 1 - 6*sum(d^2)/(n(n^2-1)), so this
#                  weights each token by exactly its share of the shape gap (1 - rho).
# Metrics must be numeric only: verl reduces them across ranks with numpy, so no mode string is logged.
# Soft modes need no rescaling: mean weight 1 keeps the token-mean scale of full training.
#
# The kept tokens' losses are rescaled by 1/(kept fraction) so that, under token-mean aggregation over the full
# response mask (the denominator is fixed upstream), the gradient is the token-mean over the KEPT tokens — i.e. the
# same per-token scale as full training, not a 0.3x learning rate.  Dropped tokens get loss 0, hence advantage 0
# and no gradient under the policy-gradient path.
import os
import torch


def _ranks(x: torch.Tensor) -> torch.Tensor:
    """0..n-1 rank of each element (ascending); stable for ties."""
    return torch.argsort(torch.argsort(x, stable=True), stable=True).float()


@torch.no_grad()
def select_tokens(s_ent: torch.Tensor, t_ent: torch.Tensor, mask: torch.Tensor, step_seed: int = 0):
    """s_ent, t_ent, mask: (bsz, resp_len) padded.  Returns (sel_mask (bsz, resp_len) bool, stats dict)."""
    mode = os.getenv("TOKEN_SELECT", "none").lower()
    frac = float(os.getenv("TOKEN_FRAC", "0.3"))
    if mode == "none":
        return mask.clone(), {}
    sel = torch.zeros_like(mask)
    gen = None
    if mode == "random":
        gen = torch.Generator(device="cpu").manual_seed(int(os.getenv("TOKEN_SEED", "0")) + step_seed)
    for i in range(mask.shape[0]):
        m = mask[i]
        idx = torch.nonzero(m, as_tuple=False).squeeze(-1)
        n = idx.numel()
        if n == 0:
            continue
        hs, ht = s_ent[i][idx].float(), t_ent[i][idx].float()
        if mode == "both_committed":
            keep = ~((hs < hs.median()) & (ht < ht.median()))
            sel[i, idx[keep]] = True
            continue
        k = max(1, int(round(frac * n)))
        if mode == "mismatch":
            score = (_ranks(hs) - _ranks(ht)).abs()
        elif mode == "entropy":
            score = hs
        elif mode == "random":
            score = torch.rand(n, generator=gen).to(hs.device)
        else:
            raise ValueError(f"TOKEN_SELECT={mode}")
        top = torch.topk(score, k).indices
        sel[i, idx[top]] = True
    kept = sel[mask].float().mean().item()
    return sel, {"select/frac_kept": kept}


@torch.no_grad()
def soft_weights(s_ent: torch.Tensor, t_ent: torch.Tensor, mask: torch.Tensor, power: int):
    """Per-token weight |rank_s - rank_t|^power, normalised to mean 1 per sequence, capped at TOKEN_WCAP."""
    cap = float(os.getenv("TOKEN_WCAP", "5"))
    w = torch.zeros(mask.shape, device=mask.device, dtype=torch.float32)
    for i in range(mask.shape[0]):
        idx = torch.nonzero(mask[i], as_tuple=False).squeeze(-1)
        n = idx.numel()
        if n == 0:
            continue
        d = (_ranks(s_ent[i][idx].float()) - _ranks(t_ent[i][idx].float())).abs() ** power
        d = d / d.mean().clamp_min(1e-8)
        w[i, idx] = d.clamp(max=cap)
    return w


def apply_token_selection(losses: torch.Tensor, s_ent, t_ent, mask: torch.Tensor, metrics: dict, step_seed: int = 0):
    """Zero the loss outside the selected tokens and rescale the kept ones so the token-mean is over kept tokens."""
    mode = os.getenv("TOKEN_SELECT", "none").lower()
    if mode == "none" or s_ent is None:
        return losses
    if mode in ("mismatch_w", "mismatch_w2"):
        w = soft_weights(s_ent, t_ent, mask, power=1 if mode == "mismatch_w" else 2)
        wv = w[mask]
        metrics["select/w_mean"] = wv.mean().item()
        metrics["select/w_max"] = wv.max().item()
        metrics["select/eff_frac"] = (wv.sum() ** 2 / (wv.numel() * (wv ** 2).sum())).item()   # (sum w)^2 / (n sum w^2)
        a = losses.detach().abs()[mask]
        metrics["select/abs_k1_weighted"] = ((a * wv).sum() / wv.sum()).item()
        metrics["select/abs_k1_unweighted"] = a.mean().item()
        return losses * w.to(losses.dtype)
    sel, st = select_tokens(s_ent, t_ent, mask, step_seed)
    kept = sel[mask].float().mean().clamp_min(1e-6)
    a = losses.detach().abs()
    metrics.update(st)
    metrics["select/abs_k1_kept"] = a[sel].mean().item() if sel.any() else 0.0
    dropped = mask & ~sel
    metrics["select/abs_k1_dropped"] = a[dropped].mean().item() if dropped.any() else 0.0
    return losses * sel.to(losses.dtype) / kept
