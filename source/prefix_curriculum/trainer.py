"""GKDTrainer subclass that applies the prefix-curriculum loss mask.

SP-FPOD (forward): keep only the first ⌊p·T⌋ response tokens unmasked in the
GKD reverse-KL loss. Achieved by tightening `labels` — positions beyond the
supervised prefix are set to -100, which the upstream JSD loss in ms-swift
already ignores (see `mask = labels != -100` in
`swift/rlhf_trainers/gkd_trainer.py`).

TP-BPOD (backward): inject a teacher prefix into the rollout so the student
generates conditioned on it. The full sequence becomes
`[prompt, teacher_prefix, student_suffix]`; labels mark everything but the
student suffix as -100. Requires `--vllm_mode colocate`: the trainer calls
`self.engine.infer` directly with a `RolloutInferRequest` whose `data_dict`
carries `prompt_token_ids`; the plugin's Template.encode patch unpacks it
and skips chat-template encoding. vLLM then continues from those raw tokens.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import torch
from swift.rlhf_trainers.gkd_trainer import DataSource, GKDTrainer
from trl import SFTTrainer as HFSFTTrainer

from .callback import PrefixPacerCallback
from .pacer import PrefixPacer

logger = logging.getLogger(__name__)

_RESPONSE_MIN_LEN = 8


class PrefixCurriculumGKDTrainer(GKDTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefix_pacer = PrefixPacer.from_env()
        self.add_callback(PrefixPacerCallback(self.prefix_pacer))
        logger.info('[PrefixCurriculum] initialized %r', self.prefix_pacer)

    # ------------------------------------------------------------------
    # Forward direction (SP-FPOD): tighten labels at right end
    # ------------------------------------------------------------------

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.prefix_pacer.direction == 'forward':
            labels = inputs.get('labels')
            ratio = self.prefix_pacer.current_ratio()
            if isinstance(labels, torch.Tensor) and ratio < 1.0:
                inputs['labels'] = _apply_forward_prefix_mask(labels, ratio)
        elif self.prefix_pacer.direction == 'entropy_topk':
            labels = inputs.get('labels')
            frac = self.prefix_pacer.current_ratio()
            if isinstance(labels, torch.Tensor) and 0.0 < frac < 1.0:
                inputs['labels'] = _apply_entropy_topk_mask(
                    model, inputs, labels, frac,
                )
        elif self.prefix_pacer.direction == 'entropy_reweight':
            # Compute top-K entropy weights now (extra no-grad student forward),
            # stash for the overridden generalized_jsd_loss below.
            labels = inputs.get('labels')
            topk = int(self.prefix_pacer.current_ratio())
            if isinstance(labels, torch.Tensor) and topk >= 2:
                self._reweight_token_weights = _compute_top_k_entropy_weights(
                    model, inputs, labels, topk=topk,
                )
            else:
                self._reweight_token_weights = None
        return super().compute_loss(
            model, inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

    def generalized_jsd_loss(self, student_logits, teacher_logits=None,
                              labels=None, beta=0.5, temperature=1.0,
                              chunk_size=512, topk=None,
                              teacher_topk_logprobs=None, teacher_topk_indices=None):
        """Override the staticmethod to inject entropy-reweighted reduction
        when direction='entropy_reweight'. Otherwise fall back to the parent."""
        if (self.prefix_pacer.direction == 'entropy_reweight'
                and getattr(self, '_reweight_token_weights', None) is not None
                and labels is not None):
            return _weighted_jsd_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=labels,
                weights=self._reweight_token_weights,
                beta=beta,
                temperature=temperature,
                topk=topk,
                teacher_topk_logprobs=teacher_topk_logprobs,
                teacher_topk_indices=teacher_topk_indices,
            )
        return GKDTrainer.generalized_jsd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            beta=beta,
            temperature=temperature,
            chunk_size=chunk_size,
            topk=topk,
            teacher_topk_logprobs=teacher_topk_logprobs,
            teacher_topk_indices=teacher_topk_indices,
        )

    # ------------------------------------------------------------------
    # Backward direction (TP-BPOD): custom training step
    # ------------------------------------------------------------------

    def training_step(self, model, inputs, num_items_in_batch=None):
        if self.prefix_pacer.direction == 'backward':
            return self._tp_bpod_training_step(model, inputs, num_items_in_batch)
        return super().training_step(model, inputs, num_items_in_batch)

    def _tp_bpod_training_step(self, model, inputs, num_items_in_batch):
        # Colocate-only: requires vLLM engine in the same process so the
        # plugin's Template.encode patch is in scope.
        if getattr(self, 'vllm_mode', None) != 'colocate':
            raise RuntimeError(
                'TP-BPOD requires --vllm_mode colocate (got %r). '
                'Server-mode rollout spawns separate worker processes that do '
                'not inherit our Template.encode monkey-patch.'
                % getattr(self, 'vllm_mode', None)
            )

        # 1. Encode prompts only.
        encoded_inputs = self._prepare_batch_inputs(inputs, encode_prompt_only=True)

        # 2. For each sample, build `prompt_ids + teacher_prefix_ids[:Q_i]`.
        q = self.prefix_pacer.current_ratio()
        prompt_with_prefix_ids: List[List[int]] = []
        prompt_plus_prefix_lengths: List[int] = []
        _dbg_prompt_lens: List[int] = []
        _dbg_prefix_lens: List[int] = []
        _dbg_teacher_full_lens: List[int] = []
        _dbg_has_field: List[bool] = []
        for i, item in enumerate(inputs):
            row_attn = encoded_inputs['attention_mask'][i].bool()
            prompt_ids = encoded_inputs['input_ids'][i][row_attn].tolist()
            has_field = 'teacher_full_token_ids' in item
            if q > 0.0 and has_field:
                full = list(item['teacher_full_token_ids'])
                Q_i = max(_RESPONSE_MIN_LEN, int(q * len(full)))
                Q_i = min(Q_i, len(full))
                prefix_ids = full[:Q_i]
                _dbg_teacher_full_lens.append(len(full))
            else:
                prefix_ids = []
                _dbg_teacher_full_lens.append(0)
            _dbg_has_field.append(has_field)
            _dbg_prompt_lens.append(len(prompt_ids))
            _dbg_prefix_lens.append(len(prefix_ids))
            prompt_with_prefix_ids.append(prompt_ids + prefix_ids)
            prompt_plus_prefix_lengths.append(len(prompt_ids) + len(prefix_ids))

        import sys as _sys
        print(
            f'[TP_BPOD_INPUTS step={self.state.global_step} q={q:.2f}] '
            f'batch_size={len(inputs)} '
            f'item0_keys={list(inputs[0].keys()) if inputs else None} '
            f'has_teacher_field={_dbg_has_field} '
            f'teacher_full_lens={_dbg_teacher_full_lens} '
            f'prompt_lens={_dbg_prompt_lens} '
            f'prefix_lens={_dbg_prefix_lens}',
            flush=True, file=_sys.stderr,
        )

        # 3. Call vLLM engine directly with RolloutInferRequest+data_dict carrier.
        #    Wrap with the same wake/sleep + weight-sync state management that
        #    `RolloutTrainerMixin._fast_infer` uses (rollout_mixin.py:908-956).
        import inspect
        from contextlib import nullcontext
        from swift.infer_engine.protocol import RolloutInferRequest, RequestConfig
        from swift.rlhf_trainers.utils import aggressive_empty_cache, set_expandable_segments

        args = self.args
        gen_cfg = self.generation_config
        infer_requests = [
            RolloutInferRequest(
                messages=[{'role': 'user', 'content': ''}],
                data_dict={'prompt_token_ids': ids},
            )
            for ids in prompt_with_prefix_ids
        ]
        request_config = RequestConfig(
            max_tokens=int(self.args.max_completion_length),
            temperature=getattr(gen_cfg, 'temperature', 1.0),
            top_p=getattr(gen_cfg, 'top_p', 1.0),
            return_details=True,
        )

        # Wake up weights if sleeping (sleep_level > 0).
        if getattr(args, 'sleep_level', 0) > 0 and self.engine.inner_model_executor.is_sleeping:
            wake_up_params = inspect.signature(self.engine.engine.wake_up).parameters
            wake_kwargs = {'tags': ['weights']} if 'tags' in wake_up_params else {}
            aggressive_empty_cache()
            self.engine.engine.wake_up(**wake_kwargs)

        # Sync training weights to vLLM if step advanced.
        if self.state.global_step != self._last_loaded_step or getattr(args, 'sleep_level', 0) == 2:
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step

        offload_ctx = self.offload_context if getattr(self, 'enable_offload', False) else nullcontext
        with offload_ctx():
            # Wake KV cache if still sleeping.
            if (getattr(args, 'sleep_level', 0) > 0
                    and self.engine.inner_model_executor.is_sleeping
                    and 'tags' in inspect.signature(self.engine.engine.wake_up).parameters):
                aggressive_empty_cache()
                set_expandable_segments(False)
                self.engine.engine.wake_up(tags=['kv_cache'])

            rollout_outputs = self.engine.infer(infer_requests, request_config, use_tqdm=False)
            import sys as _sys
            try:
                _r0 = rollout_outputs[0]
                _resp0 = getattr(_r0, 'response', None) or _r0
                _ch0 = _resp0.choices[0]
                _ptids = getattr(_resp0, 'prompt_token_ids', None) or []
                _tids = getattr(_ch0, 'token_ids', None) or []
                _content = getattr(getattr(_ch0, 'message', None), 'content', '') or ''
                print(
                    f'[TP_BPOD step={self.state.global_step} q={q:.2f}] '
                    f'sent_prompt_token_ids[0]_len={len(prompt_with_prefix_ids[0])} '
                    f'resp.prompt_token_ids_len={len(_ptids)} '
                    f'resp.choices[0].token_ids_len={len(_tids)} '
                    f'first_5_suffix={_tids[:5]} '
                    f'content_first_120={_content[:120]!r}',
                    flush=True, file=_sys.stderr,
                )
            except Exception as _e:
                print(f'[TP_BPOD step={self.state.global_step}] debug-log failed: {_e}',
                      flush=True, file=_sys.stderr)

            if getattr(args, 'sleep_level', 0) > 0:
                self.engine.engine.reset_prefix_cache()
                self.engine.engine.sleep(level=args.sleep_level)
                aggressive_empty_cache()
                set_expandable_segments(True)

        # 4. Parse generated token IDs (suffix only — vLLM does not echo prompt).
        # GRPOVllmEngine.infer wraps responses in `RolloutOutput`, which has
        # `.response: ChatCompletionResponse` → `.choices[0].token_ids`.
        suffix_token_ids_list: List[List[int]] = []
        for output in rollout_outputs:
            resp = getattr(output, 'response', None) or output
            choice = resp.choices[0]
            tids = getattr(choice, 'token_ids', None)
            suffix_token_ids_list.append(list(tids) if tids else [])

        # 5. Assemble [prompt + prefix + suffix] with correct labels mask.
        new_input_ids, new_attn, new_labels = _assemble_tp_bpod_batch(
            prompt_with_prefix_ids,
            suffix_token_ids_list,
            prompt_plus_prefix_lengths,
            pad_token_id=self.processing_class.pad_token_id,
        )

        encoded_inputs['input_ids'] = new_input_ids.to(self.model.device)
        encoded_inputs['attention_mask'] = new_attn.to(self.model.device)
        encoded_inputs['labels'] = new_labels.to(self.model.device)
        encoded_inputs['_data_source'] = DataSource.STUDENT

        with self.template.forward_context(self.model, encoded_inputs):
            return HFSFTTrainer.training_step(self, model, encoded_inputs, num_items_in_batch)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


@torch.no_grad()
def _compute_top_k_entropy_weights(
    model,
    inputs: dict,
    labels: torch.Tensor,
    topk: int = 16,
) -> torch.Tensor:
    """Per-token weight from top-K logits entropy → log → [0, 1].

    Pipeline:
      1. No-grad student forward to get logits.
      2. Keep top-K logits per position, softmax over top-K.
      3. Compute truncated entropy H_t ∈ [0, log(K)].
      4. Weight w_t = log(1 + H_t) / log(1 + log(K)) — maps low-H → ~0,
         high-H → 1, with sub-linear (log) compression in between.

    Returns weights aligned to label positions (shape == labels.shape).
    Invalid (label==-100) positions get weight 0 (so weighted sum ignores
    them downstream).
    """
    import math
    input_ids = inputs['input_ids']
    attn = inputs.get('attention_mask')
    was_training = model.training
    model.eval()
    out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
    if was_training:
        model.train()
    # Cast only the top-K slice to fp32 — avoid materializing the full
    # (B, L, V) fp32 tensor that would OOM on long-seq batches.
    logits = out.logits  # bf16
    top_logits, _ = logits.topk(topk, dim=-1)  # (B, L, K) bf16
    top_logits = top_logits.float()
    top_probs = torch.softmax(top_logits, dim=-1)
    top_log_probs = torch.log_softmax(top_logits, dim=-1)
    ent_top = -(top_probs * top_log_probs).sum(-1)  # (B, L), in [0, log(K)]
    # Map to [0, 1] via log compression
    log_K = math.log(topk)
    weights = torch.log1p(ent_top) / math.log1p(log_K)
    weights = weights.clamp_(min=0.0, max=1.0)
    # NOTE: do NOT zero out by `(labels != -100)` here. weights are aligned
    # to *logits* index space (weight[t] = entropy of logits[t] predicting
    # token at t+1). The caller (_weighted_jsd_loss) filters by mask =
    # (shifted_labels != -100), which already keeps exactly the right
    # logits positions. A naive (labels != -100) zero-out would silently
    # drop the prompt→first-response-token weight (because that logits
    # position is at a non-response label index).
    return weights


def _weighted_jsd_loss(student_logits, teacher_logits=None, labels=None,
                       weights=None, beta=0.5, temperature=1.0,
                       topk=None, teacher_topk_logprobs=None,
                       teacher_topk_indices=None):
    """JSD/KL loss reduced by per-token weights instead of unweighted mean.

    Mirrors `GKDTrainer.generalized_jsd_loss` but replaces the final
    `total_loss / num_valid` with `sum(w_t * jsd_t) / sum(w_t)`. weights
    must be shape (B, L) aligned with `labels` (caller supplies them after
    any -100 zeroing).
    """
    import torch.nn.functional as F

    if teacher_topk_logprobs is not None and teacher_topk_indices is not None:
        s_scaled = student_logits / temperature
        student_logits = torch.gather(s_scaled, dim=-1, index=teacher_topk_indices)
        teacher_logits = teacher_topk_logprobs / temperature
        del s_scaled
        temperature = 1.0
    elif topk is not None and teacher_logits is not None:
        t_scaled = teacher_logits / temperature
        s_scaled = student_logits / temperature
        teacher_logits, topk_idx = torch.topk(t_scaled, k=topk, dim=-1)
        student_logits = torch.gather(s_scaled, dim=-1, index=topk_idx)
        del t_scaled, s_scaled, topk_idx
        temperature = 1.0

    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature

    mask = labels != -100
    s_flat = student_logits[mask]
    t_flat = teacher_logits[mask]
    w_flat = weights[mask].to(s_flat.dtype)
    if s_flat.numel() == 0 or w_flat.sum() == 0:
        # Keep the autograd graph alive so DDP backward doesn't desync
        # when one rank has no valid positions but its peers do.
        return student_logits.sum() * 0.0

    s_log_probs = F.log_softmax(s_flat, dim=-1)
    t_log_probs = F.log_softmax(t_flat, dim=-1)

    if beta == 0:
        jsd_per = F.kl_div(s_log_probs, t_log_probs, reduction='none', log_target=True).sum(-1)
    elif beta == 1:
        jsd_per = F.kl_div(t_log_probs, s_log_probs, reduction='none', log_target=True).sum(-1)
    else:
        beta_t = torch.tensor(beta, dtype=s_log_probs.dtype, device=s_log_probs.device)
        log_beta = torch.log(beta_t)
        log_1mb = torch.log1p(-beta_t)
        mix = torch.logsumexp(
            torch.stack([s_log_probs + log_1mb, t_log_probs + log_beta]), dim=0)
        kl_t = F.kl_div(mix, t_log_probs, reduction='none', log_target=True).sum(-1)
        kl_s = F.kl_div(mix, s_log_probs, reduction='none', log_target=True).sum(-1)
        jsd_per = beta_t * kl_t + (1 - beta_t) * kl_s

    return (w_flat * jsd_per).sum() / w_flat.sum().clamp(min=1e-12)


@torch.no_grad()
def _apply_entropy_topk_mask(
    model,
    inputs: dict,
    labels: torch.Tensor,
    frac: float,
) -> torch.Tensor:
    """Keep only the top-`frac` highest-entropy response positions per row.

    Approach: run a no-grad forward pass on the student to get next-token
    distributions, compute per-position entropy on label-valid positions only,
    and mask the bottom-(1-frac) positions to -100. The subsequent full
    forward inside super().compute_loss() recomputes logits with grad enabled.

    This adds ~1 extra student forward per step but avoids re-implementing
    the upstream GKD loss surgery. The student model size (3B/0.5B) makes
    this overhead acceptable.
    """
    input_ids = inputs['input_ids']
    attn = inputs.get('attention_mask')
    was_training = model.training
    model.eval()
    out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
    if was_training:
        model.train()
    # Cast only top-K logits to fp32 — full-V fp32 OOMs on long sequences.
    # Truncated entropy over top-16 is monotonic with full-V entropy for
    # rank-only purposes (which is what we need here).
    logits = out.logits  # bf16
    top_logits, _ = logits.topk(16, dim=-1)  # (B, L, 16)
    top_logits = top_logits.float()
    log_probs = torch.log_softmax(top_logits, dim=-1)
    probs = log_probs.exp()
    ent_pred = -(probs * log_probs).sum(-1)  # (B, L) truncated entropy
    # ent_pred[:, t-1] is for predicting token t. We want ent at label-position t.
    ent = torch.zeros_like(ent_pred)
    ent[:, 1:] = ent_pred[:, :-1]

    new_labels = labels.clone()
    B = labels.shape[0]
    for i in range(B):
        valid = labels[i] != -100
        idx = valid.nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue
        ent_valid = ent[i, idx]
        # Strict top-fraction. No min-length floor — short rollouts should
        # honor the 20% semantics. (Original `_RESPONSE_MIN_LEN` floor would
        # silently neutralize the ablation on responses <40 tokens.)
        k = max(1, int(frac * idx.numel()))
        k = min(k, idx.numel())
        topk_local = torch.topk(ent_valid, k).indices
        keep_mask = torch.zeros_like(valid)
        keep_mask[idx[topk_local]] = True
        new_labels[i, ~keep_mask] = -100
    return new_labels


@torch.no_grad()
def _apply_forward_prefix_mask(labels: torch.Tensor, ratio: float) -> torch.Tensor:
    """Mask response positions beyond the first ⌊ratio·T⌋ tokens (SP-FPOD)."""
    if ratio >= 1.0:
        return labels
    new_labels = labels.clone()
    for i in range(labels.shape[0]):
        non_pad = (labels[i] != -100).nonzero(as_tuple=False).squeeze(-1)
        if non_pad.numel() == 0:
            continue
        response_start = int(non_pad[0].item())
        response_end = int(non_pad[-1].item())
        T = response_end - response_start + 1
        L = max(_RESPONSE_MIN_LEN, int(ratio * T))
        L = min(L, T)
        cutoff = response_start + L
        if cutoff <= response_end:
            new_labels[i, cutoff:response_end + 1] = -100
    return new_labels


def _assemble_tp_bpod_batch(
    prompt_with_prefix_ids: List[List[int]],
    suffix_token_ids_list: List[List[int]],
    prompt_plus_prefix_lengths: List[int],
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build padded `input_ids`, `attention_mask`, `labels` for the batch.

    Per row:
        input_ids       = prompt_with_prefix + suffix + pad
        attention_mask  = 1 over (prompt_with_prefix + suffix), 0 over pad
        labels          = -100 over (prompt_with_prefix), suffix token_ids, -100 over pad
    """
    assert len(prompt_with_prefix_ids) == len(suffix_token_ids_list) == len(prompt_plus_prefix_lengths)
    batch_size = len(prompt_with_prefix_ids)

    full_seqs = [prompt_with_prefix_ids[i] + suffix_token_ids_list[i] for i in range(batch_size)]
    seq_lengths = [len(s) for s in full_seqs]
    max_len = max(seq_lengths) if seq_lengths else 0

    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)

    for i in range(batch_size):
        L_full = seq_lengths[i]
        L_prompt = prompt_plus_prefix_lengths[i]
        if L_full == 0:
            continue
        seq = torch.tensor(full_seqs[i], dtype=torch.long)
        input_ids[i, :L_full] = seq
        attention_mask[i, :L_full] = 1
        # Only the suffix portion is supervised.
        if L_full > L_prompt:
            labels[i, L_prompt:L_full] = seq[L_prompt:L_full]
        # else: empty suffix (vLLM call failed or generated nothing) → fully masked

    return input_ids, attention_mask, labels
