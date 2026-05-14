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
        return super().compute_loss(
            model, inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
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
        for i, item in enumerate(inputs):
            row_attn = encoded_inputs['attention_mask'][i].bool()
            prompt_ids = encoded_inputs['input_ids'][i][row_attn].tolist()
            if q > 0.0 and 'teacher_full_token_ids' in item:
                full = list(item['teacher_full_token_ids'])
                Q_i = max(_RESPONSE_MIN_LEN, int(q * len(full)))
                Q_i = min(Q_i, len(full))
                prefix_ids = full[:Q_i]
            else:
                prefix_ids = []
            prompt_with_prefix_ids.append(prompt_ids + prefix_ids)
            prompt_plus_prefix_lengths.append(len(prompt_ids) + len(prefix_ids))

        # 3. Call vLLM engine directly with RolloutInferRequest+data_dict carrier.
        from swift.infer_engine.protocol import RolloutInferRequest, RequestConfig
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
        rollout_outputs = self.engine.infer(infer_requests, request_config, use_tqdm=False)

        # 4. Parse generated token IDs (suffix only — vLLM does not echo prompt).
        suffix_token_ids_list: List[List[int]] = []
        for output in rollout_outputs:
            choice = output.choices[0] if hasattr(output, 'choices') else output['choices'][0]
            tids = getattr(choice, 'token_ids', None)
            if tids is None and isinstance(choice, dict):
                tids = choice.get('token_ids')
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
