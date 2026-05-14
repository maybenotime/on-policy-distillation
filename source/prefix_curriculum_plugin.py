"""Prefix-curriculum plugin entry point.

Activated via:
    swift rlhf --rlhf_type gkd \
        --external_plugins source/prefix_curriculum_plugin.py ...

Two patches installed at import time:

1. Swap `swift.rlhf_trainers.GKDTrainer` for our prefix-curriculum subclass so
   `swift rlhf --rlhf_type gkd` will use it. `TrainerFactory.get_trainer_cls`
   does `getattr(module, 'GKDTrainer')` on `swift.rlhf_trainers`, so a single
   attribute swap suffices.

2. Patch `swift.template.base.Template.encode` to honor a
   `data_dict['prompt_token_ids']` carrier. When present, encode bypasses
   chat-template/messages encoding and feeds the raw token IDs to vLLM. This
   is what TP-BPOD needs to inject `prompt + teacher_prefix` into the
   student's rollout. Required to live in the trainer process — vLLM's
   colocate mode keeps the engine in this process, so Template.encode is
   called here. (For server mode the patch would have to be installed in
   the rollout server's process tree, which spawn-based subprocesses make
   inconvenient; for that reason TP-BPOD requires `--vllm_mode colocate`.)

Curriculum env vars (read by `PrefixPacer.from_env()` at trainer __init__):
    PREFIX_DIRECTION ∈ {'forward', 'backward'}   (default: 'forward')
    PREFIX_K         number of stages             (default: 5)
    PREFIX_DELTA     per-stage increment          (default: 0.2)
    PREFIX_P0        forward starting ratio       (default: 0.2)
    PREFIX_Q0        backward starting ratio      (default: 0.8)
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from typing import Any, Dict, Optional

from prefix_curriculum.trainer import PrefixCurriculumGKDTrainer

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Patch 1: swap GKDTrainer
# -----------------------------------------------------------------------------

_swift_rlhf = importlib.import_module('swift.rlhf_trainers')
_original = getattr(_swift_rlhf, 'GKDTrainer', None)
_swift_rlhf.GKDTrainer = PrefixCurriculumGKDTrainer
logger.info(
    '[prefix_curriculum_plugin] swift.rlhf_trainers.GKDTrainer: %s -> %s',
    getattr(_original, '__name__', _original),
    PrefixCurriculumGKDTrainer.__name__,
)


# -----------------------------------------------------------------------------
# Patch 2: Template.encode honors data_dict.prompt_token_ids (TP-BPOD)
# -----------------------------------------------------------------------------

def _install_prompt_token_ids_patch() -> None:
    from swift.template.base import Template
    from swift.template.template_inputs import StdTemplateInputs

    _orig_encode = Template.encode

    def _patched_encode(
        self,
        inputs,
        return_template_inputs: bool = False,
        return_length: bool = False,
    ) -> Dict[str, Any]:
        # Carrier may arrive on either a dataclass attribute or a dict key
        # depending on how the caller built the request.
        data_dict: Optional[Dict[str, Any]] = None
        if hasattr(inputs, 'data_dict'):
            data_dict = getattr(inputs, 'data_dict', None)
        elif isinstance(inputs, dict):
            data_dict = inputs.get('data_dict')

        token_ids = None
        if isinstance(data_dict, dict):
            token_ids = data_dict.get('prompt_token_ids')

        if token_ids:
            ids = list(token_ids)
            encoded: Dict[str, Any] = {
                'input_ids': ids,
                'attention_mask': [1] * len(ids),
                'labels': None,
            }
            if return_length:
                encoded['length'] = len(ids)
            if return_template_inputs:
                # _create_chat_completion_response reads `.images / .audios /
                # .videos` on the template_inputs object, so we hand back a
                # StdTemplateInputs (which the upstream code path also returns
                # here — see template/base.py:572 `encoded['template_inputs'] =
                # chosen`).
                encoded['template_inputs'] = StdTemplateInputs(
                    messages=[{'role': 'user', 'content': ''}]
                )
            return encoded

        return _orig_encode(
            self,
            inputs,
            return_template_inputs=return_template_inputs,
            return_length=return_length,
        )

    Template.encode = _patched_encode
    logger.info(
        '[prefix_curriculum_plugin] patched Template.encode (prompt_token_ids carrier)'
    )


_install_prompt_token_ids_patch()

# -----------------------------------------------------------------------------
# Backward direction also needs the curriculum-backward dataset registered.
# -----------------------------------------------------------------------------

_DIRECTION = os.environ.get('PREFIX_DIRECTION', 'forward').strip().lower()
if _DIRECTION == 'backward':
    from prefix_curriculum import data as _data
    _data.register()
