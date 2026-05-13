"""Prefix-curriculum plugin entry point.

Activated via:
    swift rlhf --rlhf_type gkd \
        --external_plugins source/prefix_curriculum_plugin.py ...

On import this swaps `swift.rlhf_trainers.GKDTrainer` for the prefix-curriculum
subclass. `TrainerFactory.get_trainer_cls` does `getattr(module, 'GKDTrainer')`
on `swift.rlhf_trainers`, so a single attribute swap suffices.

Curriculum is controlled by environment variables read at trainer __init__:
    PREFIX_DIRECTION ∈ {'forward', 'backward'}   (default: 'forward')
    PREFIX_K         number of stages             (default: 5)
    PREFIX_DELTA     per-stage increment          (default: 0.2)
    PREFIX_P0        forward starting ratio       (default: 0.2)
    PREFIX_Q0        backward starting ratio      (default: 0.8)

When PREFIX_DIRECTION=backward, also registers the `prefix-curriculum-backward-math`
dataset (loaded from `final_math_dataset_updated.jsonl`, tokenized with the
student tokenizer). See `prefix_curriculum.data` for extra env vars used.
"""

from __future__ import annotations

import importlib
import logging
import os

from prefix_curriculum.trainer import PrefixCurriculumGKDTrainer

logger = logging.getLogger(__name__)

_swift_rlhf = importlib.import_module('swift.rlhf_trainers')
_original = getattr(_swift_rlhf, 'GKDTrainer', None)
_swift_rlhf.GKDTrainer = PrefixCurriculumGKDTrainer
logger.info(
    '[prefix_curriculum_plugin] swift.rlhf_trainers.GKDTrainer: %s -> %s',
    getattr(_original, '__name__', _original),
    PrefixCurriculumGKDTrainer.__name__,
)

_DIRECTION = os.environ.get('PREFIX_DIRECTION', 'forward').strip().lower()
if _DIRECTION == 'backward':
    from prefix_curriculum import data as _data
    _data.register()
