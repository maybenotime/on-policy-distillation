"""TrainerCallback that advances the curriculum ratio along training steps."""

from __future__ import annotations

import logging

from transformers import TrainerCallback

from .pacer import PrefixPacer

logger = logging.getLogger(__name__)


class PrefixPacerCallback(TrainerCallback):
    """Step-based prefix curriculum.

    Stage k covers steps [k * max_steps / K, (k+1) * max_steps / K). On every
    `on_step_begin`, recomputes the target stage from `state.global_step /
    state.max_steps`. Falls back to per-epoch advance if `max_steps` is not
    yet set (very early on_train_begin).
    """

    def __init__(self, pacer: PrefixPacer):
        self.pacer = pacer

    def on_train_begin(self, args, state, control, **kwargs):
        logger.info(
            '[PrefixPacer] train start: %r  max_steps=%s',
            self.pacer, getattr(state, 'max_steps', None),
        )

    def on_step_begin(self, args, state, control, **kwargs):
        max_steps = getattr(state, 'max_steps', 0) or 0
        if max_steps <= 0:
            return
        target = self.pacer.stage_for_step(state.global_step, max_steps)
        if target != self.pacer.stage:
            prev = self.pacer.current_ratio()
            self.pacer.stage = target
            new = self.pacer.current_ratio()
            logger.info(
                '[PrefixPacer] step %d/%d: stage %d/%d, ratio %.3f -> %.3f',
                state.global_step, max_steps,
                self.pacer.stage, self.pacer.num_stages - 1, prev, new,
            )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        logs['prefix_ratio'] = self.pacer.current_ratio()
        logs['prefix_stage'] = self.pacer.stage
