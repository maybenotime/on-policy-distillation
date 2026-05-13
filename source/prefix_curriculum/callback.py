"""TrainerCallback that advances the curriculum ratio at each epoch boundary."""

from __future__ import annotations

import logging

from transformers import TrainerCallback

from .pacer import PrefixPacer

logger = logging.getLogger(__name__)


class PrefixPacerCallback(TrainerCallback):

    def __init__(self, pacer: PrefixPacer):
        self.pacer = pacer

    def on_train_begin(self, args, state, control, **kwargs):
        logger.info('[PrefixPacer] train start: %r', self.pacer)

    def on_epoch_end(self, args, state, control, **kwargs):
        prev = self.pacer.current_ratio()
        self.pacer.advance()
        new = self.pacer.current_ratio()
        logger.info(
            '[PrefixPacer] epoch %.0f end: stage %d/%d, ratio %.3f -> %.3f',
            state.epoch, self.pacer.stage, self.pacer.num_stages - 1, prev, new,
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        logs['prefix_ratio'] = self.pacer.current_ratio()
        logs['prefix_stage'] = self.pacer.stage
