"""Curriculum ratio controller for prefix-curriculum OPD.

SP-FPOD (forward): ratio = p, starts at p0 and advances toward 1.0 each stage.
TP-BPOD (backward): ratio = q, starts at q0 and decreases toward 0.0 each stage.

Stages are evenly spaced along the *total training steps* (not epochs). Stage k
covers the step range `[k * max_steps / K, (k+1) * max_steps / K)`. The
PrefixPacerCallback updates `self.stage` on each `on_step_begin` based on the
trainer's `state.global_step` / `state.max_steps`. The ratio is constant
within a stage.

This decouples curriculum length from epoch count — a 1-epoch run can still
sweep all K stages, which matters on time-limited clusters where multi-epoch
runs won't finish.
"""

from __future__ import annotations

import os


class PrefixPacer:

    def __init__(self, direction: str, ratio_0: float, delta: float, num_stages: int):
        if direction not in ('forward', 'backward', 'entropy_topk', 'entropy_reweight'):
            raise ValueError(f"direction must be 'forward' / 'backward' / 'entropy_topk' / 'entropy_reweight', got {direction!r}")
        if num_stages < 1:
            raise ValueError(f'num_stages must be >= 1, got {num_stages}')
        self.direction = direction
        self.ratio_0 = float(ratio_0)
        self.delta = float(delta)
        self.num_stages = int(num_stages)
        self.stage = 0

    def current_ratio(self) -> float:
        if self.direction == 'forward':
            return min(1.0, self.ratio_0 + self.delta * self.stage)
        if self.direction == 'entropy_topk':
            # entropy_topk_frac: fraction of *highest-entropy* student tokens
            # whose KL we keep. Constant across stages by default.
            return max(0.0, min(1.0, self.ratio_0 + self.delta * self.stage))
        return max(0.0, self.ratio_0 - self.delta * self.stage)

    def stage_for_step(self, global_step: int, max_steps: int) -> int:
        """Return the stage index for a given training step (step-based curriculum)."""
        if max_steps <= 0:
            return 0
        s = int(global_step * self.num_stages / max_steps)
        return max(0, min(s, self.num_stages - 1))

    def advance(self) -> None:
        if self.stage < self.num_stages - 1:
            self.stage += 1

    def __repr__(self) -> str:
        return (f'PrefixPacer(direction={self.direction!r}, '
                f'stage={self.stage}/{self.num_stages - 1}, '
                f'ratio={self.current_ratio():.3f})')

    @classmethod
    def from_env(cls) -> 'PrefixPacer':
        """Build a pacer from PREFIX_* environment variables.

        PREFIX_DIRECTION : 'forward' (default), 'backward', or 'entropy_topk'
        PREFIX_K         : number of stages, default 10
        PREFIX_DELTA     : per-stage increment magnitude, default 0.1
        PREFIX_P0        : forward starting ratio, default 0.1
        PREFIX_Q0        : backward starting ratio, default 0.9
        PREFIX_TOPK_FRAC : entropy_topk fraction of high-H student tokens to keep, default 0.2
        """
        direction = os.environ.get('PREFIX_DIRECTION', 'forward').strip().lower()
        num_stages = int(os.environ.get('PREFIX_K', '10'))
        delta = float(os.environ.get('PREFIX_DELTA', '0.1'))
        if direction == 'forward':
            ratio_0 = float(os.environ.get('PREFIX_P0', '0.1'))
        elif direction == 'entropy_topk':
            ratio_0 = float(os.environ.get('PREFIX_TOPK_FRAC', '0.2'))
        elif direction == 'entropy_reweight':
            # Top-K for the entropy reweighting computation (default top-16)
            ratio_0 = float(os.environ.get('PREFIX_REWEIGHT_TOPK', '16'))
        else:
            ratio_0 = float(os.environ.get('PREFIX_Q0', '0.9'))
        return cls(direction, ratio_0, delta, num_stages)
