"""Curriculum ratio controller for prefix-curriculum OPD.

SP-FPOD (forward): ratio = p, starts at p0 and advances toward 1.0 each stage.
TP-BPOD (backward): ratio = q, starts at q0 and decreases toward 0.0 each stage.

One stage = one epoch over the fixed query set. The ratio is constant within
a stage and advances via `advance()` (called from `on_epoch_end`).
"""

from __future__ import annotations

import os


class PrefixPacer:

    def __init__(self, direction: str, ratio_0: float, delta: float, num_stages: int):
        if direction not in ('forward', 'backward'):
            raise ValueError(f"direction must be 'forward' or 'backward', got {direction!r}")
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
        return max(0.0, self.ratio_0 - self.delta * self.stage)

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

        PREFIX_DIRECTION : 'forward' (default) or 'backward'
        PREFIX_K         : number of stages, default 5
        PREFIX_DELTA     : per-stage increment magnitude, default 0.2
        PREFIX_P0        : forward starting ratio, default 0.2
        PREFIX_Q0        : backward starting ratio, default 0.8
        """
        direction = os.environ.get('PREFIX_DIRECTION', 'forward').strip().lower()
        num_stages = int(os.environ.get('PREFIX_K', '5'))
        delta = float(os.environ.get('PREFIX_DELTA', '0.2'))
        if direction == 'forward':
            ratio_0 = float(os.environ.get('PREFIX_P0', '0.2'))
        else:
            ratio_0 = float(os.environ.get('PREFIX_Q0', '0.8'))
        return cls(direction, ratio_0, delta, num_stages)
