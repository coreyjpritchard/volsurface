"""Train, validation, and test seeds. DESIGN section 22.

``SeedSequence(entropy=E, spawn_key=(split,))`` with split 0 for train,
1 for validation, and 2 for test. A run then spawns four children, in this
order: 0 the regime chain, 1 the vol drivers, 2 the spot-orthogonal driver,
3 surrogate noise. That order is the draw order of
:class:`~volsurface.markets.regime.RegimeSwitchingBergomi`.
"""

from __future__ import annotations

import numpy as np

from volsurface.mc.rng import spawn

DRAW_ORDER = ("regime", "vol", "spot", "surrogate")


def run_seed(entropy: int, split: int) -> np.random.SeedSequence:
    """One split of an experiment. ``split`` is 0, 1, or 2."""
    if split not in (0, 1, 2):
        raise ValueError(f"split must be 0 (train), 1 (validation) or 2 (test), got {split}")
    return np.random.SeedSequence(entropy=int(entropy), spawn_key=(int(split),))


def stream_seeds(seed: int | np.random.SeedSequence) -> dict[str, np.random.SeedSequence]:
    """The four children, keyed by :data:`DRAW_ORDER`."""
    children = spawn(seed, 4)
    return dict(zip(DRAW_ORDER, children, strict=True))
