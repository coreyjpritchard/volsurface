"""Seed handling for every Monte Carlo draw.

``docs/DESIGN.md`` section 9. An ``int`` and a ``numpy.random.SeedSequence``
both go through :func:`numpy.random.default_rng`. A model run spawns one child
of the parent seed, not one child per strike or expiry. The order in which a
model then draws standard normals is part of that model's docstring: two
implementations match pathwise only when they consume draws in the same order.
"""

from __future__ import annotations

import numpy as np


def as_seed_sequence(seed: int | np.random.SeedSequence) -> np.random.SeedSequence:
    """Return a fresh ``SeedSequence``. An int becomes one. A sequence is copied.

    ``SeedSequence.spawn`` advances ``n_children_spawned`` on the object it is
    called on. Common random numbers pass the same seed into two repricings and
    need the same child 0 both times. Returning the caller's object would make
    the second call spawn the next child. The copy keeps ``entropy``,
    ``spawn_key``, and ``pool_size``, and leaves the caller unmoved.
    """
    if isinstance(seed, np.random.SeedSequence):
        return np.random.SeedSequence(
            seed.entropy, spawn_key=seed.spawn_key, pool_size=seed.pool_size
        )
    return np.random.SeedSequence(int(seed))


def generator(seed: int | np.random.SeedSequence) -> np.random.Generator:
    """The generator for one model run.

    This is ``np.random.default_rng`` on :func:`as_seed_sequence`. Call it once
    per run and draw from the result in the order the model documents.
    """
    return np.random.default_rng(as_seed_sequence(seed))


def spawn(seed: int | np.random.SeedSequence, n: int) -> list[np.random.SeedSequence]:
    """``n`` independent children. One child per stream, not per grid node."""
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    return list(as_seed_sequence(seed).spawn(n))
