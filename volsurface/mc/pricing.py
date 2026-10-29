"""Prices with a standard error, kept off the ``ndarray`` that ``price`` returns.

An ndarray subclass would lose a ``.stderr`` attribute on the first arithmetic
operation, and a tuple would break callers that expect an array. ``price``
stays an array. Uncertainty is a :class:`~volsurface.core.types.PriceResult`.
"""

from __future__ import annotations

import time

import numpy as np

from volsurface.core.types import PriceResult, State


def price_with_error(
    model,
    F: float,
    K,
    T: float,
    df: float = 1.0,
    kind="call",
    *,
    state: State | None = None,
    n_paths: int | None = None,
    seed: int | np.random.SeedSequence | None = None,
) -> PriceResult:
    """Price, and attach a standard error when the model has one.

    A Monte Carlo model implements ``price_with_error`` itself and uses
    ``n_paths`` and ``seed`` in place of its fields. A closed-form model is
    timed around ``price`` and returns ``stderr=None``.
    """
    method = getattr(model, "price_with_error", None)
    if method is not None and method is not price_with_error:
        return method(F, K, T, df, kind, state=state, n_paths=n_paths, seed=seed)
    started = time.perf_counter()
    value = np.asarray(model.price(F, K, T, df, kind, state=state), dtype=float)
    return PriceResult(value=value, stderr=None, n_evals=1, wall_time=time.perf_counter() - started)
