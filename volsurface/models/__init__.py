"""One interface for every pricing model.

A model is an immutable object holding its parameters, with one method::

    price(F, K, T, df=1.0, kind="call", *, state=None) -> np.ndarray

- ``F`` (float): forward price to expiry ``T``.
- ``K`` (array-like): strikes. The result has the shape of ``K``.
- ``T`` (float): time to expiry in years. One maturity per call, so a Monte Carlo
  model can price every strike from one set of paths.
- ``df`` (float): discount factor to ``T``. Prices are ``df`` times the forward price.
- ``kind``: ``"call"`` or ``"put"``, or an array of those with the shape of ``K``.
- ``state``: latent state at the pricing time. ``None`` means ``initial_state()``.
  Black-Scholes has an empty state. Heston carries the variance ``v``.

Rates and dividends enter only through ``F`` and ``df``, so a model never sees spot,
rate or yield. Standard errors are not attached to the array. Use
:func:`volsurface.mc.price_with_error` when the estimate has one.

:func:`implied_surface` turns any model into a grid of Black implied volatilities.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from volsurface.bs import implied_vol
from volsurface.core.protocols import Model
from volsurface.core.types import State
from volsurface.models.bs import BlackScholes
from volsurface.models.heston import Heston

__all__ = ["BlackScholes", "Heston", "Model", "implied_surface"]


def implied_surface(
    model: Model,
    F: float,
    strikes: ArrayLike,
    maturities: ArrayLike,
    df: float | ArrayLike = 1.0,
    *,
    state: State | None = None,
) -> np.ndarray:
    """Black implied vols of ``model`` on a strike by maturity grid.

    ``strikes`` is either one row of strikes shared by all maturities, shape ``(n_K,)``, or
    one row per maturity, shape ``(n_T, n_K)``. ``df`` is a scalar or one value per maturity.
    Out-of-the-money options are priced and inverted (puts below ``F``, calls at or above),
    which keeps the inversion well conditioned. Returns shape ``(n_T, n_K)``; NaN where the
    price carries no volatility information (at intrinsic value or at its upper bound).

    ``state`` is forwarded only when it is not None, so a ``price`` method written before
    the state argument still works.
    """
    T = np.atleast_1d(np.asarray(maturities, dtype=float))
    K = np.asarray(strikes, dtype=float)
    K = np.broadcast_to(K, (T.size, K.shape[-1]))
    dfs = np.broadcast_to(np.asarray(df, dtype=float), T.shape)
    out = np.empty(K.shape)
    for i, (t, d) in enumerate(zip(T, dfs, strict=True)):
        kind = np.where(K[i] < F, "put", "call")
        if state is None:
            p = model.price(F, K[i], float(t), float(d), kind)
        else:
            p = model.price(F, K[i], float(t), float(d), kind, state=state)
        out[i] = implied_vol(p, F, K[i], t, d, kind)
    return out
