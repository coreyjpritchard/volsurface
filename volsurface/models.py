"""One interface for every pricing model.

A model is an immutable object holding its parameters, with one method::

    price(F, K, T, df=1.0, kind="call") -> np.ndarray

- ``F`` (float): forward price to expiry ``T``.
- ``K`` (array-like): strikes. The result has the shape of ``K``.
- ``T`` (float): time to expiry in years. One maturity per call, so a Monte Carlo model can
  price every strike from one set of paths.
- ``df`` (float): discount factor to ``T``. Prices are ``df`` times the forward price.
- ``kind``: ``"call"`` or ``"put"``, or an array of those with the shape of ``K``.

Rates and dividends enter only through ``F`` and ``df``, so a model never sees spot, rate
or yield. Models with a closed form or Fourier price (Black-Scholes, Heston, SABR) are
deterministic; Monte Carlo models (rough Bergomi) take ``seed`` and path counts as fields
and must return the same prices for the same fields.

:func:`implied_surface` turns any model into a grid of Black implied volatilities, which
is how models are compared with each other and with the market.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike

from volsurface import bs, heston


@runtime_checkable
class Model(Protocol):
    def price(self, F: float, K: ArrayLike, T: float, df: float = 1.0,
              kind="call") -> np.ndarray: ...


@dataclass(frozen=True)
class BlackScholes:
    """Constant volatility ``sigma``: the implied vol surface is flat at ``sigma``."""

    sigma: float

    def price(self, F, K, T, df=1.0, kind="call"):
        return bs.price(F, K, T, self.sigma, df, kind)


@dataclass(frozen=True)
class Heston:
    """Heston (1993). See :mod:`volsurface.heston` for the dynamics."""

    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float

    def price(self, F, K, T, df=1.0, kind="call"):
        return heston.price(F, K, T, self.v0, self.kappa, self.theta, self.xi, self.rho,
                            df, kind)


def implied_surface(model: Model, F: float, strikes: ArrayLike, maturities: ArrayLike,
                    df: float | ArrayLike = 1.0) -> np.ndarray:
    """Black implied vols of ``model`` on a strike by maturity grid.

    ``strikes`` is either one row of strikes shared by all maturities, shape ``(n_K,)``, or
    one row per maturity, shape ``(n_T, n_K)``. ``df`` is a scalar or one value per maturity.
    Out-of-the-money options are priced and inverted (puts below ``F``, calls at or above),
    which keeps the inversion well conditioned. Returns shape ``(n_T, n_K)``; NaN where the
    price carries no volatility information (at intrinsic value or at its upper bound).
    """
    T = np.atleast_1d(np.asarray(maturities, dtype=float))
    K = np.asarray(strikes, dtype=float)
    K = np.broadcast_to(K, (T.size, K.shape[-1]))
    dfs = np.broadcast_to(np.asarray(df, dtype=float), T.shape)
    out = np.empty(K.shape)
    for i, (t, d) in enumerate(zip(T, dfs, strict=True)):
        kind = np.where(K[i] < F, "put", "call")
        p = model.price(F, K[i], t, d, kind)
        out[i] = bs.implied_vol(p, F, K[i], t, d, kind)
    return out
