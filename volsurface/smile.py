"""Summary statistics of a smile: at-the-money level and skew."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def atm_fit(k: ArrayLike, iv: ArrayLike, half_width: float,
            min_points: int = 5) -> tuple[float, float]:
    """At-the-money implied vol and skew ``d iv / d k`` at ``k = 0``.

    Fits ``iv = a + b k + c k^2`` by least squares to the points with ``|k| <= half_width``
    and returns ``(a, b)``. NaN if fewer than ``min_points`` points fall in the window.
    """
    k, iv = np.asarray(k, dtype=float), np.asarray(iv, dtype=float)
    use = (np.abs(k) <= half_width) & np.isfinite(iv)
    if use.sum() < min_points:
        return float("nan"), float("nan")
    c, b, a = np.polyfit(k[use], iv[use], 2)
    return float(a), float(b)


def skew_window(T: float, scale: float = 0.15, floor: float = 0.01) -> float:
    """Half width in ``k`` for :func:`atm_fit`: ``scale * sqrt(T)``, at least ``floor``.

    ``0.15 * sqrt(T)`` is about one standard deviation of ``log(F_T / F)`` at 15% vol.
    """
    return max(scale * float(np.sqrt(T)), floor)


def power_law(T: ArrayLike, skew: ArrayLike) -> tuple[float, float]:
    """Fit ``|skew| = A * T^(-alpha)`` by least squares in logs. Returns ``(A, alpha)``.

    For rough volatility models ``alpha = 1/2 - H`` (Bayer, Friz and Gatheral 2016).
    """
    T, skew = np.asarray(T, dtype=float), np.abs(np.asarray(skew, dtype=float))
    use = np.isfinite(skew) & (skew > 0) & (T > 0)
    slope, intercept = np.polyfit(np.log(T[use]), np.log(skew[use]), 1)
    return float(np.exp(intercept)), float(-slope)


def market_term_structure(market: pd.DataFrame) -> pd.DataFrame:
    """ATM vol and skew per expiry from :func:`volsurface.chains.market_implied_vols` output.

    Returns columns ``expiry``, ``T``, ``atm``, ``skew``; expiries with too few points near
    the money are dropped.
    """
    rows = []
    for expiry, g in market.groupby("expiry", sort=True):
        T = float(g["T"].iloc[0])
        atm, skew = atm_fit(g["k"], g["iv"], skew_window(T))
        rows.append({"expiry": expiry, "T": T, "atm": atm, "skew": skew})
    return pd.DataFrame(rows).dropna().reset_index(drop=True)


def model_term_structure(model, maturities: ArrayLike, n: int = 9) -> np.ndarray:
    """ATM vol and skew of a model, measured exactly as for the market.

    Prices ``n`` strikes across :func:`skew_window` at each maturity, inverts them and
    applies :func:`atm_fit`. Returns shape ``(len(maturities), 2)``: columns atm, skew.
    """
    from volsurface.models import implied_surface

    out = []
    for T in np.atleast_1d(maturities):
        h = skew_window(T)
        k = np.linspace(-h, h, n)
        iv = implied_surface(model, 1.0, np.exp(k), [T])[0]
        out.append(atm_fit(k, iv, h))
    return np.array(out)
