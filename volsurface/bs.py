"""Black-Scholes in forward form (Black 1976), vectorised.

All functions take the forward ``F`` to expiry and the discount factor ``df`` to expiry
instead of spot, rate and dividend yield. With spot ``S``, rate ``r`` and yield ``q``:
``F = S * exp((r - q) T)`` and ``df = exp(-r T)``.

``kind`` is ``"call"`` or ``"put"``, or an array of those strings broadcastable with the
other arguments. Greeks are derivatives with respect to the forward ``F`` (delta, gamma)
and the volatility ``sigma`` (vega).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import ndtr

_SQRT_2PI = np.sqrt(2.0 * np.pi)


def _is_call(kind) -> np.ndarray:
    kind = np.asarray(kind)
    bad = ~np.isin(kind, ("call", "put"))
    if np.any(bad):
        raise ValueError(f"kind must be 'call' or 'put', got {np.unique(kind[bad])}")
    return kind == "call"


def _d1_d2(F, K, T, sigma):
    sd = sigma * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = np.log(F / K) / sd + 0.5 * sd
    return d1, d1 - sd


def _pdf(x):
    return np.exp(-0.5 * x * x) / _SQRT_2PI


def price(
    F: ArrayLike, K: ArrayLike, T: ArrayLike, sigma: ArrayLike, df: ArrayLike = 1.0,
    kind="call",
) -> np.ndarray:
    """Discounted Black price ``df * E[(F_T - K)^+]`` (call) or ``df * E[(K - F_T)^+]``."""
    F, K, T, sigma, df = (np.asarray(a, dtype=float) for a in (F, K, T, sigma, df))
    is_call = _is_call(kind)
    d1, d2 = _d1_d2(F, K, T, sigma)
    call = F * ndtr(d1) - K * ndtr(d2)
    put = K * ndtr(-d2) - F * ndtr(-d1)
    # sigma * sqrt(T) == 0: intrinsic value.
    zero = sigma * np.sqrt(T) == 0
    call = np.where(zero, np.maximum(F - K, 0.0), call)
    put = np.where(zero, np.maximum(K - F, 0.0), put)
    return df * np.where(is_call, call, put)


def delta(F, K, T, sigma, df=1.0, kind="call") -> np.ndarray:
    """dPrice/dF. Call: ``df * N(d1)``; put: ``df * (N(d1) - 1)``."""
    F, K, T, sigma, df = (np.asarray(a, dtype=float) for a in (F, K, T, sigma, df))
    d1, _ = _d1_d2(F, K, T, sigma)
    return df * np.where(_is_call(kind), ndtr(d1), ndtr(d1) - 1.0)


def gamma(F, K, T, sigma, df=1.0) -> np.ndarray:
    """d2Price/dF2, the same for calls and puts: ``df * n(d1) / (F sigma sqrt(T))``."""
    F, K, T, sigma, df = (np.asarray(a, dtype=float) for a in (F, K, T, sigma, df))
    d1, _ = _d1_d2(F, K, T, sigma)
    return df * _pdf(d1) / (F * sigma * np.sqrt(T))


def vega(F, K, T, sigma, df=1.0) -> np.ndarray:
    """dPrice/dsigma, the same for calls and puts: ``df * F * n(d1) * sqrt(T)``."""
    F, K, T, sigma, df = (np.asarray(a, dtype=float) for a in (F, K, T, sigma, df))
    d1, _ = _d1_d2(F, K, T, sigma)
    return df * F * _pdf(d1) * np.sqrt(T)


def implied_vol(
    price_: ArrayLike, F: ArrayLike, K: ArrayLike, T: ArrayLike, df: ArrayLike = 1.0,
    kind="call", *, lo: float = 1e-6, hi: float = 10.0, tol: float = 1e-12,
    max_iter: int = 100,
) -> np.ndarray:
    """Black implied volatility, vectorised.

    The price is first converted to the out-of-the-money option by put-call parity, where
    the inversion is best conditioned. Then a safeguarded Newton iteration runs on the
    bracket ``[lo, hi]``: a Newton step is taken when it stays inside the current bracket,
    otherwise the bracket is bisected. Returns NaN where the price is at or below intrinsic
    value, at or above the no-arbitrage upper bound, or not finite.
    """
    p, F, K, T, df = np.broadcast_arrays(
        *(np.asarray(a, dtype=float) for a in (price_, F, K, T, df))
    )
    is_call = np.broadcast_to(_is_call(kind), p.shape)
    with np.errstate(invalid="ignore", divide="ignore"):
        undiscounted = p / df
        # Out-of-the-money equivalent: call for K >= F, put for K < F.
        otm_call = K >= F
        target = np.where(
            is_call == otm_call, undiscounted,
            undiscounted - np.where(is_call, F - K, K - F),
        )
        upper = np.where(otm_call, F, K)
        valid = np.isfinite(target) & (T > 0) & (target > 0) & (target < upper)
    kind_otm = np.where(otm_call, "call", "put")

    sigma = np.full(p.shape, np.nan)
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return sigma
    t, f, k, tt, kd = (a.ravel()[idx] for a in (target, F, K, T, kind_otm))
    low = np.full(idx.size, lo)
    high = np.full(idx.size, hi)
    # Start at the inflection point of price in sigma (Manaster and Koehler 1982), or 0.2.
    x = np.clip(np.maximum(np.sqrt(2.0 * np.abs(np.log(f / k)) / tt), 0.2), lo, hi)
    for _ in range(max_iter):
        diff = price(f, k, tt, x, 1.0, kd) - t
        if np.all(np.abs(diff) <= tol * np.maximum(t, 1e-300) + 1e-300):
            break
        high = np.where(diff > 0, x, high)
        low = np.where(diff <= 0, x, low)
        v = vega(f, k, tt, x, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            newton = x - diff / v
        inside = np.isfinite(newton) & (newton > low) & (newton < high)
        x = np.where(inside, newton, 0.5 * (low + high))
    sigma.flat[idx] = x
    return sigma
