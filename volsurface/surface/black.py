"""Normalised Black call and its derivatives in total variance.

The cash Black call divided by ``df * F`` is a function of log-moneyness ``k``
and total variance ``w = sigma^2 T`` only::

    C(k, w) = N(d1) - exp(k) N(d2),
    d2 = -k / sqrt(w) - sqrt(w) / 2,
    d1 = d2 + sqrt(w).

Holding ``w`` fixed, the strike derivative collapses to ``-exp(k) N(d2)``.
The identity ``n(d1) = exp(k) n(d2)`` cancels the density terms. When ``w``
depends on ``k``, the chain rule adds ``dC/dw * dw/dk``, and

    dC/dw = n(d1) / (2 sqrt(w)).

Durrleman's function ``g`` is the strike convexity written in ``w``. For a
flat smile ``g = 1`` and the density of ``log(F_T / F)`` is exactly normal
with mean ``-w/2`` and variance ``w``.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

_SQRT_2PI = np.sqrt(2.0 * np.pi)


def _sqrt_w(w: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(w, 0.0))


def d1_d2(k, w):
    """``(d1, d2)``. Where ``w`` is not positive the values are unused by callers."""
    s = _sqrt_w(w)
    with np.errstate(divide="ignore", invalid="ignore"):
        d2 = -k / s - 0.5 * s
    d1 = d2 + s
    return d1, d2


def normal_density(x) -> np.ndarray:
    """Standard normal density ``n(x)``."""
    return np.exp(-0.5 * np.asarray(x, dtype=float) ** 2) / _SQRT_2PI


def call_from_total_variance(k, w) -> np.ndarray:
    """Normalised call. Non-positive ``w`` returns the intrinsic value ``(1 - exp(k))+``."""
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)
    d1, d2 = d1_d2(k, w)
    black = ndtr(d1) - np.exp(k) * ndtr(d2)
    intrinsic = np.maximum(1.0 - np.exp(k), 0.0)
    return np.where(w > 0.0, black, intrinsic)


def durrleman_g(k, w, w_k, w_kk) -> np.ndarray:
    """Durrleman's ``g(k)``. Butterfly arbitrage is ``g < 0``.

    ::

        g = (1 - k w' / (2 w))^2 - (w')^2 / 4 * (1/w + 1/4) + w'' / 2.

    Gatheral and Jacquier (2014), equation (2.1). ``w`` must be positive.
    """
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)
    w_k = np.asarray(w_k, dtype=float)
    w_kk = np.asarray(w_kk, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = (1.0 - k * w_k / (2.0 * w)) ** 2
        g = g - (w_k**2) / 4.0 * (1.0 / w + 0.25) + 0.5 * w_kk
    return np.where(w > 0.0, g, np.nan)


def log_density(k, w, w_k, w_kk) -> np.ndarray:
    """Risk-neutral density of ``log(F_T / F)``.

    ::

        phi(k) = g(k) / sqrt(2 pi w) * exp(-d2(k)^2 / 2).

    Integrates to one when ``g`` comes from a martingale. Negative ``g`` is a
    negative density, which is the butterfly arbitrage.
    """
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)
    g = durrleman_g(k, w, w_k, w_kk)
    _, d2 = d1_d2(k, w)
    phi = g / np.sqrt(2.0 * np.pi * w) * np.exp(-0.5 * d2 * d2)
    return np.where(w > 0.0, phi, np.nan)


def dcall_dk_from_w(k, w, w_k) -> np.ndarray:
    """``dC/dk``, chain rule through total variance. See the module note."""
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)
    w_k = np.asarray(w_k, dtype=float)
    _, d2 = d1_d2(k, w)
    d1 = d2 + _sqrt_w(w)
    n1 = normal_density(d1)
    s = _sqrt_w(w)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = -np.exp(k) * ndtr(d2) + n1 / (2.0 * s) * w_k
    intrinsic_slope = np.where(k < 0.0, -np.exp(k), 0.0)
    return np.where(w > 0.0, slope, intrinsic_slope)


def d2call_dk2_from_w(k, w, w_k, w_kk) -> np.ndarray:
    """``d²C/dk²``, differentiating :func:`dcall_dk_from_w` in ``k``.

    ``d2`` and ``d1`` depend on ``k`` both directly and through ``w(k)``.
    With ``s = sqrt(w)``::

        d(d2)/dk = -1/s + k w' / (2 s^3) - w' / (4 s),
        d(d1)/dk = d(d2)/dk + w' / (2 s).

    The strike piece ``-exp(k) N(d2)`` contributes
    ``-exp(k) N(d2) - exp(k) n(d2) d(d2)/dk``. The variance piece is a product
    ``n(d1) * w' / (2 s)``, differentiated with the product rule. ``n'(d1)``
    brings a factor ``-d1``.
    """
    k = np.asarray(k, dtype=float)
    w = np.asarray(w, dtype=float)
    w_k = np.asarray(w_k, dtype=float)
    w_kk = np.asarray(w_kk, dtype=float)
    _, d2 = d1_d2(k, w)
    s = _sqrt_w(w)
    d1 = d2 + s
    n1 = normal_density(d1)
    # n(d1) = exp(k) n(d2), so n(d2) = exp(-k) n(d1).
    n2 = n1 * np.exp(-k)
    dd2 = -1.0 / s + k * w_k / (2.0 * s**3) - w_k / (4.0 * s)
    dd1 = dd2 + w_k / (2.0 * s)
    d_strike = -np.exp(k) * ndtr(d2) - np.exp(k) * n2 * dd2
    dn1 = n1 * (-d1) * dd1
    d_inv_s = -w_k / (4.0 * s**3)  # derivative of 1/(2s)
    d_var = dn1 * (w_k / (2.0 * s)) + n1 * d_inv_s * w_k + n1 * (w_kk / (2.0 * s))
    second = d_strike + d_var
    # Intrinsic (1 - exp(k))+ has second derivative -exp(k) for k < 0 and 0 for k > 0.
    intrinsic_second = np.where(k < 0.0, -np.exp(k), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(w > 0.0, second, intrinsic_second)
    return out


def dcall_dw(k, w) -> np.ndarray:
    """``dC/dw = n(d1) / (2 sqrt(w))`` for ``w > 0``."""
    d1, _ = d1_d2(k, w)
    s = _sqrt_w(w)
    with np.errstate(divide="ignore", invalid="ignore"):
        dual = normal_density(d1) / (2.0 * s)
    return np.where(np.asarray(w, dtype=float) > 0.0, dual, 0.0)
