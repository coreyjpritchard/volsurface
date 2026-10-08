"""Heston (1993) stochastic volatility: characteristic function, call prices, simulation.

Under the pricing measure, with forward ``F_t`` and variance ``v_t``::

    dF_t / F_t = sqrt(v_t) dW_t
    dv_t       = kappa (theta - v_t) dt + xi sqrt(v_t) dZ_t,    d<W, Z>_t = rho dt

``v0`` is the initial variance, ``theta`` the long-run variance, ``kappa`` the speed of mean
reversion, ``xi`` the volatility of variance and ``rho`` the correlation.

The characteristic function is written in the form of Gatheral (2006, ch. 2), which is the
formulation Albrecher, Mayer, Schoutens and Tistaert (2007, "The little Heston trap",
Wilmott) show to be continuous in ``u`` for all parameters, so the complex logarithm never
crosses its branch cut. Calls are priced with the single-integral formula of Lewis (2000):

    C = df * (F - sqrt(F K) / pi * int_0^inf Re[exp(i u x) phi(u - i/2)] / (u^2 + 1/4) du)

with ``x = log(F / K)`` and ``phi`` the characteristic function of ``log(F_T / F)``. The
integral is computed with composite Gauss-Legendre quadrature, shared across strikes.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def char_func(u: ArrayLike, T: float, v0: float, kappa: float, theta: float, xi: float,
              rho: float) -> np.ndarray:
    """``E[exp(i u log(F_T / F_0))]`` for complex ``u``, in the trap-free form."""
    u = np.asarray(u, dtype=complex)
    alpha = -0.5 * u * u - 0.5j * u
    beta = kappa - rho * xi * 1j * u
    gamma = 0.5 * xi * xi
    d = np.sqrt(beta * beta - 4.0 * alpha * gamma)
    # r_minus = (beta - d) / xi^2, rewritten so it stays finite as xi -> 0.
    r_minus = 2.0 * alpha / (beta + d)
    g = (beta - d) / (beta + d)
    e = np.exp(-d * T)
    D = r_minus * (1.0 - e) / (1.0 - g * e)
    # (2 / xi^2) log((1 - g e) / (1 - g)); g = O(xi^2), so use log1p.
    log_term = np.log1p(-g * e) - np.log1p(-g)
    C = kappa * (r_minus * T - log_term / gamma)
    return np.exp(C * theta + D * v0)


def _nodes(u_max: float, x_max: float, order: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Composite Gauss-Legendre nodes and weights on ``[0, u_max]``.

    Panels start at width 1 and grow by 25% each, capped at one period of ``cos(x u)`` for
    the largest ``|x|``, so a slowly decaying tail costs few nodes.
    """
    cap = 2.0 * np.pi / max(x_max, 0.05)
    edges = [0.0]
    while edges[-1] < u_max:
        edges.append(edges[-1] + min(max(0.25 * edges[-1], 1.0), cap))
    edges = np.array(edges)
    x, w = np.polynomial.legendre.leggauss(order)
    half = 0.5 * np.diff(edges)
    mid = 0.5 * (edges[1:] + edges[:-1])
    nodes = (mid[:, None] + half[:, None] * x[None, :]).ravel()
    weights = (half[:, None] * w[None, :]).ravel()
    return nodes, weights


def _u_max(T: float, params: tuple, tol: float = 1e-12) -> float:
    """Upper limit beyond which ``|phi(u - i/2)| / u^2 < tol``."""
    u = np.geomspace(1.0, 1e5, 400)
    size = np.abs(char_func(u - 0.5j, T, *params)) / (u * u + 0.25)
    above = np.flatnonzero(size > tol)
    return float(u[above[-1] + 1]) if above.size and above[-1] + 1 < u.size else float(u[-1])


def call_price(F: float, K: ArrayLike, T: float, v0: float, kappa: float, theta: float,
               xi: float, rho: float, df: float = 1.0) -> np.ndarray:
    """Heston call prices for an array of strikes at one maturity."""
    K = np.asarray(K, dtype=float)
    params = (v0, kappa, theta, xi, rho)
    x = np.log(F / K.ravel())
    u, w = _nodes(_u_max(T, params), float(np.max(np.abs(x), initial=0.0)))
    phi = char_func(u - 0.5j, T, *params) / (u * u + 0.25)
    integral = (np.cos(np.outer(x, u)) * phi.real - np.sin(np.outer(x, u)) * phi.imag) @ w
    call = F - np.sqrt(F * K.ravel()) / np.pi * integral
    # Clip quadrature noise to the no-arbitrage bounds.
    call = np.clip(call, np.maximum(F - K.ravel(), 0.0), F)
    return df * call.reshape(K.shape)


def price(F: float, K: ArrayLike, T: float, v0: float, kappa: float, theta: float, xi: float,
          rho: float, df: float = 1.0, kind="call") -> np.ndarray:
    """Heston call or put prices; puts by put-call parity ``P = C - df (F - K)``."""
    K = np.asarray(K, dtype=float)
    call = call_price(F, K, T, v0, kappa, theta, xi, rho, df)
    put = call - df * (F - K)
    return np.where(np.asarray(kind) == "call", call, put)


def simulate(F: float, T: float, v0: float, kappa: float, theta: float, xi: float,
             rho: float, n: int, m: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Euler full-truncation scheme (Lord, Koekkoek and van Dijk 2010).

    The variance is stepped with ``v^+ = max(v, 0)`` in drift and diffusion; the log forward
    is stepped exactly given ``v^+``. Returns ``(F_paths, v_paths)``, each of shape
    ``(m, n + 1)`` including the ``t = 0`` column.
    """
    rng = np.random.default_rng(seed)
    dt = T / n
    sq = np.sqrt(dt)
    logf = np.empty((m, n + 1))
    v = np.empty((m, n + 1))
    logf[:, 0] = np.log(F)
    v[:, 0] = v0
    for i in range(n):
        z1 = rng.standard_normal(m)
        z2 = rho * z1 + np.sqrt(1.0 - rho * rho) * rng.standard_normal(m)
        vp = np.maximum(v[:, i], 0.0)
        logf[:, i + 1] = logf[:, i] - 0.5 * vp * dt + np.sqrt(vp) * sq * z1
        v[:, i + 1] = v[:, i] + kappa * (theta - vp) * dt + xi * np.sqrt(vp) * sq * z2
    return np.exp(logf), v
