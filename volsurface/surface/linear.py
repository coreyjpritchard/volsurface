"""Piecewise-linear normalised call, in the strike.

At one expiry the call is linear in the normalised strike ``x = exp(k)``
between quoted nodes, and flat outside them. Between expiries it is linear in
``T`` at fixed ``k``. This is the ``eta = 0`` interpolation: the price, not
the implied volatility, is what gets joined by straight lines.

Flat extrapolation on the put wing is not convex. A call that is still
decreasing in strike cannot meet a flat line without a negative kink. That
region is marked extrapolated. On the quoted nodes themselves the discrete
butterfly is the jump in slope from the segment on the left to the segment on
the right. Endpoint nodes use the interior slope on both sides, so the flat
extension is not counted as a quoted-node violation. The extension is still
visible to ``check_surface`` if the grid steps outside the quotes, where the
slope is 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from volsurface.core.types import Curves, Domain
from volsurface.surface.base import SurfaceMixin, as_axes


class LinearSurface(SurfaceMixin):
    """Linear interpolation of the normalised call. See the module note."""

    def __init__(self, pillars: list[dict], market: Curves, domain: Domain):
        if not pillars:
            raise ValueError("LinearSurface needs at least one expiry")
        self.pillars = sorted(pillars, key=lambda p: p["T"])
        self.market = market
        self.domain = domain
        for pillar in self.pillars:
            x = np.asarray(pillar["x"], dtype=float)
            if np.any(np.diff(x) <= 0.0):
                raise ValueError("pillar strikes must be strictly increasing")

    def call(self, T, k) -> np.ndarray:
        return self._fields(T, k)[0]

    def dcall_dk(self, T, k, h: float = 1e-4) -> np.ndarray:
        return self._fields(T, k)[1]

    def d2call_dk2(self, T, k, h: float = 1e-4) -> np.ndarray:
        return self._fields(T, k)[2]

    def dcall_dT(self, T, k, h: float = 1e-4) -> np.ndarray:
        return self._fields(T, k)[3]

    def density(self, T, k) -> np.ndarray:
        """Continuous density is 0 between nodes. Atoms at the nodes are not a density value."""
        T, k = as_axes(T, k)
        return np.zeros((T.size, k.size))

    def local_var(self, T, k, h: float = 1e-4) -> np.ndarray:
        """Not defined: Dupire divides by a second derivative that is zero between nodes."""
        T, k = as_axes(T, k)
        return np.full((T.size, k.size), np.nan)

    def extrapolated(self, T, k) -> np.ndarray:
        """True outside the quoted expiries, or outside a bracketing expiry's strikes."""
        T, k = as_axes(T, k)
        out = np.zeros((T.size, k.size), dtype=bool)
        times = self._times()
        for i, t in enumerate(T):
            left, right, alpha = _bracket(times, float(t))
            k_lo, k_hi = self._k_span(left, right, alpha)
            out[i] = (k < k_lo) | (k > k_hi)
            if float(t) < times[0] or float(t) > times[-1]:
                out[i] = True
        return out

    def _fields(self, T, k):
        T, k = as_axes(T, k)
        call = np.empty((T.size, k.size))
        dk = np.empty_like(call)
        dkk = np.empty_like(call)
        dt = np.empty_like(call)
        times = self._times()
        for i, t in enumerate(T):
            # Outside the quoted expiries the call is the edge expiry, held flat in T.
            left, right, alpha = _bracket(times, float(t))
            c0, k0, kk0 = _on_pillar(self.pillars[left], k)
            if right is None:
                call[i], dk[i], dkk[i] = c0, k0, kk0
                # On a pillar the time derivative is one-sided. If both neighbours
                # exist, the more negative one is kept, so a calendar break on
                # either side is visible at the node.
                dt[i] = _pillar_time_slope(self.pillars, times, left, k)
                continue
            c1, k1, kk1 = _on_pillar(self.pillars[right], k)
            call[i] = (1.0 - alpha) * c0 + alpha * c1
            dk[i] = (1.0 - alpha) * k0 + alpha * k1
            dkk[i] = (1.0 - alpha) * kk0 + alpha * kk1
            gap = self.pillars[right]["T"] - self.pillars[left]["T"]
            dt[i] = (c1 - c0) / gap
        return call, dk, dkk, dt

    def _times(self) -> np.ndarray:
        return np.array([p["T"] for p in self.pillars], dtype=float)

    def _k_span(self, left: int, right: int | None, alpha: float) -> tuple[float, float]:
        lo = self.pillars[left]["k"][0]
        hi = self.pillars[left]["k"][-1]
        if right is not None:
            lo = max(lo, self.pillars[right]["k"][0])
            hi = min(hi, self.pillars[right]["k"][-1])
        return float(lo), float(hi)


def linear_surface(quotes: pd.DataFrame, market: Curves) -> LinearSurface:
    """Build a linear surface from the quote table of a ``MarketData``.

    Puts are converted by parity, ``cash_call = cash_put + df (F - K)``, then
    divided by ``df F``. On American quotes that parity is approximate.
    """
    pillars = _pillars(quotes)
    domain = _domain(pillars)
    return LinearSurface(pillars, market, domain)


def _pillars(quotes: pd.DataFrame) -> list[dict]:
    pillars = []
    for _, group in quotes.groupby("expiry", sort=True):
        group = group.drop_duplicates("k").sort_values("k")
        if len(group) < 2:
            continue
        F = float(group["F"].iloc[0])
        df = float(group["df"].iloc[0])
        strike = group["K"].to_numpy(dtype=float)
        price = group["price"].to_numpy(dtype=float)
        put = group["kind"].to_numpy() == "put"
        cash_call = np.where(put, price + df * (F - strike), price)
        normalised = cash_call / (df * F)
        k = group["k"].to_numpy(dtype=float)
        pillars.append(
            {
                "T": float(group["T"].iloc[0]),
                "k": k,
                "x": np.exp(k),
                "c": normalised,
            }
        )
    if not pillars:
        raise ValueError("no expiry had two distinct strikes")
    return pillars


def _domain(pillars: list[dict]) -> Domain:
    return Domain(
        t_min=min(p["T"] for p in pillars),
        t_max=max(p["T"] for p in pillars),
        k_min=min(p["k"][0] for p in pillars),
        k_max=max(p["k"][-1] for p in pillars),
    )


def _bracket(times: np.ndarray, t: float) -> tuple[int, int | None, float]:
    """Left index, right index, and the weight on the right.

    Outside the first and last expiry, returns that edge expiry with weight 0.
    An exact pillar is not blended with its neighbour.
    """
    t = float(np.clip(t, times[0], times[-1]))
    if times.size == 1 or np.isclose(t, times[-1]):
        return times.size - 1, None, 0.0
    right = int(np.searchsorted(times, t, side="right"))
    left = right - 1
    if np.isclose(t, times[left]) or np.isclose(times[right], times[left]):
        return left, None, 0.0
    alpha = (t - times[left]) / (times[right] - times[left])
    return left, right, float(alpha)


def _pillar_time_slope(pillars: list[dict], times: np.ndarray, index: int, k: np.ndarray):
    """One-sided dC/dT at a pillar. The smaller slope if both neighbours exist."""

    def calls(i):
        return _on_pillar(pillars[i], k)[0]

    slopes = []
    if index > 0:
        slopes.append((calls(index) - calls(index - 1)) / (times[index] - times[index - 1]))
    if index < times.size - 1:
        slopes.append((calls(index + 1) - calls(index)) / (times[index + 1] - times[index]))
    if not slopes:
        return np.zeros_like(k, dtype=float)
    return np.vstack(slopes).min(axis=0)


def _on_pillar(pillar: dict, k: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Call, dC/dk, and d²C/dk² on one expiry.

    Between nodes the second derivative equals dC/dk, which is the chain rule
    for a slope that is constant in ``x = exp(k)``: ``dC/dk = s x`` and
    ``d²C/dk² = s x``. At an interior node, d²C/dk² is chosen so the checker's
    strike convexity equals the slope jump ``s_right - s_left``.
    """
    x = np.exp(k)
    nodes = pillar["x"]
    calls = pillar["c"]
    slopes = np.diff(calls) / np.diff(nodes)
    c = np.interp(x, nodes, calls)
    ck = np.empty_like(c)
    ckk = np.empty_like(c)
    # Interval to the right of searchsorted(side="left") - 1, clipped to a real segment.
    for j, xj in enumerate(x):
        at = np.flatnonzero(np.isclose(nodes, xj, rtol=1e-8, atol=1e-8))
        if at.size and 0 < int(at[0]) < nodes.size - 1:
            i = int(at[0])
            s_left = slopes[i - 1]
            s_right = slopes[i]
            s_mid = 0.5 * (s_left + s_right)
            ck[j] = s_mid * xj
            # gamma = s_right - s_left = exp(-2k) (C_kk - C_k) => C_kk = gamma exp(2k) + C_k.
            ckk[j] = (s_right - s_left) * xj**2 + ck[j]
            continue
        if at.size and slopes.size and int(at[0]) in (0, nodes.size - 1):
            # Endpoint: the one-sided interior slope, and convexity 0 (ckk = ck).
            # The flat extension outside the quotes is a different point.
            s = slopes[0] if int(at[0]) == 0 else slopes[-1]
            ck[j] = s * xj
            ckk[j] = ck[j]
            continue
        if xj < nodes[0] or xj > nodes[-1] or slopes.size == 0:
            ck[j] = 0.0
            ckk[j] = 0.0
            continue
        seg = int(np.searchsorted(nodes, xj, side="right") - 1)
        seg = min(max(seg, 0), slopes.size - 1)
        ck[j] = slopes[seg] * xj
        ckk[j] = ck[j]
    return c, ck, ckk
