"""Raw SVI total variance, with analytic derivatives.

One expiry::

    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2)),

with ``b > 0``, ``|rho| < 1``, ``sigma > 0``, and
``a + b * sigma * sqrt(1 - rho^2) >= 0`` so the minimum of ``w`` is not
negative. Differentiating in ``k``::

    w'(k)  = b * (rho + y / s),
    w''(k) = b * sigma^2 / s^3,

where ``y = k - m`` and ``s = sqrt(y^2 + sigma^2)``. Durrleman's ``g`` is then
an explicit function of those two derivatives. A negative ``g`` is a butterfly
arbitrage. The fit does not enforce ``g >= 0``.

Between expiries, ``w`` is linear in ``T`` at fixed ``k``. That blend can
cross, which is a calendar arbitrage the checker will see.

``VOGT_2014`` is Example 3.1 of Gatheral and Jacquier, the parameters Axel Vogt
posted. The paper prints ``(a, b, m, rho, sigma)``. That is not the field
order of :class:`RawSVI`, which is ``(a, b, rho, m, sigma)``. Splatting the
tuple into the constructor swaps ``m`` and ``rho``. Use :meth:`RawSVI.vogt`.
Read in the paper's order, the SVI-JW parameters match their Example 5.1.
``g`` is negative on that slice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from volsurface.core.types import Curves, Domain
from volsurface.surface.base import SurfaceMixin, as_axes
from volsurface.surface.black import (
    call_from_total_variance,
    d2call_dk2_from_w,
    dcall_dk_from_w,
    dcall_dw,
    durrleman_g,
    log_density,
)
from volsurface.surface.linear import _bracket

# Paper order (a, b, m, rho, sigma), not RawSVI(a, b, rho, m, sigma).
# Gatheral and Jacquier (2014), equation (3.8), T = 1. Build with RawSVI.vogt().
VOGT_2014 = (-0.0410, 0.1331, 0.3586, 0.3060, 0.4153)


@dataclass(frozen=True)
class RawSVI:
    """One raw-SVI slice. ``sigma`` here is the SVI curvature parameter, not Black vol."""

    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def w(self, k) -> np.ndarray:
        y = np.asarray(k, dtype=float) - self.m
        return self.a + self.b * (self.rho * y + np.sqrt(y * y + self.sigma**2))

    def w_k(self, k) -> np.ndarray:
        y = np.asarray(k, dtype=float) - self.m
        s = np.sqrt(y * y + self.sigma**2)
        return self.b * (self.rho + y / s)

    def w_kk(self, k) -> np.ndarray:
        y = np.asarray(k, dtype=float) - self.m
        s = np.sqrt(y * y + self.sigma**2)
        return self.b * (self.sigma**2) / s**3

    def g(self, k) -> np.ndarray:
        """Durrleman's ``g``. Negative means this slice has a butterfly arbitrage."""
        return durrleman_g(k, self.w(k), self.w_k(k), self.w_kk(k))

    @classmethod
    def vogt(cls) -> RawSVI:
        """The arbitrageable Vogt slice at ``T = 1``."""
        a, b, m, rho, sigma = VOGT_2014
        return cls(a=a, b=b, rho=rho, m=m, sigma=sigma)


class SVISurface(SurfaceMixin):
    """Raw SVI per expiry, total variance linear in time between expiries."""

    def __init__(self, pillars: list[tuple[float, RawSVI]], market: Curves, domain: Domain):
        if not pillars:
            raise ValueError("SVISurface needs at least one expiry")
        self.pillars = sorted(pillars, key=lambda item: item[0])
        self.market = market
        self.domain = domain

    def call(self, T, k) -> np.ndarray:
        _, k, w, _, _, _, _ = self._parts(T, k)
        return call_from_total_variance(k[None, :], w)

    def total_variance(self, T, k) -> np.ndarray:
        return self._parts(T, k)[2]

    def dcall_dk(self, T, k) -> np.ndarray:
        _, k, w, w_k, _, _, _ = self._parts(T, k)
        return dcall_dk_from_w(k[None, :], w, w_k)

    def d2call_dk2(self, T, k) -> np.ndarray:
        _, k, w, w_k, w_kk, _, _ = self._parts(T, k)
        return d2call_dk2_from_w(k[None, :], w, w_k, w_kk)

    def dcall_dT(self, T, k) -> np.ndarray:
        _, k, w, _, _, w_T, _ = self._parts(T, k)
        return dcall_dw(k[None, :], w) * w_T

    def density(self, T, k) -> np.ndarray:
        _, k, w, w_k, w_kk, _, _ = self._parts(T, k)
        return log_density(k[None, :], w, w_k, w_kk)

    def variance_slopes(self, T, k):
        """Analytic ``(w, dw/dk, d²w/dk², dw/dT)``. Dupire uses these, not a difference."""
        _, _, w, w_k, w_kk, w_T, _ = self._parts(T, k)
        return w, w_k, w_kk, w_T

    def local_var(self, T, k) -> np.ndarray:
        """``(dw/dT) / g(k)``. NaN where a single expiry gives no ``dw/dT``."""
        _, k, w, w_k, w_kk, w_T, calendar = self._parts(T, k)
        g = durrleman_g(k[None, :], w, w_k, w_kk)
        with np.errstate(divide="ignore", invalid="ignore"):
            local = w_T / g
        return np.where(calendar & (w > 0.0) & (np.abs(g) > 1e-12), local, np.nan)

    def g(self, T, k) -> np.ndarray:
        """Durrleman's ``g`` on the blended total variance."""
        _, k, w, w_k, w_kk, _, _ = self._parts(T, k)
        return durrleman_g(k[None, :], w, w_k, w_kk)

    def extrapolated(self, T, k) -> np.ndarray:
        """True outside the quoted expiries or outside the quoted strike window.

        The formula is defined for every ``k``. A value past ``domain.k_min``
        or ``domain.k_max`` is not an interpolation of the quotes that were fit.
        """
        T, k = as_axes(T, k)
        times = np.array([t for t, _ in self.pillars])
        outside_t = (T < times[0]) | (T > times[-1])
        outside_k = (k < self.domain.k_min) | (k > self.domain.k_max)
        return outside_t[:, None] | outside_k[None, :]

    def _parts(self, T, k):
        """Coerce axes, then the total-variance blend. Lists are allowed.

        ``_w`` coerces internally and does not hand the coerced ``k`` back.
        Indexing the caller's ``k`` with ``[None, :]`` fails when that object
        is a Python list, which is how the in-spread check samples one quote.
        """
        T, k = as_axes(T, k)
        w, w_k, w_kk, w_T, calendar = self._w(T, k)
        return T, k, w, w_k, w_kk, w_T, calendar

    def _w(self, T, k):
        T, k = as_axes(T, k)
        w = np.empty((T.size, k.size))
        w_k = np.empty_like(w)
        w_kk = np.empty_like(w)
        w_T = np.empty_like(w)
        calendar = np.zeros(T.size, dtype=bool)
        times = np.array([t for t, _ in self.pillars], dtype=float)
        for i, t in enumerate(T):
            left, right, alpha = _bracket(times, float(t))
            slice_l = self.pillars[left][1]
            if right is None:
                w[i] = slice_l.w(k)
                w_k[i] = slice_l.w_k(k)
                w_kk[i] = slice_l.w_kk(k)
                w_T[i] = self._pillar_w_T(times, left, k)
                calendar[i] = times.size > 1
                continue
            slice_r = self.pillars[right][1]
            w0, w1 = slice_l.w(k), slice_r.w(k)
            w[i] = (1.0 - alpha) * w0 + alpha * w1
            w_k[i] = (1.0 - alpha) * slice_l.w_k(k) + alpha * slice_r.w_k(k)
            w_kk[i] = (1.0 - alpha) * slice_l.w_kk(k) + alpha * slice_r.w_kk(k)
            w_T[i] = (w1 - w0) / (times[right] - times[left])
            calendar[i] = True
        return w, w_k, w_kk, w_T, calendar[:, None]

    def _pillar_w_T(self, times: np.ndarray, index: int, k: np.ndarray) -> np.ndarray:
        def level(i):
            return self.pillars[i][1].w(k)

        slopes = []
        if index > 0:
            slopes.append((level(index) - level(index - 1)) / (times[index] - times[index - 1]))
        if index < times.size - 1:
            slopes.append((level(index + 1) - level(index)) / (times[index + 1] - times[index]))
        if not slopes:
            return np.zeros_like(k, dtype=float)
        return np.vstack(slopes).min(axis=0)


def svi_surface(
    pillars: list[tuple[float, RawSVI]], market: Curves, k_span: tuple[float, float]
) -> SVISurface:
    """Surface from slices that are already fitted. ``k_span`` is the quoted window."""
    domain = Domain(
        t_min=min(t for t, _ in pillars),
        t_max=max(t for t, _ in pillars),
        k_min=k_span[0],
        k_max=k_span[1],
    )
    return SVISurface(pillars, market, domain)


def slices_from_quotes(quotes: pd.DataFrame) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """``(T, k, total_variance)`` per expiry, for the fitter."""
    out = []
    for _, group in quotes.groupby("expiry", sort=True):
        group = group.dropna(subset=["iv"]).drop_duplicates("k").sort_values("k")
        if len(group) < 5:
            continue
        T = float(group["T"].iloc[0])
        k = group["k"].to_numpy(dtype=float)
        w = group["iv"].to_numpy(dtype=float) ** 2 * T
        out.append((T, k, w))
    return out

