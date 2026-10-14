"""Cubic spline of implied volatility against log-moneyness.

A natural cubic spline through the quoted vols, one expiry at a time. Between
expiries, total variance is linear in calendar time at fixed ``k``. The cubic
is a polynomial, so past the last quote it keeps curving. That curve is the
wing in the demo: implied vol can oscillate or cross zero, and Durrleman's
``g`` turns negative. ``iv`` returns the spline, including negative values.
``call`` uses Black's formula only where the spline vol is positive, and the
intrinsic value otherwise, so the price stays defined.

``extrapolated`` is true outside the quoted log-moneyness of the bracketing
expiries, which is exactly where that polynomial is no longer an interpolant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from volsurface.core.types import Curves, Domain
from volsurface.surface.base import SurfaceMixin, as_axes
from volsurface.surface.black import (
    call_from_total_variance,
    d2call_dk2_from_w,
    dcall_dk_from_w,
    dcall_dw,
    log_density,
)
from volsurface.surface.linear import _bracket, _domain


class SplineSurface(SurfaceMixin):
    """Natural cubic spline in implied vol. See the module note."""

    def __init__(self, pillars: list[dict], market: Curves, domain: Domain):
        if not pillars:
            raise ValueError("SplineSurface needs at least one expiry")
        self.pillars = sorted(pillars, key=lambda p: p["T"])
        self.market = market
        self.domain = domain
        self._splines = [
            CubicSpline(p["k"], p["iv"], bc_type="natural", extrapolate=True) for p in self.pillars
        ]

    def call(self, T, k) -> np.ndarray:
        return self._pack(T, k)[0]

    def iv(self, T, k) -> np.ndarray:
        """Spline volatility, not the inversion of ``call``.

        Where the spline is negative, ``call`` has already fallen back to
        intrinsic value, which has no implied vol. This method still returns
        the negative number, because that is the wing.
        """
        T, k = as_axes(T, k)
        out = np.empty((T.size, k.size))
        times = np.array([p["T"] for p in self.pillars])
        for i, t in enumerate(T):
            left, right, alpha = _bracket(times, float(t))
            if right is None:
                out[i] = self._splines[left](k)
                continue
            # Blend total variance, then convert back, keeping the sign of neither
            # pillar when one of them is negative: a negative vol is not a variance.
            w = self._variance_row(left, right, alpha, float(t), k)
            out[i] = np.where(w > 0.0, np.sqrt(w / float(t)), self._splines[left](k))
        return out

    def dcall_dk(self, T, k, h: float = 1e-4) -> np.ndarray:
        return self._pack(T, k)[1]

    def d2call_dk2(self, T, k, h: float = 1e-4) -> np.ndarray:
        return self._pack(T, k)[2]

    def dcall_dT(self, T, k, h: float = 1e-4) -> np.ndarray:
        return self._pack(T, k)[3]

    def total_variance(self, T, k) -> np.ndarray:
        T, k = as_axes(T, k)
        return self._pack(T, k)[4]

    def density(self, T, k) -> np.ndarray:
        T, k = as_axes(T, k)
        w, w_k, w_kk, _, _ = self._w_derivs(T, k)
        return log_density(k[None, :], w, w_k, w_kk)

    def variance_slopes(self, T, k):
        """Analytic ``(w, dw/dk, d²w/dk², dw/dT)`` from the spline of implied vol."""
        w, w_k, w_kk, w_T, _ = self._w_derivs(T, k)
        return w, w_k, w_kk, w_T

    def local_var(self, T, k, h: float = 1e-4) -> np.ndarray:
        """``(dw/dT) / g``. NaN with one expiry, where ``dw/dT`` is not observed."""
        from volsurface.surface.black import durrleman_g

        T, k = as_axes(T, k)
        w, w_k, w_kk, w_T, _ = self._w_derivs(T, k)
        g = durrleman_g(k[None, :], w, w_k, w_kk)
        with np.errstate(divide="ignore", invalid="ignore"):
            local = w_T / g
        observed = np.array([self._has_calendar(float(t)) for t in T])
        return np.where(observed[:, None] & (w > 0.0) & (np.abs(g) > 1e-12), local, np.nan)

    def extrapolated(self, T, k) -> np.ndarray:
        T, k = as_axes(T, k)
        out = np.zeros((T.size, k.size), dtype=bool)
        times = np.array([p["T"] for p in self.pillars])
        for i, t in enumerate(T):
            left, right, _ = _bracket(times, float(t))
            lo = self.pillars[left]["k"][0]
            hi = self.pillars[left]["k"][-1]
            if right is not None:
                lo = max(lo, self.pillars[right]["k"][0])
                hi = min(hi, self.pillars[right]["k"][-1])
            out[i] = (k < lo) | (k > hi)
            if float(t) < times[0] or float(t) > times[-1]:
                out[i] = True
        return out

    def _pack(self, T, k):
        T, k = as_axes(T, k)
        w, w_k, w_kk, w_T, positive = self._w_derivs(T, k)
        call = call_from_total_variance(k[None, :], w)
        dk = dcall_dk_from_w(k[None, :], w, w_k)
        dkk = d2call_dk2_from_w(k[None, :], w, w_k, w_kk)
        # dC/dT = dC/dw * dw/dT where the variance is positive. Intrinsic has none.
        dt = np.where(positive, dcall_dw(k[None, :], w) * w_T, 0.0)
        return call, dk, dkk, dt, w

    def _w_derivs(self, T, k):
        T, k = as_axes(T, k)
        w = np.empty((T.size, k.size))
        w_k = np.empty_like(w)
        w_kk = np.empty_like(w)
        w_T = np.empty_like(w)
        positive = np.empty_like(w, dtype=bool)
        times = np.array([p["T"] for p in self.pillars], dtype=float)
        for i, t in enumerate(T):
            left, right, alpha = _bracket(times, float(t))
            w0, k0, kk0 = self._pillar_w(left, k)
            if right is None:
                w[i], w_k[i], w_kk[i] = w0, k0, kk0
                w_T[i] = self._pillar_w_T(times, left, k)
            else:
                w1, k1, kk1 = self._pillar_w(right, k)
                w[i] = (1.0 - alpha) * w0 + alpha * w1
                w_k[i] = (1.0 - alpha) * k0 + alpha * k1
                w_kk[i] = (1.0 - alpha) * kk0 + alpha * kk1
                gap = times[right] - times[left]
                w_T[i] = (w1 - w0) / gap
            positive[i] = w[i] > 0.0
        return w, w_k, w_kk, w_T, positive

    def _pillar_w(self, index: int, k: np.ndarray):
        """Total variance of one spline. Negative vols contribute 0, not a square."""
        spline = self._splines[index]
        iv = spline(k)
        iv_k = spline(k, 1)
        iv_kk = spline(k, 2)
        T = self.pillars[index]["T"]
        positive = iv > 0.0
        w = np.where(positive, iv**2 * T, 0.0)
        # d(iv^2)/dk = 2 iv iv'. Below zero the clipped variance is flat.
        w_k = np.where(positive, 2.0 * iv * iv_k * T, 0.0)
        w_kk = np.where(positive, 2.0 * (iv_k**2 + iv * iv_kk) * T, 0.0)
        return w, w_k, w_kk

    def _variance_row(self, left, right, alpha, t, k):
        w0, _, _ = self._pillar_w(left, k)
        w1, _, _ = self._pillar_w(right, k)
        return (1.0 - alpha) * w0 + alpha * w1

    def _pillar_w_T(self, times: np.ndarray, index: int, k: np.ndarray) -> np.ndarray:
        """One-sided dw/dT. The smaller slope if the pillar has two neighbours."""

        def level(i):
            return self._pillar_w(i, k)[0]

        slopes = []
        if index > 0:
            slopes.append((level(index) - level(index - 1)) / (times[index] - times[index - 1]))
        if index < times.size - 1:
            slopes.append((level(index + 1) - level(index)) / (times[index + 1] - times[index]))
        if not slopes:
            return np.zeros_like(k, dtype=float)
        return np.vstack(slopes).min(axis=0)

    def _has_calendar(self, t: float) -> bool:
        times = np.array([p["T"] for p in self.pillars])
        if times.size < 2:
            return False
        return bool(times[0] <= t <= times[-1])


def spline_surface(quotes: pd.DataFrame, market: Curves) -> SplineSurface:
    """Spline each expiry's implied vol. Needs a finite ``iv`` column and four strikes."""
    pillars = []
    for _, group in quotes.groupby("expiry", sort=True):
        group = group.dropna(subset=["iv"]).drop_duplicates("k").sort_values("k")
        if len(group) < 4:
            continue
        pillars.append(
            {
                "T": float(group["T"].iloc[0]),
                "k": group["k"].to_numpy(dtype=float),
                "iv": group["iv"].to_numpy(dtype=float),
            }
        )
    if not pillars:
        raise ValueError("no expiry had four implied vols to spline")
    return SplineSurface(pillars, market, _domain(pillars))
