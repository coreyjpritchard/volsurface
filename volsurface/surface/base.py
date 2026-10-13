"""Mixin that turns a normalised call into implied vol, density, and local variance.

A surface overrides any of these when it has an analytic form. The defaults are
central differences with step ``1e-4`` in ``k`` and in ``T``. That step is a
choice of the mixin, not a property of the interpolator, which is why SVI and
the spline do not use it.
"""

from __future__ import annotations

import numpy as np

from volsurface import bs


def as_axes(T, k) -> tuple[np.ndarray, np.ndarray]:
    """``T`` shaped ``(m,)`` and ``k`` shaped ``(n,)``. Scalars become length 1."""
    T = np.atleast_1d(np.asarray(T, dtype=float))
    k = np.atleast_1d(np.asarray(k, dtype=float))
    if T.ndim != 1 or k.ndim != 1:
        raise ValueError(f"T and k must be 1-d, got shapes {T.shape} and {k.shape}")
    return T, k


class SurfaceMixin:
    """Derivatives and transforms of ``call``. Subclasses set ``market`` and ``domain``."""

    def extrapolated(self, T, k) -> np.ndarray:
        """True outside the quoted rectangle. Subclasses refine this per expiry."""
        T, k = as_axes(T, k)
        domain = self.domain
        t_out = (T < domain.t_min) | (T > domain.t_max)
        k_out = (k < domain.k_min) | (k > domain.k_max)
        return t_out[:, None] | k_out[None, :]

    def iv(self, T, k) -> np.ndarray:
        """Black implied volatility of the normalised call. NaN if it does not exist."""
        T, k = as_axes(T, k)
        normalised = self.call(T, k)
        F = np.asarray(self.market.F(T), dtype=float).reshape(T.shape)
        df = np.asarray(self.market.df(T), dtype=float).reshape(T.shape)
        cash = normalised * df[:, None] * F[:, None]
        strikes = F[:, None] * np.exp(k)[None, :]
        return bs.implied_vol(cash, F[:, None], strikes, T[:, None], df[:, None], "call")

    def total_variance(self, T, k) -> np.ndarray:
        """``iv(T, k)^2 * T``. NaN where the implied vol is."""
        T, k = as_axes(T, k)
        return self.iv(T, k) ** 2 * T[:, None]

    def dcall_dk(self, T, k, h: float = 1e-4) -> np.ndarray:
        """Central difference of ``call`` in ``k``."""
        return (self.call(T, k + h) - self.call(T, k - h)) / (2.0 * h)

    def d2call_dk2(self, T, k, h: float = 1e-4) -> np.ndarray:
        """Central second difference of ``call`` in ``k``."""
        return (self.call(T, k + h) - 2.0 * self.call(T, k) + self.call(T, k - h)) / h**2

    def dcall_dT(self, T, k, h: float = 1e-4) -> np.ndarray:
        """Central difference of ``call`` in ``T``. ``T`` is kept positive."""
        T = np.atleast_1d(np.asarray(T, dtype=float))
        up = np.maximum(T + h, h)
        down = np.maximum(T - h, h * 0.5)
        return (self.call(up, k) - self.call(down, k)) / (up - down)[:, None]

    def density(self, T, k) -> np.ndarray:
        """Density of ``log(F_T / F)`` from the call's strike derivatives.

        ``phi(k) = exp(-k) * (d²C/dk² - dC/dk)``. A negative number is a
        butterfly arbitrage. The finite-difference default is noisy; surfaces
        with an analytic ``g`` override this.
        """
        _, k_axis = as_axes(T, k)
        gap = self.d2call_dk2(T, k) - self.dcall_dk(T, k)
        return np.exp(-k_axis)[None, :] * gap

    def local_var(self, T, k, h: float = 1e-4) -> np.ndarray:
        """Dupire local variance ``(dw/dT) / g(k)``, estimated by differences.

        Returned as a variance, so a negative value is kept. NaN where ``g``
        is too small to divide by or ``w`` is not positive. Analytic surfaces
        override this.
        """
        T, k_axis = as_axes(T, k)
        w = self.total_variance(T, k_axis)
        w_up = self.total_variance(T + h, k_axis)
        w_down = self.total_variance(np.maximum(T - h, h * 0.5), k_axis)
        w_k = (self.total_variance(T, k_axis + h) - self.total_variance(T, k_axis - h)) / (2 * h)
        w_kk = (
            self.total_variance(T, k_axis + h)
            - 2.0 * w
            + self.total_variance(T, k_axis - h)
        ) / h**2
        # dw/dT at fixed k. The down step may have been clipped near 0.
        dt = (T + h) - np.maximum(T - h, h * 0.5)
        w_T = (w_up - w_down) / dt[:, None]
        from volsurface.surface.black import durrleman_g

        g = durrleman_g(k_axis[None, :], w, w_k, w_kk)
        with np.errstate(divide="ignore", invalid="ignore"):
            local = w_T / g
        return np.where((w > 0.0) & (np.abs(g) > 1e-10), local, np.nan)
