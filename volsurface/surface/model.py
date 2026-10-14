"""A surface whose normalised call is a model's cash price, divided by ``df F``."""

from __future__ import annotations

import numpy as np

from volsurface.core.protocols import Model
from volsurface.core.types import Curves, Domain
from volsurface.models import BlackScholes
from volsurface.surface.base import SurfaceMixin, as_axes
from volsurface.surface.black import d2call_dk2_from_w, dcall_dk_from_w, dcall_dw


class ModelSurface(SurfaceMixin):
    """Sample ``model.price`` on the normalised-call grid.

    Black-Scholes derivatives are the flat-smile formulas (total variance
    ``sigma^2 T``, no strike dependence). Any other model uses the mixin
    differences, and ``finite_difference_gamma`` tells the checker to floor
    the butterfly tolerance. A comparison that is about greeks, rather than
    about interpolation, should still call the model's own greek engine.
    """

    def __init__(self, model: Model, market: Curves, domain: Domain):
        self.model = model
        self.market = market
        self.domain = domain
        self.finite_difference_gamma = not isinstance(model, BlackScholes)

    def _black_total_variance(self, T, k):
        """``(T, k, w)`` for a flat smile. ``w`` has shape ``(m, n)``."""
        T, k = as_axes(T, k)
        sigma = float(self.model.sigma)
        w = np.full((T.size, k.size), sigma**2) * T[:, None]
        return T, k, w

    def dcall_dk(self, T, k, h: float = 1e-4) -> np.ndarray:
        if not isinstance(self.model, BlackScholes):
            return super().dcall_dk(T, k, h)
        _, k_axis, w = self._black_total_variance(T, k)
        return dcall_dk_from_w(k_axis[None, :], w, np.zeros_like(w))

    def d2call_dk2(self, T, k, h: float = 1e-3) -> np.ndarray:
        if not isinstance(self.model, BlackScholes):
            return super().d2call_dk2(T, k, h)
        _, k_axis, w = self._black_total_variance(T, k)
        zeros = np.zeros_like(w)
        return d2call_dk2_from_w(k_axis[None, :], w, zeros, zeros)

    def dcall_dT(self, T, k, h: float = 1e-4) -> np.ndarray:
        if not isinstance(self.model, BlackScholes):
            return super().dcall_dT(T, k, h)
        _, k_axis, w = self._black_total_variance(T, k)
        # dw/dT = sigma^2 at fixed k. dC/dT = dC/dw * sigma^2.
        return dcall_dw(k_axis[None, :], w) * float(self.model.sigma) ** 2

    def call(self, T, k) -> np.ndarray:
        T, k = as_axes(T, k)
        out = np.empty((T.size, k.size))
        for i, t in enumerate(T):
            forward = float(np.asarray(self.market.F(t), dtype=float))
            discount = float(np.asarray(self.market.df(t), dtype=float))
            strike = forward * np.exp(k)
            cash = np.asarray(
                self.model.price(forward, strike, float(t), discount, "call"), dtype=float
            )
            out[i] = cash / (discount * forward)
        return out
