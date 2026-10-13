"""A surface whose normalised call is a model's cash price, divided by ``df F``."""

from __future__ import annotations

import numpy as np

from volsurface.core.protocols import Model
from volsurface.core.types import Curves, Domain
from volsurface.surface.base import SurfaceMixin, as_axes


class ModelSurface(SurfaceMixin):
    """Sample ``model.price`` on the normalised-call grid.

    Derivatives fall through to the mixin differences. A model with a closed
    form is smoother than those differences, so prefer the model's own greeks
    when the comparison is about greeks rather than about interpolation.
    """

    def __init__(self, model: Model, market: Curves, domain: Domain):
        self.model = model
        self.market = market
        self.domain = domain

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
