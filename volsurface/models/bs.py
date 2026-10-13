"""Black-Scholes as a :class:`~volsurface.core.protocols.Model`.

The formulas live in :mod:`volsurface.bs`. This class is the frozen parameter
``sigma`` plus the protocol methods: ``price``, ``initial_state``, the
characteristic function, and the unconstrained coordinate used by a fitter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from volsurface import bs
from volsurface.core.params import from_unconstrained_value, to_unconstrained_value
from volsurface.core.types import ParamSpec, State
from volsurface.plugins import register


def _spec() -> list[ParamSpec]:
    # sigma is positive. The log coordinate is the one a least-squares fitter steps in.
    return [ParamSpec("sigma", lower=1e-4, upper=5.0, default=0.2, transform="log")]


@register("black_scholes")
@dataclass(frozen=True)
class BlackScholes:
    """Constant volatility. The implied-vol surface is flat at ``sigma``.

    The state is empty: there is no latent process. ``state=None`` and
    ``state={}`` price the same contract.
    """

    sigma: float

    def initial_state(self) -> State:
        return {}

    def price(self, F, K, T, df=1.0, kind="call", *, state: State | None = None):
        """Cash Black price. ``state`` is accepted and ignored (the state is empty)."""
        if state:
            raise ValueError(f"BlackScholes has an empty state, got keys {sorted(state)}")
        return bs.price(F, K, T, self.sigma, df, kind)

    def char_func(self, u, T: float, *, state: State | None = None) -> np.ndarray:
        """``E[exp(i u X_T)]`` for ``X_T = log(F_T / F) ~ N(-w/2, w)``, ``w = sigma^2 T``.

        Completing the normal integral gives ``exp(-i u w / 2 - u^2 w / 2)``.
        """
        if state:
            raise ValueError(f"BlackScholes has an empty state, got keys {sorted(state)}")
        u = np.asarray(u, dtype=complex)
        w = self.sigma**2 * float(T)
        return np.exp(-0.5j * u * w - 0.5 * u * u * w)

    def cumulants(self, T: float, *, state: State | None = None) -> tuple[float, float, float]:
        """``(c1, c2, c4)`` of ``log(F_T / F)``. The fourth cumulant of a normal is 0."""
        if state:
            raise ValueError(f"BlackScholes has an empty state, got keys {sorted(state)}")
        w = self.sigma**2 * float(T)
        return (-0.5 * w, w, 0.0)

    @classmethod
    def param_spec(cls) -> list[ParamSpec]:
        return _spec()

    def to_unconstrained(self) -> np.ndarray:
        return np.array([to_unconstrained_value(self.sigma, _spec()[0])], dtype=float)

    @classmethod
    def from_unconstrained(cls, x: np.ndarray) -> BlackScholes:
        raw = float(np.asarray(x, dtype=float).reshape(-1)[0])
        return cls(sigma=from_unconstrained_value(raw, _spec()[0]))
