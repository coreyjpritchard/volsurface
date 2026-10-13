"""Heston as a :class:`~volsurface.core.protocols.Model`.

Dynamics, under the forward measure::

    dF / F = sqrt(v) dW
    dv    = kappa (theta - v) dt + xi sqrt(v) dZ,    corr(dW, dZ) = rho dt

The characteristic function and the Lewis price are :mod:`volsurface.heston`.
The variance state is the scalar ``v`` that those formulas accept. A vector of
variances, one per path, needs a pricer that can take it; the Fourier formula
here is for one variance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from volsurface import heston as heston_f
from volsurface.core.params import from_unconstrained, to_unconstrained
from volsurface.core.types import ParamSpec, State
from volsurface.plugins import register


def _spec() -> list[ParamSpec]:
    return [
        ParamSpec("v0", lower=1e-6, upper=2.0, default=0.04, transform="log"),
        ParamSpec("kappa", lower=1e-3, upper=20.0, default=1.5, transform="log"),
        ParamSpec("theta", lower=1e-6, upper=2.0, default=0.04, transform="log"),
        ParamSpec("xi", lower=1e-6, upper=5.0, default=0.5, transform="log"),
        ParamSpec("rho", lower=-0.999, upper=0.999, default=-0.7, transform="logit"),
    ]


@register("heston")
@dataclass(frozen=True)
class Heston:
    """Heston (1993). See :mod:`volsurface.heston` for the pricing formula."""

    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float

    def initial_state(self) -> State:
        return {"v": np.array(self.v0)}

    def _variance(self, state: State | None) -> float:
        if state is None or "v" not in state:
            return float(self.v0)
        v = np.asarray(state["v"], dtype=float)
        if v.shape == () or v.size == 1:
            return float(v.reshape(()))
        raise ValueError(
            "Heston.price uses the Fourier formula, which takes one variance. "
            f"Got state['v'] with shape {v.shape}."
        )

    def price(self, F, K, T, df=1.0, kind="call", *, state: State | None = None):
        """Cash Heston price. A scalar ``state['v']`` replaces ``v0`` for this call."""
        v0 = self._variance(state)
        return heston_f.price(F, K, T, v0, self.kappa, self.theta, self.xi, self.rho, df, kind)

    def char_func(self, u, T: float, *, state: State | None = None) -> np.ndarray:
        """``E[exp(i u log(F_T / F))]`` in the trap-free Gatheral form.

        ``u`` may be complex. The branch is the one Albrecher, Mayer, Schoutens
        and Tistaert (2007) show stays continuous: see :func:`volsurface.heston.char_func`.
        """
        v0 = self._variance(state)
        return heston_f.char_func(u, float(T), v0, self.kappa, self.theta, self.xi, self.rho)

    def cumulants(self, T: float, *, state: State | None = None) -> tuple[float, float, float]:
        """``(c1, c2, c4)`` of ``X_T = log(F_T / F)``.

        The variance is an autonomous square-root process, so

            E[v_t] = theta + (v0 - theta) exp(-kappa t),
            c1 = E[X_T] = -1/2 * integral_0^T E[v_t] dt
               = (1 - exp(-kappa T)) (theta - v0) / (2 kappa) - theta T / 2.

        ``c2`` and ``c4`` are derivatives of the cumulant generating function
        ``K(t) = log E[exp(t X_T)] = log phi(-i t)`` at ``t = 0``. They are
        read from a degree-8 polynomial on the stencil ``t = -4h, ..., 4h``
        with ``h = 0.01``, which matches the closed form for ``c1`` and is
        stable for ``c2``. A cosine expansion only needs them to place a grid.
        """
        v0 = self._variance(state)
        kappa, theta = self.kappa, self.theta
        c1 = (1.0 - np.exp(-kappa * T)) * (theta - v0) / (2.0 * kappa) - 0.5 * theta * T
        c2, c4 = _higher_cumulants(self, float(T), v0)
        return (float(c1), c2, c4)

    @classmethod
    def param_spec(cls) -> list[ParamSpec]:
        return _spec()

    def to_unconstrained(self) -> np.ndarray:
        values = {
            "v0": self.v0,
            "kappa": self.kappa,
            "theta": self.theta,
            "xi": self.xi,
            "rho": self.rho,
        }
        return to_unconstrained(values, _spec())

    @classmethod
    def from_unconstrained(cls, x: np.ndarray) -> Heston:
        values = from_unconstrained(np.asarray(x, dtype=float), _spec())
        return cls(**values)


def _higher_cumulants(model: Heston, T: float, v0: float) -> tuple[float, float]:
    """``(c2, c4)`` from a polynomial fit to ``log phi(-i t)`` around 0."""
    h = 0.01
    t = np.arange(-4, 5) * h
    # phi(-i t) = E[exp(t X)]. log of that is the cumulant generating function.
    psi = np.log(model.char_func(-1j * t, T, state={"v": np.array(v0)})).real
    coeff = np.polyfit(t, psi, 8)
    poly = np.poly1d(coeff)
    # The Taylor coefficient of t^n / n! is the nth cumulant, so c_n = d^n K/dt^n at 0.
    c2 = float(poly.deriv(2)(0.0))
    c4 = float(poly.deriv(4)(0.0))
    return c2, c4
