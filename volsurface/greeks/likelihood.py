"""Likelihood-ratio greeks for exact geometric Brownian motion.

The payoff is left alone. The derivative falls on the lognormal density of
the terminal forward ``Y = F Z``, and the estimator multiplies the payoff by
that score.

Write ``w = sigma² T`` and ``u = log Z + w/2``. Holding ``Y`` fixed,

    score for F     = u / (F w),
    score for sigma = -(1 + u) / sigma + u² / (sigma w).

A second derivative uses ``score² + d(score)/d(parameter)``, which is
``(1/p) d²p``. For the forward that weight is

    u² / (F² w²) - (u + 1) / (F² w).

For sigma, with ``x = log Z``,

    T² sigma² / 16 + T / 4 - x² / (2 sigma²) + 2 / sigma²
        - 5 x² / (T sigma⁴) + x⁴ / (T² sigma⁶).

The score for sigma divides by ``sigma`` and by ``w``, so a short expiry has
a large weight and a large standard error. That comparison is a test, not a
failure of the estimator.

The same weights multiply a digital's indicator. Pathwise differentiation
cannot see that indicator. Euler paths are not lognormal, so this engine
refuses them.
"""

from __future__ import annotations

import time

import numpy as np

from volsurface.core.types import (
    Contract,
    Digital,
    Forward,
    GreeksResult,
    Param,
    Spot,
    State,
    Vanilla,
)
from volsurface.greeks.pathwise import _reduce
from volsurface.models.gbm import GBM


class LikelihoodRatio:
    """See the module note. ``F`` is the forward inside the density and the payoff."""

    def __init__(self, F: float, df: float = 1.0):
        self.F = float(F)
        self.df = float(df)

    def supports(self, model, contract, order: int) -> bool:
        """Orders 1 and 2, exact GBM, vanilla or cash-or-nothing digital."""
        return (
            isinstance(model, GBM)
            and model.scheme == "exact"
            and order in (1, 2)
            and isinstance(contract, Contract)
            and isinstance(contract.payoff, (Vanilla, Digital))
        )

    def compute(
        self, model, contracts, wrt, *, order: int = 1, state: State | None = None,
        n_paths: int | None = None, seed: int | np.random.SeedSequence | None = None,
    ) -> GreeksResult:
        started = time.perf_counter()
        contracts = list(contracts)
        factors = list(wrt)
        for contract in contracts:
            if not self.supports(model, contract, order):
                raise ValueError("likelihood ratio supports exact GBM vanillas and digitals")
        n_paths = int(model.n_paths if n_paths is None else n_paths)
        seed = model.seed if seed is None else seed
        value = np.empty((len(contracts), len(factors)), dtype=float)
        stderr = np.empty_like(value)
        evals = 0
        groups: dict[float, list[int]] = {}
        for i, contract in enumerate(contracts):
            groups.setdefault(float(contract.T), []).append(i)
        for maturity, indices in groups.items():
            times = np.linspace(0.0, maturity, model.n_steps + 1)
            paths = model.simulate(n_paths, times, seed=seed, state0=state)
            z = paths.Z[:, -1]
            evals += int(z.size)
            for j, factor in enumerate(factors):
                samples = _samples(
                    z, self.F, self.df, contracts, indices, model.sigma, maturity, factor, order,
                )
                mean, error = _reduce(samples, model.antithetic)
                for slot, index in enumerate(indices):
                    value[index, j] = mean[slot]
                    stderr[index, j] = error[slot]
        return GreeksResult(
            value=value, stderr=stderr, n_evals=evals,
            wall_time=time.perf_counter() - started, method="likelihood_ratio",
        )


def _samples(z, F, df, contracts, indices, sigma, maturity, factor, order) -> np.ndarray:
    w = sigma**2 * maturity
    log_z = np.log(np.maximum(z, 1e-300))
    u = log_z + 0.5 * w
    weight = _weight(u, log_z, F, sigma, maturity, w, factor, order)
    payoff = np.empty((z.size, len(indices)), dtype=float)
    for slot, index in enumerate(indices):
        payoff[:, slot] = _payoff(z, F, float(contracts[index].payoff.K), contracts[index].payoff)
    return df * payoff * weight[:, None]


def _payoff(z, F, strike, payoff) -> np.ndarray:
    """Undiscounted vanilla or digital. The score carries the derivative."""
    forward = F * z
    call = getattr(payoff, "kind", "call") == "call"
    if isinstance(payoff, Vanilla):
        if call:
            return np.maximum(forward - strike, 0.0)
        return np.maximum(strike - forward, 0.0)
    if isinstance(payoff, Digital):
        if call:
            return (forward > strike).astype(float)
        return (forward < strike).astype(float)
    raise ValueError(f"no likelihood-ratio payoff for {payoff!r}")


def _weight(u, log_z, F, sigma, maturity, w, factor, order) -> np.ndarray:
    if isinstance(factor, (Spot, Forward)):
        score = u / (F * w)
        if order == 1:
            return score
        return score**2 - (u + 1.0) / (F * F * w)
    if isinstance(factor, Param) and factor.name == "sigma":
        score = -(1.0 + u) / sigma + u**2 / (sigma * w)
        if order == 1:
            return score
        # (1/p) d²p/d sigma², in log Z. See the module note.
        x = log_z
        return (
            maturity**2 * sigma**2 / 16.0
            + maturity / 4.0
            - x**2 / (2.0 * sigma**2)
            + 2.0 / sigma**2
            - 5.0 * x**2 / (maturity * sigma**4)
            + x**4 / (maturity**2 * sigma**6)
        )
    raise ValueError(f"likelihood-ratio engine has no factor {factor!r}")
