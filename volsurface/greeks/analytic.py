"""Closed-form Black-Scholes greeks for vanillas and cash-or-nothing digitals.

The forward is the coordinate. With the discount factor held fixed, a bump of
spot that moves the forward one for one is the same derivative, so ``Spot`` and
``Forward`` return the same number. Phase 1 does not bump spot while the
forward moves by a rate spread.

A vanilla uses :mod:`volsurface.bs`. A cash-or-nothing call pays 1 when
``F_T > K`` and is worth ``df * N(d2)``. A put pays 1 when ``F_T < K`` and is
worth ``df * N(-d2)``. Differentiating the put is the negative of
differentiating the call, because the two prices add up to ``df``.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.stats import norm

from volsurface import bs
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
from volsurface.models.bs import BlackScholes


class Analytic:
    """Black-Scholes greeks. ``F`` is the forward. ``df`` discounts every contract."""

    def __init__(self, F: float, df: float = 1.0):
        self.F = float(F)
        self.df = float(df)

    def supports(self, model, contract, order: int) -> bool:
        """Orders 1 and 2, vanillas and digitals, and only a Black-Scholes model."""
        return (
            isinstance(model, BlackScholes)
            and order in (1, 2)
            and isinstance(contract, Contract)
            and isinstance(contract.payoff, (Vanilla, Digital))
        )

    def compute(
        self, model, contracts, wrt, *, order: int = 1, state: State | None = None,
        n_paths: int | None = None, seed: int | np.random.SeedSequence | None = None,
    ) -> GreeksResult:
        """One closed form per contract and factor. ``n_paths`` and ``seed`` are unused."""
        del n_paths, seed
        started = time.perf_counter()
        if state:
            raise ValueError("BlackScholes has an empty state")
        contracts = list(contracts)
        factors = list(wrt)
        value = np.empty((len(contracts), len(factors)), dtype=float)
        for i, contract in enumerate(contracts):
            if not self.supports(model, contract, order):
                raise ValueError(f"analytic engine does not support {contract} at order {order}")
            for j, factor in enumerate(factors):
                value[i, j] = _greek(self.F, self.df, model.sigma, contract, factor, order)
        return GreeksResult(
            value=value, stderr=None, n_evals=len(contracts) * len(factors),
            wall_time=time.perf_counter() - started, method="analytic",
        )


def _greek(F: float, df: float, sigma: float, contract: Contract, factor, order: int) -> float:
    T = float(contract.T)
    K = float(contract.payoff.K)
    if isinstance(factor, Param):
        if factor.name != "sigma":
            raise ValueError(f"analytic engine has no parameter {factor.name!r}")
        kind = "vega"
    elif isinstance(factor, (Spot, Forward)):
        kind = "delta" if order == 1 else "gamma"
    else:
        raise ValueError(f"unknown risk factor {factor!r}")
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order}")
    if isinstance(factor, Param) and order == 2:
        kind = "volga"
    if isinstance(contract.payoff, Vanilla):
        return _vanilla(F, K, T, sigma, df, contract.payoff.kind, kind)
    sign = 1.0 if contract.payoff.kind == "call" else -1.0
    return sign * _digital_call(F, K, T, sigma, df, kind)


def _vanilla(F, K, T, sigma, df, kind, greek) -> float:
    if greek == "delta":
        return float(bs.delta(F, K, T, sigma, df, kind))
    if greek == "gamma":
        return float(bs.gamma(F, K, T, sigma, df))
    if greek == "vega":
        return float(bs.vega(F, K, T, sigma, df))
    # d²price/d sigma². Vega is df F n(d1) sqrt(T), and n'(d1) = -d1 n(d1).
    s = sigma * np.sqrt(T)
    d1 = np.log(F / K) / s + 0.5 * s
    d1_sigma = np.sqrt(T) * (0.5 - np.log(F / K) / s**2)
    return float(bs.vega(F, K, T, sigma, df)) * (-d1) * d1_sigma


def _digital_call(F, K, T, sigma, df, greek) -> float:
    """Cash-or-nothing call. ``s = sigma sqrt(T)``, ``d2 = log(F/K)/s - s/2``.

    Price ``df N(d2)``. Delta ``df n(d2) / (F s)``. Gamma follows by
    differentiating that, using ``n'(d2) = -d2 n(d2)``. Vega differentiates
    ``d2`` in ``sigma``. Volga differentiates the vega again.
    """
    s = sigma * np.sqrt(T)
    d2 = np.log(F / K) / s - 0.5 * s
    n = float(norm.pdf(d2))
    if greek == "delta":
        return df * n / (F * s)
    if greek == "gamma":
        # d/dF of n(d2)/(F s), with d(d2)/dF = 1/(F s).
        return -df * n / (F * F * s) * (d2 / s + 1.0)
    log_m = np.log(F / K)
    d2_sigma = -np.sqrt(T) * (log_m / s**2 + 0.5)
    if greek == "vega":
        return df * n * d2_sigma
    # d²d2/d sigma² = 2 T log(F/K) / s³, and n' contributes -d2 (d2_sigma)².
    d2_sigma2 = 2.0 * T * log_m / s**3
    return df * n * (-d2 * d2_sigma**2 + d2_sigma2)
