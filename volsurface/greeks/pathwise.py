"""Pathwise greeks for an exact geometric Brownian motion.

The normals are held fixed and the discounted payoff is differentiated.
For a vanilla call the forward delta on a path is

    df * Z * 1_{F Z > K},

because the cash payoff is ``df (F Z - K)+`` and ``Z`` does not depend on
``F``. The vega differentiates ``Z``. Under the exact scheme

    log Z_T = -sigma² T / 2 + sigma G,    Var(G) = T,

so

    dZ/d sigma = Z (log Z / sigma - sigma T / 2),

whether ``G`` was drawn in one step or as a sum of steps. The estimator is
the average of those path derivatives. It is unbiased for a Lipschitz payoff.

A digital and a gamma both differentiate a kink. The pathwise derivative is
zero almost surely and wrong as a greek. ``supports`` returns false for a
digital and for order 2, and ``compute`` refuses them. Returning a silent
zero would be a wrong greek.

Euler paths are a different function of the normals. This engine does not
differentiate them. A control variate changes the price after the paths are
drawn. This engine differentiates the paths, so a model with the control on
is refused rather than given a greek whose standard error is not the price's.
"""

from __future__ import annotations

import time

import numpy as np

from volsurface.core.types import Contract, Forward, GreeksResult, Param, Spot, State, Vanilla
from volsurface.models.gbm import GBM


class Pathwise:
    """See the module note. ``F`` is the forward the payoff is written on."""

    def __init__(self, F: float, df: float = 1.0):
        self.F = float(F)
        self.df = float(df)

    def supports(self, model, contract, order: int) -> bool:
        """Order 1, exact GBM, vanilla. Not a digital, not a gamma, not Euler.

        A control variate is refused. The greek would not be the derivative
        of the controlled price.
        """
        return (
            isinstance(model, GBM)
            and model.scheme == "exact"
            and not _controlled(model)
            and order == 1
            and isinstance(contract, Contract)
            and isinstance(contract.payoff, Vanilla)
        )

    def compute(
        self, model, contracts, wrt, *, order: int = 1, state: State | None = None,
        n_paths: int | None = None, seed: int | np.random.SeedSequence | None = None,
    ) -> GreeksResult:
        started = time.perf_counter()
        contracts = list(contracts)
        factors = list(wrt)
        _refuse_control(model, "pathwise")
        for contract in contracts:
            if not self.supports(model, contract, order):
                raise ValueError(
                    "pathwise greeks support an exact GBM vanilla at order 1 only"
                )
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
            strikes = np.array([float(contracts[i].payoff.K) for i in indices])
            call = np.array([contracts[i].payoff.kind == "call" for i in indices])
            for j, factor in enumerate(factors):
                samples = _samples(z, self.F, self.df, strikes, call, model.sigma, maturity, factor)
                mean, error = _reduce(samples, model.antithetic)
                for slot, index in enumerate(indices):
                    value[index, j] = mean[slot]
                    stderr[index, j] = error[slot]
        return GreeksResult(
            value=value, stderr=stderr, n_evals=evals,
            wall_time=time.perf_counter() - started, method="pathwise",
        )


def _controlled(model) -> bool:
    """True when the model prices with a control variate this engine does not use."""
    return bool(getattr(model, "control_variate", False))


def _refuse_control(model, engine: str) -> None:
    if _controlled(model):
        raise ValueError(
            f"{engine} refuses a model with the control variate on: "
            "it differentiates the paths, not the controlled price"
        )


def _samples(z, F, df, strikes, call, sigma, maturity, factor) -> np.ndarray:
    """Path derivatives, shape ``(n_stored, n_strikes)``.

    A call contributes ``+d(F Z)/d(parameter)`` on ``{F Z > K}``. A put
    contributes the negative on ``{F Z < K}``, because the put payoff is
    ``df (K - F Z)+``.
    """
    z_col = z[:, None]
    call_ind = F * z_col > strikes[None, :]
    put_ind = F * z_col < strikes[None, :]
    active = np.where(call[None, :], call_ind, put_ind)
    sign = np.where(call[None, :], 1.0, -1.0)
    if isinstance(factor, (Spot, Forward)):
        return df * sign * z_col * active
    if isinstance(factor, Param) and factor.name == "sigma":
        dlog = np.log(np.maximum(z, 1e-300)) / sigma - 0.5 * sigma * maturity
        return df * sign * F * (z * dlog)[:, None] * active
    raise ValueError(f"pathwise engine has no factor {factor!r}")


def _reduce(samples: np.ndarray, antithetic: bool) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard error. Antithetic pairs are averaged before the division."""
    if antithetic:
        half = samples.shape[0] // 2
        samples = 0.5 * (samples[:half] + samples[half:])
    mean = samples.mean(axis=0)
    stderr = samples.std(axis=0, ddof=1) / np.sqrt(samples.shape[0])
    return mean, stderr
