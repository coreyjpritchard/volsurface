"""Torch autodiff of the same exact simulation the pathwise engine uses.

The normals are drawn by the same child seed, in the same order: one vector
per time step, path index fastest, the antithetic mirror in the second half.
``sigma`` and ``F`` are float64 tensors. The payoff is ``df * max(F Z - K, 0)``
and ``backward`` differentiates it. Because this is the pathwise calculation
done by the library instead of by hand, the two averages agree to about
1e-10 in float64.

A kink is still a kink. The subgradient of ``max`` is zero at the strike and
does not produce a gamma or a digital. ``supports`` agrees with the pathwise
engine: exact GBM, vanilla, order 1, and no control variate.

Torch is not imported with the package. ``compute`` imports it, and the
install extra is ``autodiff``.
"""

from __future__ import annotations

import time

import numpy as np

from volsurface.core.types import Contract, Forward, GreeksResult, Param, Spot, State, Vanilla
from volsurface.greeks.pathwise import _reduce, _refuse_control, _samples
from volsurface.mc.rng import generator, spawn
from volsurface.models.gbm import GBM


class Autodiff:
    """See the module note. Float64 throughout."""

    def __init__(self, F: float, df: float = 1.0):
        self.F = float(F)
        self.df = float(df)

    def supports(self, model, contract, order: int) -> bool:
        """Same contracts as the pathwise engine. Autodiff does not repair a kink."""
        return (
            isinstance(model, GBM)
            and model.scheme == "exact"
            and not bool(getattr(model, "control_variate", False))
            and order == 1
            and isinstance(contract, Contract)
            and isinstance(contract.payoff, Vanilla)
        )

    def compute(
        self, model, contracts, wrt, *, order: int = 1, state: State | None = None,
        n_paths: int | None = None, seed: int | np.random.SeedSequence | None = None,
    ) -> GreeksResult:
        started = time.perf_counter()
        if state:
            raise ValueError("GBM has an empty state")
        _refuse_control(model, "autodiff")
        contracts = list(contracts)
        factors = list(wrt)
        for contract in contracts:
            if not self.supports(model, contract, order):
                raise ValueError("autodiff supports an exact GBM vanilla at order 1 only")
        torch = _torch()
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
            sigma = torch.tensor(model.sigma, dtype=torch.float64, requires_grad=True)
            forward = torch.tensor(self.F, dtype=torch.float64, requires_grad=True)
            z = _terminal(sigma, n_paths, times, seed, model.antithetic)
            evals += int(z.shape[0])
            strikes = [float(contracts[i].payoff.K) for i in indices]
            calls = [contracts[i].payoff.kind == "call" for i in indices]
            # The standard error is the pathwise one on these paths. The value
            # is the torch gradient of the same payoff. They are one estimator.
            z_np = z.detach().cpu().numpy()
            for j, factor in enumerate(factors):
                column = _grad_column(z, forward, sigma, strikes, calls, self.df, factor)
                samples = _samples(
                    z_np, self.F, self.df,
                    np.array(strikes), np.array(calls), model.sigma, maturity, factor,
                )
                _, error = _reduce(samples, model.antithetic)
                for slot, index in enumerate(indices):
                    value[index, j] = column[slot]
                    stderr[index, j] = error[slot]
        return GreeksResult(
            value=value, stderr=stderr, n_evals=evals,
            wall_time=time.perf_counter() - started, method="autodiff",
        )


def _torch():
    """Import torch at call time. The package import does not pull it in."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "autodiff requires torch; install with pip install 'volsurface[autodiff]'"
        ) from exc
    return torch


def _terminal(sigma, n_paths: int, times: np.ndarray, seed, antithetic: bool):
    """Exact log update, normals from ``spawn(seed, 1)``, matching :class:`GBM`."""
    torch = _torch()
    rng = generator(spawn(seed, 1)[0])
    rows = n_paths * (2 if antithetic else 1)
    log_z = torch.zeros(rows, dtype=torch.float64)
    for i in range(1, times.size):
        dt = float(times[i] - times[i - 1])
        g = rng.standard_normal(n_paths)
        shock = np.concatenate([g, -g]) if antithetic else g
        shock_t = torch.as_tensor(shock, dtype=torch.float64)
        log_z = log_z - 0.5 * sigma**2 * dt + sigma * float(np.sqrt(dt)) * shock_t
    return torch.exp(log_z)


def _grad_column(z, forward, sigma, strikes, calls, df, factor) -> list[float]:
    """One torch gradient per strike. The graph is kept until the last strike."""
    torch = _torch()
    out = []
    target = forward if isinstance(factor, (Spot, Forward)) else sigma
    if isinstance(factor, Param) and factor.name != "sigma":
        raise ValueError(f"autodiff has no parameter {factor.name!r}")
    if not isinstance(factor, (Spot, Forward, Param)):
        raise ValueError(f"unknown risk factor {factor!r}")
    for strike, call in zip(strikes, calls, strict=True):
        gap = forward * z - strike
        payoff = torch.clamp(gap, min=0.0) if call else torch.clamp(-gap, min=0.0)
        mean = df * payoff.mean()
        (grad,) = torch.autograd.grad(mean, target, retain_graph=True)
        out.append(float(grad))
    return out
