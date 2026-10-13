"""Central finite differences of ``price``.

Order 1 is ``(f(x+h) - f(x-h)) / (2h)``. Order 2 is
``(f(x+h) - 2 f(x) + f(x-h)) / h²``. Above the round-off floor the order-1
error falls like ``h²``: halving the bump cuts the error by about four.

``crn=True`` reuses the seed on every bumped price, so a Monte Carlo model
sees the same normals and the bump mostly cancels. ``crn=False`` gives each
bumped price its own child of the seed. A closed form ignores the seed.

The standard error of the difference is not the root-sum-square of the two
price errors when the seeds match, so this engine reports ``stderr=None``.
A repeated run is how the variance reduction is measured.

Digitals are not supported. ``price`` covers vanillas. A digital's price is
not that function, and a silent vanilla bump would be the wrong contract.
"""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np

from volsurface.core.types import Contract, Forward, GreeksResult, Param, Spot, State, Vanilla
from volsurface.mc.rng import spawn


class FiniteDifference:
    """Bump ``bump`` in the forward, or in a named parameter. See the module note."""

    def __init__(self, F: float, df: float = 1.0, bump: float = 1e-4, crn: bool = True):
        if bump <= 0.0:
            raise ValueError(f"bump must be positive, got {bump}")
        self.F = float(F)
        self.df = float(df)
        self.bump = float(bump)
        self.crn = bool(crn)

    def supports(self, model, contract, order: int) -> bool:
        """Orders 1 and 2 for vanillas, on any model that implements ``price``."""
        return (
            order in (1, 2)
            and isinstance(contract, Contract)
            and isinstance(contract.payoff, Vanilla)
            and callable(getattr(model, "price", None))
        )

    def compute(
        self, model, contracts, wrt, *, order: int = 1, state: State | None = None,
        n_paths: int | None = None, seed: int | np.random.SeedSequence | None = None,
    ) -> GreeksResult:
        started = time.perf_counter()
        if order not in (1, 2):
            raise ValueError(f"order must be 1 or 2, got {order}")
        contracts = list(contracts)
        factors = list(wrt)
        if seed is None:
            seed = getattr(model, "seed", 0)
        value = np.empty((len(contracts), len(factors)), dtype=float)
        n_evals = 0
        for j, factor in enumerate(factors):
            column, evals = _column(
                self, model, contracts, factor, order, state, n_paths, seed,
            )
            value[:, j] = column
            n_evals += evals
        return GreeksResult(
            value=value, stderr=None, n_evals=n_evals,
            wall_time=time.perf_counter() - started, method="finite_difference",
        )


def _column(engine, model, contracts, factor, order, state, n_paths, seed):
    for contract in contracts:
        if not engine.supports(model, contract, order):
            raise ValueError(f"finite differences do not support {contract} at order {order}")
    groups: dict[float, list[int]] = {}
    for i, contract in enumerate(contracts):
        groups.setdefault(float(contract.T), []).append(i)
    out = np.empty(len(contracts), dtype=float)
    evals = 0
    for maturity, indices in groups.items():
        strikes = np.array([float(contracts[i].payoff.K) for i in indices])
        kinds = [contracts[i].payoff.kind for i in indices]
        kind = kinds[0] if len(set(kinds)) == 1 else kinds
        levels, used = _levels(
            engine, model, strikes, kind, maturity, factor, order, state, n_paths, seed,
        )
        evals += used
        h = engine.bump
        if order == 1:
            greek = (levels["up"] - levels["down"]) / (2.0 * h)
        else:
            greek = (levels["up"] - 2.0 * levels["mid"] + levels["down"]) / h**2
        flat = np.asarray(greek, dtype=float).reshape(-1)
        for slot, index in enumerate(indices):
            out[index] = flat[slot]
    return out, evals


def _levels(engine, model, strikes, kind, maturity, factor, order, state, n_paths, seed):
    slots = ("up", "down", "mid") if order == 2 else ("up", "down")
    seeds = _bump_seeds(seed, engine.crn, len(slots))
    levels = {}
    for slot, bump_seed in zip(slots, seeds, strict=True):
        bumped, forward = _bumped(engine, model, factor, slot)
        levels[slot] = _price(
            bumped, forward, strikes, kind, maturity, engine.df, state, n_paths, bump_seed,
        )
    per_price = _one_eval(model, n_paths)
    return levels, per_price * len(slots)


def _bumped(engine, model, factor, slot: str):
    """Model and forward at this node of the stencil."""
    sign = {"up": 1.0, "down": -1.0, "mid": 0.0}[slot]
    if isinstance(factor, (Spot, Forward)):
        return model, engine.F + sign * engine.bump
    if isinstance(factor, Param):
        if not hasattr(model, factor.name):
            raise ValueError(f"model has no parameter {factor.name!r}")
        current = float(getattr(model, factor.name))
        return replace(model, **{factor.name: current + sign * engine.bump}), engine.F
    raise ValueError(f"unknown risk factor {factor!r}")


def _price(model, forward, strikes, kind, maturity, df, state, n_paths, seed):
    """Cash prices. A simulator is called through ``price_with_error`` so the seed is honoured."""
    if hasattr(model, "simulate") and hasattr(model, "price_with_error"):
        kwargs = {"state": state, "seed": seed}
        if n_paths is not None:
            kwargs["n_paths"] = n_paths
        result = model.price_with_error(forward, strikes, maturity, df, kind, **kwargs)
        return np.asarray(result.value, dtype=float)
    return np.asarray(model.price(forward, strikes, maturity, df, kind, state=state), dtype=float)


def _one_eval(model, n_paths: int | None) -> int:
    """Paths in one price, or 1 for a closed form."""
    if not hasattr(model, "simulate"):
        return 1
    stored = int(model.n_paths if n_paths is None else n_paths)
    if getattr(model, "antithetic", False):
        stored *= 2
    return stored


def _bump_seeds(seed, crn: bool, n: int):
    """``n`` seeds. Common random numbers repeat the parent. Otherwise, one child each."""
    if crn:
        return [seed] * n
    return spawn(seed, n)
