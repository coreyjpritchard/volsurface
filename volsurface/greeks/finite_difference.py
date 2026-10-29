"""Central finite differences of ``price``.

Order 1 is ``(f(x+h) - f(x-h)) / (2h)``. Order 2 is
``(f(x+h) - 2 f(x) + f(x-h)) / h²``. Above the round-off floor the order-1
error falls like ``h²``: halving the bump cuts the error by about four.

``crn=True`` reuses the seed on every bumped price, so a Monte Carlo model
sees the same normals and the bump mostly cancels. ``crn=False`` gives each
bumped price its own child of the seed. A closed form ignores the seed.

On a simulator the standard error is reported. With common random numbers
the greek on each batch is the difference of that batch's discounted
payoffs, antithetic pairs averaged first. The standard error is the sample
standard deviation of those batch greeks, divided by the square root of the
batch count. A forward or spot bump simulates once, because ``Z`` does not
depend on ``F``. A parameter bump resimulates each stencil node on the same
seed. Without common random numbers the standard error is the
root-sum-square of the node price errors: order 1 divides by ``2h``, and
order 2 divides by ``h²`` with the middle node counted twice. A closed form
reports ``stderr=None``.

When ``bump`` is omitted the step is relative to the level being bumped. A
closed form uses ``1e-4`` times the level. A simulator uses ``0.01`` times
the level: a step of ``1e-4`` on a forward near 100 puts a gamma on almost
no paths. The level is ``F`` for spot and forward, and the current value for
a named parameter. Passing ``bump`` sets an absolute step instead.

Digitals are not supported. ``price`` covers vanillas. A digital's price is
not that function, and a silent vanilla bump would be the wrong contract.
"""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np

from volsurface.core.types import Contract, Forward, GreeksResult, Param, Spot, State, Vanilla
from volsurface.mc.grid import _cash_payoff, _control
from volsurface.mc.rng import spawn

# Relative step when the caller does not pass an absolute bump.
_CLOSED_FORM_REL = 1e-4
_SIMULATOR_REL = 0.01


class FiniteDifference:
    """Central bump of ``price``. See the module note for the step and the error."""

    def __init__(
        self,
        F: float,
        df: float = 1.0,
        bump: float | None = None,
        crn: bool = True,
    ):
        if bump is not None and bump <= 0.0:
            raise ValueError(f"bump must be positive, got {bump}")
        self.F = float(F)
        self.df = float(df)
        self.bump = None if bump is None else float(bump)
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
        self,
        model,
        contracts,
        wrt,
        *,
        order: int = 1,
        state: State | None = None,
        n_paths: int | None = None,
        seed: int | np.random.SeedSequence | None = None,
    ) -> GreeksResult:
        started = time.perf_counter()
        if order not in (1, 2):
            raise ValueError(f"order must be 1 or 2, got {order}")
        contracts = list(contracts)
        factors = list(wrt)
        if seed is None:
            seed = getattr(model, "seed", 0)
        value = np.empty((len(contracts), len(factors)), dtype=float)
        stderr = np.empty_like(value)
        saw_error = False
        n_evals = 0
        for j, factor in enumerate(factors):
            column, error, evals = _column(
                self,
                model,
                contracts,
                factor,
                order,
                state,
                n_paths,
                seed,
            )
            value[:, j] = column
            if error is None:
                stderr[:, j] = np.nan
            else:
                stderr[:, j] = error
                saw_error = True
            n_evals += evals
        return GreeksResult(
            value=value,
            stderr=stderr if saw_error else None,
            n_evals=n_evals,
            wall_time=time.perf_counter() - started,
            method="finite_difference",
        )


def _column(engine, model, contracts, factor, order, state, n_paths, seed):
    for contract in contracts:
        if not engine.supports(model, contract, order):
            raise ValueError(f"finite differences do not support {contract} at order {order}")
    groups: dict[float, list[int]] = {}
    for i, contract in enumerate(contracts):
        groups.setdefault(float(contract.T), []).append(i)
    out = np.empty(len(contracts), dtype=float)
    err = np.empty(len(contracts), dtype=float)
    saw_error = False
    evals = 0
    h = _step(engine, model, factor)
    for maturity, indices in groups.items():
        strikes = np.array([float(contracts[i].payoff.K) for i in indices])
        kinds = [contracts[i].payoff.kind for i in indices]
        kind = kinds[0] if len(set(kinds)) == 1 else kinds
        if _path_difference(model, engine.crn):
            greek, error, used = _crn_difference(
                engine,
                model,
                strikes,
                kind,
                maturity,
                factor,
                order,
                h,
                state,
                n_paths,
                seed,
            )
        else:
            greek, error, used = _repriced(
                engine,
                model,
                strikes,
                kind,
                maturity,
                factor,
                order,
                h,
                state,
                n_paths,
                seed,
            )
        evals += used
        flat = np.asarray(greek, dtype=float).reshape(-1)
        for slot, index in enumerate(indices):
            out[index] = flat[slot]
        if error is not None:
            saw_error = True
            err_flat = np.asarray(error, dtype=float).reshape(-1)
            for slot, index in enumerate(indices):
                err[index] = err_flat[slot]
    return out, (err if saw_error else None), evals


def _step(engine, model, factor) -> float:
    """Absolute bump. An explicit ``bump`` wins. Otherwise the step scales with the level."""
    if engine.bump is not None:
        return engine.bump
    if isinstance(factor, (Spot, Forward)):
        level = engine.F
    elif isinstance(factor, Param):
        if not hasattr(model, factor.name):
            raise ValueError(f"model has no parameter {factor.name!r}")
        level = float(getattr(model, factor.name))
    else:
        raise ValueError(f"unknown risk factor {factor!r}")
    rel = _SIMULATOR_REL if _is_simulator(model) else _CLOSED_FORM_REL
    step = rel * abs(level)
    if step <= 0.0:
        raise ValueError(f"relative bump is zero at level {level}")
    return step


def _path_difference(model, crn: bool) -> bool:
    """Per-path differences need a simulator, a step count, and a shared seed."""
    return bool(crn) and _is_simulator(model) and hasattr(model, "n_steps")


def _is_simulator(model) -> bool:
    return hasattr(model, "simulate") and hasattr(model, "price_with_error")


def _crn_difference(
    engine,
    model,
    strikes,
    kind,
    maturity,
    factor,
    order,
    h,
    state,
    n_paths,
    seed,
):
    """Mean and standard error of the per-batch greek. One shared seed."""
    n_paths = int(model.n_paths if n_paths is None else n_paths)
    slots = ("up", "down", "mid") if order == 2 else ("up", "down")
    if isinstance(factor, (Spot, Forward)):
        z = _terminal(model, maturity, n_paths, seed, state)
        payoffs = {
            slot: _batched(z, engine.F + _sign(slot) * h, strikes, engine.df, kind, model)
            for slot in slots
        }
        n_sims = 1
    elif isinstance(factor, Param):
        payoffs = {}
        for slot in slots:
            bumped, forward = _bumped(engine, model, factor, slot, h)
            z = _terminal(bumped, maturity, n_paths, seed, state)
            payoffs[slot] = _batched(z, forward, strikes, engine.df, kind, bumped)
        n_sims = len(slots)
    else:
        raise ValueError(f"unknown risk factor {factor!r}")
    samples = _combine(payoffs, order, h)
    mean = samples.mean(axis=0)
    n_batches = int(samples.shape[0])
    if n_batches < 2:
        error = np.full(np.shape(mean), np.nan)
    else:
        error = samples.std(axis=0, ddof=1) / np.sqrt(n_batches)
    stored = n_paths * (2 if getattr(model, "antithetic", False) else 1)
    return mean, error, stored * n_sims


def _repriced(engine, model, strikes, kind, maturity, factor, order, h, state, n_paths, seed):
    """Difference of prices. Independent seeds when ``crn`` is off and the model simulates."""
    slots = ("up", "down", "mid") if order == 2 else ("up", "down")
    seeds = _bump_seeds(seed, engine.crn, len(slots))
    levels: dict[str, np.ndarray] = {}
    errors: dict[str, np.ndarray] = {}
    saw_error = True
    for slot, bump_seed in zip(slots, seeds, strict=True):
        bumped, forward = _bumped(engine, model, factor, slot, h)
        value, error = _price(
            bumped,
            forward,
            strikes,
            kind,
            maturity,
            engine.df,
            state,
            n_paths,
            bump_seed,
        )
        levels[slot] = value
        if error is None:
            saw_error = False
        else:
            errors[slot] = error
    greek = _combine(levels, order, h)
    stderr = _rss(errors, order, h) if saw_error else None
    return greek, stderr, _one_eval(model, n_paths) * len(slots)


def _combine(levels: dict[str, np.ndarray], order: int, h: float) -> np.ndarray:
    if order == 1:
        return (levels["up"] - levels["down"]) / (2.0 * h)
    return (levels["up"] - 2.0 * levels["mid"] + levels["down"]) / h**2


def _rss(errors: dict[str, np.ndarray], order: int, h: float) -> np.ndarray:
    """Independent node errors. The middle price is multiplied by 2, so its variance by 4."""
    if order == 1:
        return np.sqrt(errors["up"] ** 2 + errors["down"] ** 2) / (2.0 * h)
    return np.sqrt(errors["up"] ** 2 + 4.0 * errors["mid"] ** 2 + errors["down"] ** 2) / h**2


def _sign(slot: str) -> float:
    return {"up": 1.0, "down": -1.0, "mid": 0.0}[slot]


def _bumped(engine, model, factor, slot: str, h: float):
    """Model and forward at this node of the stencil."""
    sign = _sign(slot)
    if isinstance(factor, (Spot, Forward)):
        return model, engine.F + sign * h
    if isinstance(factor, Param):
        if not hasattr(model, factor.name):
            raise ValueError(f"model has no parameter {factor.name!r}")
        current = float(getattr(model, factor.name))
        return replace(model, **{factor.name: current + sign * h}), engine.F
    raise ValueError(f"unknown risk factor {factor!r}")


def _batched(z, forward, strikes, df, kind, model) -> np.ndarray:
    """Discounted payoff per batch, shape ``(n_batches, n_strikes)``.

    This is the reduction inside ``estimate_terminal``: average an antithetic
    pair, then subtract the least-squares control on ``Z`` when the model
    asks for one. Differencing these rows is the same greek as differencing
    the two prices, and the rows are what the standard error is taken from.
    """
    payoff = _cash_payoff(np.asarray(z, dtype=float), float(forward), strikes, float(df), kind)
    z_level = np.asarray(z, dtype=float)
    if getattr(model, "antithetic", False):
        if z_level.size % 2 != 0:
            raise ValueError("antithetic paths must come in pairs")
        half = z_level.size // 2
        payoff = 0.5 * (payoff[:half] + payoff[half:])
        z_level = 0.5 * (z_level[:half] + z_level[half:])
    if getattr(model, "control_variate", False):
        payoff = _control(payoff, z_level)
    return payoff


def _terminal(model, maturity, n_paths, seed, state) -> np.ndarray:
    """Terminal ``Z`` on the same time grid ``price_with_error`` uses."""
    times = np.linspace(0.0, float(maturity), int(model.n_steps) + 1)
    paths = model.simulate(int(n_paths), times, seed=seed, state0=state)
    return np.asarray(paths.Z[:, -1], dtype=float)


def _price(model, forward, strikes, kind, maturity, df, state, n_paths, seed):
    """Cash prices, and a standard error when the model reports one."""
    if hasattr(model, "simulate") and hasattr(model, "price_with_error"):
        kwargs = {"state": state, "seed": seed}
        if n_paths is not None:
            kwargs["n_paths"] = n_paths
        result = model.price_with_error(forward, strikes, maturity, df, kind, **kwargs)
        return np.asarray(result.value, dtype=float), np.asarray(result.stderr, dtype=float)
    value = np.asarray(
        model.price(forward, strikes, maturity, df, kind, state=state),
        dtype=float,
    )
    return value, None


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
