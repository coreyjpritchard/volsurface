"""Types shared by pricing, surfaces, and the arbitrage checker.

``docs/DESIGN.md`` section 3 is the definition. The notes below are the
calculus a reader needs in order to use the objects.

The forward measure prices a call as ``df * E[(F_T - K)+]``. ``F_T`` is the
random terminal forward. ``F`` in the calling code is today's forward curve
``F(T)``. Dividing by ``df * F(T)`` gives the normalised call ``C``, a number
between the intrinsic value ``(1 - K/F(T))+`` and 1. Log-moneyness
``k = log(K / F(T))`` is the strike coordinate of every surface.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd

# Batched over paths when a path is in hand. Black-Scholes uses {}.
State = Mapping[str, np.ndarray]


@dataclass(frozen=True)
class PriceResult:
    """A price and, for Monte Carlo, the standard error of that estimate.

    ``stderr`` is None for a closed form. ``n_evals`` counts primitive price
    evaluations: 1 for a formula, and the number of paths (times the number of
    bumped repricings, for a finite-difference greek) for a simulation.
    ``wall_time`` is seconds of computing, not of importing.
    """

    value: np.ndarray
    stderr: np.ndarray | None
    n_evals: int
    wall_time: float


@dataclass(frozen=True)
class Paths:
    """Simulated unit-mean martingale.

    ``times`` has shape ``(n + 1,)`` and starts at 0. ``Z`` has shape
    ``(m, n + 1)`` and holds ``S_t / F_t``. Each entry of ``state`` is an
    array broadcastable to the paths, typically ``(m, n + 1)``.
    """

    times: np.ndarray
    Z: np.ndarray
    state: dict[str, np.ndarray]


@dataclass(frozen=True)
class ParamSpec:
    """One slider and one fitter coordinate.

    ``transform`` maps an unconstrained real to the parameter:

    - ``"log"``: ``value = exp(x)``, for a positive parameter.
    - ``"logit"``: ``value = lower + (upper - lower) / (1 + exp(-x))``.
    - ``"identity"``: ``value = x``, clipped to ``[lower, upper]``.
    """

    name: str
    lower: float
    upper: float
    default: float
    transform: str


@dataclass(frozen=True)
class Quote:
    """One option, in the units the chain loader uses (cash prices, year fraction)."""

    T: float
    K: float
    kind: str
    bid: float
    ask: float
    exercise: str
    settlement: str
    volume: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("call", "put"):
            raise ValueError(f"kind must be 'call' or 'put', got {self.kind!r}")
        if self.exercise not in ("european", "american"):
            raise ValueError(f"exercise must be 'european' or 'american', got {self.exercise!r}")
        if self.settlement not in ("cash", "physical"):
            raise ValueError(f"settlement must be 'cash' or 'physical', got {self.settlement!r}")


@dataclass(frozen=True)
class Curves:
    """Forward and discount factor as functions of expiry.

    Both callables accept a scalar or an array and return the same shape.
    ``F(T)`` is the forward to ``T``. ``df(T)`` is the discount factor to ``T``.
    """

    F: Callable[[Any], Any]
    df: Callable[[Any], Any]


@dataclass(frozen=True)
class MarketData:
    """A chain at one instant, plus the curves implied by it.

    ``quotes`` uses the :class:`Quote` columns and may carry ``price``, ``iv``,
    ``k``, ``F``, ``df``, ``expiry``, and ``price_source`` from the loader.
    ``F_err(T)`` is an uncertainty on the forward, in price units, or None.
    ``day_count`` is ``"ACT/365"`` for this package's chains.
    """

    asof: datetime
    quotes: pd.DataFrame
    F: Callable[[Any], Any]
    df: Callable[[Any], Any]
    F_err: Callable[[Any], Any] | None
    day_count: str

    def curves(self) -> Curves:
        return Curves(self.F, self.df)


@dataclass(frozen=True)
class Domain:
    """Quoted rectangle in ``(T, k)``. Outside it, a surface marks extrapolation."""

    t_min: float
    t_max: float
    k_min: float
    k_max: float

    def contains(self, T, k) -> np.ndarray:
        """Boolean array, true where ``(T, k)`` lies inside the rectangle."""
        T = np.asarray(T, dtype=float)
        k = np.asarray(k, dtype=float)
        t_ok = (T >= self.t_min) & (T <= self.t_max)
        k_ok = (k >= self.k_min) & (k <= self.k_max)
        return t_ok & k_ok


@dataclass(frozen=True)
class FitResult:
    """Output of a fitter.

    ``in_spread`` is the fraction of two-sided quotes whose model call sits
    inside the bid and ask, or NaN when the snapshot has no two-sided quote.

    ``slice_report`` is one row per fitted expiry when the fitter fills it,
    and None otherwise. The raw SVI fitter's columns are ``T``, ``min_g``,
    ``lee_ok``, ``calendar_ok``, and ``in_spread``.

    ``detail`` holds fitter-specific measured quantities (the SANOS objective,
    ``dx``, ``eta``, ``omega``, and the projection correction). It is empty
    for the Phase 1 fitters.
    """

    surface: Any
    residuals: np.ndarray
    in_spread: float
    wall_time: float
    solver_status: str
    slice_report: pd.DataFrame | None = None
    detail: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ArbTol:
    """Tolerances in normalised-call units.

    ``price`` is the slack allowed on bounds, slope, convexity, and calendar.
    ``spreads`` is a half-spread used only to fill ``magnitude_in_spreads``.
    """

    price: float = 1e-9
    spreads: float | None = None


@dataclass(frozen=True)
class ArbReport:
    """Pointwise arbitrage report. See ``docs/DESIGN.md`` section 6.

    ``ok`` is True when every checked node passed, False when a violation was
    recorded, and None when nothing was checked (``n_checked == 0``). An empty
    quote file is the None case. There is no pass to report.

    ``n_checked`` counts the nodes that entered the test. For a surface that
    is the size of the grid. For quotes it is the number of quote rows that
    entered the linear program.

    ``note`` is part of the result. For a surface check it records that the
    nodes do not constrain the surface between them.
    """

    violations: pd.DataFrame
    masks: dict[str, np.ndarray]
    min_density: float
    ok: bool | None
    note: str
    n_checked: int


@dataclass(frozen=True)
class Vanilla:
    """European call or put on the forward. ``kind`` is ``"call"`` or ``"put"``."""

    K: float
    kind: str = "call"

    def __post_init__(self) -> None:
        if self.kind not in ("call", "put"):
            raise ValueError(f"kind must be 'call' or 'put', got {self.kind!r}")


@dataclass(frozen=True)
class Digital:
    """Cash-or-nothing digital: pays 1 when the forward finishes in the money.

    A call pays ``1_{F_T > K}``. A put pays ``1_{F_T < K}``. The discounted
    Black-Scholes call price is ``df * N(d2)``.
    """

    K: float
    kind: str = "call"

    def __post_init__(self) -> None:
        if self.kind not in ("call", "put"):
            raise ValueError(f"kind must be 'call' or 'put', got {self.kind!r}")


@dataclass(frozen=True)
class Contract:
    """A payoff at one expiry. ``T`` is the year fraction."""

    payoff: Vanilla | Digital
    T: float


@dataclass(frozen=True)
class Spot:
    """d/dS under a one-for-one move of the forward (``r = q``).

    With ``df`` held fixed this is the same derivative as :class:`Forward`.
    A later adapter that bumps spot while the forward moves by ``exp((r-q)T)``
    will distinguish the two. Phase 1 does not.
    """


@dataclass(frozen=True)
class Forward:
    """d/dF, the discount factor held fixed."""


@dataclass(frozen=True)
class Param:
    """A named model parameter. ``Param("sigma")`` is Black-Scholes vega."""

    name: str


@dataclass(frozen=True)
class GreeksResult:
    """Greeks for several contracts against several risk factors.

    ``value`` has shape ``(n_contracts, n_factors)``. ``stderr`` has that
    shape, or is None when every entry is a closed form. ``method`` names the
    engine (``"analytic"``, ``"finite_difference"``, ``"pathwise"``,
    ``"likelihood_ratio"``, ``"autodiff"``).
    """

    value: np.ndarray
    stderr: np.ndarray | None
    n_evals: int
    wall_time: float
    method: str


@dataclass(frozen=True)
class MarketPaths(Paths):
    """Paths plus the curves and the regime labels a hedger reads.

    ``F0(T)`` is today's forward curve and ``df(T)`` is today's discount
    curve. Along a path, the spot at grid time ``t`` is ``F0(t) * Z``, and
    the forward from ``t`` to a later expiry ``T`` is ``Z * F0(T)``.

    ``regime`` has shape ``(m, n + 1)`` and dtype int8. It is the true label.
    It is an oracle. It is not a default feature. ``regime_prob`` has shape
    ``(m, n + 1, R)`` and holds the causal Hamilton-filter probabilities.
    ``meta`` records the model name, the parameters, the seed entropy, the
    spawn key, and the draw order.
    """

    F0: Callable
    df: Callable
    regime: np.ndarray | None = None
    regime_prob: np.ndarray | None = None
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RegimeSpec:
    """A continuous-time Markov chain over model parameters.

    ``generator`` has shape ``(R, R)``. Off-diagonal entries are rates and
    each row sums to 0. ``params`` has one dict per regime. ``p0`` is the
    law at ``t = 0``, shape ``(R,)``. ``switch`` names which coordinates
    differ across ``params``. This phase implements ``"nu"`` only.
    ``"xi_level"`` and ``"both"`` are named so a later arm can use them.
    """

    generator: np.ndarray
    params: list[dict]
    p0: np.ndarray
    switch: Literal["nu", "xi_level", "both"]


@dataclass(frozen=True)
class SurfaceBatch:
    """Normalised calls on one date, for every path in a batch.

    ``T`` has shape ``(n_exp,)``, ``k`` has shape ``(n_k,)``, and ``call``
    has shape ``(m, n_exp, n_k)``. ``stderr`` matches ``call`` when the
    price is a Monte Carlo estimate. ``total_variance`` and
    ``total_variance_se`` have shape ``(m, n_exp)`` and are the log
    contract ``-2 E[log Z]`` and its standard error.
    """

    T: np.ndarray
    k: np.ndarray
    call: np.ndarray
    stderr: np.ndarray | None = None
    total_variance: np.ndarray | None = None
    total_variance_se: np.ndarray | None = None
