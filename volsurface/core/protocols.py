"""Pricing protocols.

Structural, and runtime-checkable: a class from another repository satisfies
``Model`` by having the methods, without importing this package. ``docs/DESIGN.md``
section 4 lists the signatures. ``capabilities`` is the helper a dashboard
should call instead of keeping its own flag list.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike

from volsurface.core.types import (
    FitResult,
    GreeksResult,
    MarketData,
    Paths,
    State,
)


@runtime_checkable
class Model(Protocol):
    """Forward price at one maturity.

    ``state=None`` means :meth:`initial_state`. The return is always an
    ndarray of cash prices, with no standard error attached.
    """

    def price(
        self,
        F: float,
        K: ArrayLike,
        T: float,
        df: float = 1.0,
        kind="call",
        *,
        state: State | None = None,
    ) -> np.ndarray: ...

    def initial_state(self) -> State: ...


@runtime_checkable
class SupportsSimulate(Protocol):
    """Simulate the unit-mean martingale ``Z`` on a time grid that includes 0.

    ``seed`` is an ``int`` or a ``numpy.random.SeedSequence``. One child of
    that seed covers the whole run.
    """

    def simulate(
        self,
        n_paths: int,
        times: np.ndarray,
        *,
        seed: int | np.random.SeedSequence,
        state0: State | None = None,
    ) -> Paths: ...


@runtime_checkable
class SupportsCharFunc(Protocol):
    """Characteristic function of ``X_T = log(F_T / F(T))``.

    ``F(T)`` is today's forward for maturity ``T``. ``F_T`` is the random
    terminal forward. ``char_func`` returns ``E[exp(i u X_T)]``. There is no
    extra drift term: the mean of ``X_T`` is already inside the function (for
    Black-Scholes it is ``-sigma^2 T / 2``). ``cumulants`` returns
    ``(c1, c2, c4)`` of ``X_T``, which a cosine expansion uses to place its grid.
    """

    def char_func(self, u: np.ndarray, T: float, *, state: State | None = None) -> np.ndarray: ...

    def cumulants(self, T: float, *, state: State | None = None) -> tuple[float, float, float]: ...


@runtime_checkable
class Parametric(Protocol):
    """Frozen parameters with a vector of unconstrained coordinates."""

    def to_unconstrained(self) -> np.ndarray: ...

    @classmethod
    def from_unconstrained(cls, x: np.ndarray) -> Parametric: ...

    @classmethod
    def param_spec(cls) -> list: ...


@runtime_checkable
class Surface(Protocol):
    """The only required method is the normalised call ``call(T, k)``.

    ``T`` has shape ``(m,)``, ``k`` has shape ``(n,)``, the result has shape
    ``(m, n)``. ``market`` and ``domain`` are attributes. Derived quantities
    live on the mixin, and a surface overrides them when it has an analytic form.
    ``isinstance`` checks the method only; non-method members are not part of
    the runtime check.
    """

    market: object
    domain: object

    def call(self, T, k) -> np.ndarray: ...


@runtime_checkable
class Fitter(Protocol):
    def fit(self, data: MarketData) -> FitResult: ...


@runtime_checkable
class GreeksEngine(Protocol):
    """One way of computing greeks.

    ``supports`` is how a dashboard greys out a cell. ``compute`` returns one
    row per contract and one column per risk factor.
    """

    def supports(self, model, contract, order: int) -> bool: ...

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
    ) -> GreeksResult: ...


def capabilities(model: object) -> set[str]:
    """Names of the protocols ``model`` satisfies.

    The strings are ``"price"``, ``"simulate"``, ``"char_func"``, and
    ``"parametric"``. Membership is structural, so a hand-kept flag cannot
    drift away from the methods.
    """
    found: set[str] = set()
    if isinstance(model, Model):
        found.add("price")
    if isinstance(model, SupportsSimulate):
        found.add("simulate")
    if isinstance(model, SupportsCharFunc):
        found.add("char_func")
    if isinstance(model, Parametric):
        found.add("parametric")
    return found
