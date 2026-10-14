"""Geometric Brownian motion, simulated, then priced as a Monte Carlo average.

The forward measure makes ``Z_t = S_t / F_t`` a martingale with unit mean.
Constant volatility ``sigma`` gives

    dZ = sigma Z dW.

The exact transition across a gap ``dt`` is lognormal,

    Z <- Z * exp(-sigma^2 dt / 2 + sigma sqrt(dt) g),    g ~ N(0, 1).

Euler-Maruyama drops the Ito term and the state dependence inside the step,

    Z <- Z + sigma Z sqrt(dt) g.

That leaves ``E[Z] = 1`` (each step is a martingale) but the distribution is
not lognormal, so a call price is biased. More paths do not remove the bias.
More steps do.

Draw order, for a call to ``simulate``. One child of the seed is spawned, then
for each time after 0 a vector of ``n_paths`` standard normals is drawn, path
index fastest. Nothing else is drawn. With ``antithetic`` on, the paired path
uses the negated normal and is stored in the second half of ``Z``. The batch
count stays ``n_paths``; the stored row count doubles.

The control variate, applied when the price is formed and not inside
``simulate``, is the terminal ``Z``. Its mean is 1. The coefficient is the
ordinary least-squares slope of the (antithetic-averaged) payoff on ``Z``.
The reported standard error is the sample standard deviation of those
controlled batch payoffs, divided by ``sqrt(n_paths)``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from volsurface.core.types import Paths, PriceResult, State
from volsurface.mc.grid import estimate_terminal
from volsurface.mc.rng import generator, spawn
from volsurface.plugins import register


@register("gbm")
@dataclass(frozen=True)
class GBM:
    """Black dynamics, priced by simulation. See the module note.

    ``n_paths`` is the number of batches, not the stored row count. ``n_steps``
    is the number of steps ``price`` uses between 0 and ``T``. ``price_grid``
    steps only on the quoted maturities and does not subdivide by ``n_steps``.
    ``seed`` is the parent seed; each run spawns one child from it.
    """

    sigma: float
    n_paths: int = 20_000
    n_steps: int = 1
    seed: int = 0
    antithetic: bool = False
    control_variate: bool = False
    scheme: str = "exact"

    def __post_init__(self) -> None:
        if self.sigma <= 0.0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")
        if self.n_paths < 2:
            raise ValueError(f"n_paths must be at least 2, got {self.n_paths}")
        if self.n_steps < 1:
            raise ValueError(f"n_steps must be at least 1, got {self.n_steps}")
        if self.scheme not in ("exact", "euler"):
            raise ValueError(f"scheme must be 'exact' or 'euler', got {self.scheme!r}")

    def initial_state(self) -> State:
        """Empty. Geometric Brownian motion has no latent variable."""
        return {}

    def price(self, F, K, T, df=1.0, kind="call", *, state: State | None = None):
        """Cash price, the Monte Carlo mean. The standard error is not on this array."""
        return self.price_with_error(F, K, T, df, kind, state=state).value

    def price_with_error(
        self, F, K, T, df=1.0, kind="call", *, state: State | None = None,
        n_paths: int | None = None, seed: int | np.random.SeedSequence | None = None,
    ) -> PriceResult:
        """Mean and standard error at one maturity, every strike from one set of paths."""
        _reject_state(state)
        started = time.perf_counter()
        n_paths = self.n_paths if n_paths is None else int(n_paths)
        seed = self.seed if seed is None else seed
        T = float(T)
        if T < 0.0:
            raise ValueError(f"T must be non-negative, got {T}")
        K_in = np.asarray(K, dtype=float)
        K_flat = np.atleast_1d(K_in)
        if T == 0.0:
            value = _intrinsic(float(F), K_flat, float(df), kind).reshape(K_in.shape)
            stderr = np.zeros(K_in.shape, dtype=float)
            return PriceResult(
                value=value, stderr=stderr, n_evals=0,
                wall_time=time.perf_counter() - started,
            )
        times = np.linspace(0.0, T, self.n_steps + 1)
        paths = self.simulate(n_paths, times, seed=seed, state0=state)
        value, stderr = estimate_terminal(
            paths.Z[:, -1], float(F), K_flat, float(df), kind,
            antithetic=self.antithetic, control_variate=self.control_variate,
        )
        return PriceResult(
            value=value.reshape(K_in.shape), stderr=stderr.reshape(K_in.shape),
            n_evals=int(paths.Z.shape[0]), wall_time=time.perf_counter() - started,
        )

    def simulate(
        self, n_paths: int, times: np.ndarray, *,
        seed: int | np.random.SeedSequence, state0: State | None = None,
    ) -> Paths:
        """Unit-mean ``Z`` on ``times``. ``times[0]`` is 0. See the module note for draws."""
        _reject_state(state0)
        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or times.size < 1 or times[0] != 0.0:
            raise ValueError("times must be 1-d and start at 0")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("times must be strictly increasing")
        n_paths = int(n_paths)
        if n_paths < 1:
            raise ValueError(f"n_paths must be positive, got {n_paths}")
        # One child for the whole run, shared by every strike priced off these paths.
        rng = generator(spawn(seed, 1)[0])
        rows = n_paths * (2 if self.antithetic else 1)
        Z = np.empty((rows, times.size), dtype=float)
        Z[:, 0] = 1.0
        if self.scheme == "exact":
            _exact(Z, times, self.sigma, n_paths, self.antithetic, rng)
        else:
            _euler(Z, times, self.sigma, n_paths, self.antithetic, rng)
        return Paths(times=times, Z=Z, state={})


def _reject_state(state: State | None) -> None:
    if state:
        raise ValueError(f"GBM has an empty state, got keys {sorted(state)}")


def _shock(rng, n_paths: int, antithetic: bool) -> np.ndarray:
    """One vector of normals. The mirror, if any, is the second half and is not a new draw."""
    g = rng.standard_normal(n_paths)
    if antithetic:
        return np.concatenate([g, -g])
    return g


def _exact(Z, times, sigma, n_paths, antithetic, rng) -> None:
    """Lognormal transition. Exact for geometric Brownian motion, at any step size."""
    log_z = np.zeros(Z.shape[0], dtype=float)
    for i in range(1, times.size):
        dt = float(times[i] - times[i - 1])
        shock = _shock(rng, n_paths, antithetic)
        log_z = log_z - 0.5 * sigma * sigma * dt + sigma * np.sqrt(dt) * shock
        Z[:, i] = np.exp(log_z)


def _euler(Z, times, sigma, n_paths, antithetic, rng) -> None:
    """Euler-Maruyama. A martingale, not a lognormal. ``Z`` may become negative."""
    z = np.ones(Z.shape[0], dtype=float)
    for i in range(1, times.size):
        dt = float(times[i] - times[i - 1])
        shock = _shock(rng, n_paths, antithetic)
        z = z + sigma * z * np.sqrt(dt) * shock
        Z[:, i] = z


def _intrinsic(F: float, K: np.ndarray, df: float, kind) -> np.ndarray:
    """Expiry payoff. Used when ``T`` is 0 and there is nothing to simulate."""
    call = np.maximum(F - K, 0.0)
    put = np.maximum(K - F, 0.0)
    if isinstance(kind, str):
        if kind == "call":
            return df * call
        if kind == "put":
            return df * put
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    kinds = np.asarray(kind)
    if not np.all((kinds == "call") | (kinds == "put")):
        raise ValueError(f"kind must be 'call' or 'put', got {kinds}")
    return df * np.where(kinds == "call", call, put)
