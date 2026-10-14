"""Dupire local variance, and a log-Euler Monte Carlo on that variance.

The total-variance form is

    sigma_loc^2(k, T) = (dw/dT) / g(k),

where ``g`` is Durrleman's denominator

    g = 1 - (k/w) w_k + (1/4) (-1/4 - 1/w + k^2/w^2) w_k^2 + (1/2) w_kk.

A surface that implements ``variance_slopes`` supplies analytic derivatives
(raw SVI, the implied-vol spline, flat Black-Scholes). Anything else is
differenced from ``total_variance``, with the same steps as
:class:`~volsurface.surface.base.SurfaceMixin`: ``1e-4`` for a first
derivative and ``1e-3`` for a second. Values outside ``extrapolated`` are
NaN. There is no fill-in past the quotes.

Negative ``g`` and a local variance whose absolute value exceeds ``explode``
are counted and left in the array. They are not replaced by a floor or a
cap. ``g <= 0`` is the flag ``bad_denominator``. A huge but finite ratio is
``exploding``.

The Monte Carlo is log-Euler on ``Z``, the unit-mean forward. Local variance
is bilinear in ``(t, k)`` with ``k = log Z``. A query outside the grid is
clipped to the edge, which is flat extrapolation in ``t`` and in ``k``.
Draw order: one child of the seed, then one standard normal per step (the
forward shock only). A sample that is not strictly positive contributes no
diffusion on that step. That count is ``state["invalid"]``. A query that
was clipped is ``state["clipped"]``, whether or not the edge cell was
usable. The Euler bias does not shrink when more paths are added. The
flat-Black test states the bias it allows.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from volsurface.core.types import Paths, PriceResult, State
from volsurface.mc.grid import estimate_terminal
from volsurface.mc.rng import generator, spawn
from volsurface.plugins import register
from volsurface.surface.base import as_axes
from volsurface.surface.black import durrleman_g

# |local variance| above this is reported as exploding. 5 is a variance:
# an instantaneous vol above sqrt(5), about 224%.
EXPLODE = 5.0


@dataclass(frozen=True)
class DupireReport:
    """Local variance on one grid, plus the cells that are not a usable variance.

    ``variance`` is NaN outside the quoted domain. Inside the domain a
    negative denominator or an exploding ratio is still the number the
    formula produced (or NaN if the division overflowed). The masks say
    which cells those are. ``n_negative`` and ``n_exploding`` count them.
    """

    variance: np.ndarray
    denominator: np.ndarray
    bad_denominator: np.ndarray
    exploding: np.ndarray
    extrapolated: np.ndarray
    n_negative: int
    n_exploding: int


def dupire(surface, T, k, *, explode: float = EXPLODE, mask_outside: bool = True) -> DupireReport:
    """Dupire local variance of ``surface`` on the axes ``T`` and ``k``.

    See the module note for the formula, the analytic derivatives, and the
    masks. ``explode`` is the absolute variance above which a finite cell
    is counted as exploding.
    """
    T, k = as_axes(T, k)
    w, w_k, w_kk, w_T = _slopes(surface, T, k)
    denominator = durrleman_g(k[None, :], w, w_k, w_kk)
    # A flat slice with w = 0 is the T = 0 limit of Black. g is undefined
    # there because of the divisions by w, and the local variance is dw/dT.
    flat_zero = (w <= 0.0) & (np.abs(w_k) <= 1e-12) & (np.abs(w_kk) <= 1e-12)
    denominator = np.where(flat_zero, 1.0, denominator)
    with np.errstate(divide="ignore", invalid="ignore"):
        variance = w_T / denominator
    outside = np.asarray(surface.extrapolated(T, k), dtype=bool)
    if mask_outside:
        variance = np.where(outside, np.nan, variance)
        denominator = np.where(outside, np.nan, denominator)
    inside = ~outside
    bad = inside & np.isfinite(denominator) & (denominator <= 0.0)
    huge = inside & np.isfinite(variance) & (np.abs(variance) > float(explode))
    overflow = inside & ~bad & ~np.isfinite(variance) & np.isfinite(w_T)
    exploding = huge | overflow
    return DupireReport(
        variance=variance,
        denominator=denominator,
        bad_denominator=bad,
        exploding=exploding,
        extrapolated=outside,
        n_negative=int(bad.sum()),
        n_exploding=int(exploding.sum()),
    )


@register("localvol")
@dataclass(frozen=True)
class LocalVol:
    """Log-Euler Monte Carlo on a precomputed local-variance grid.

    ``variance`` has shape ``(len(t_grid), len(k_grid))``. Non-positive
    entries are skipped by the stepper. A query off the grid uses the
    edge cell.
    ``n_paths`` is the batch count. ``n_steps`` is how ``price`` divides
    ``[0, T]``. ``simulate`` uses the caller's ``times`` and ignores
    ``n_steps``.
    """

    t_grid: np.ndarray
    k_grid: np.ndarray
    variance: np.ndarray
    n_paths: int = 8_000
    n_steps: int = 32
    seed: int = 0

    def __post_init__(self) -> None:
        t_grid = np.asarray(self.t_grid, dtype=float)
        k_grid = np.asarray(self.k_grid, dtype=float)
        variance = np.asarray(self.variance, dtype=float)
        if t_grid.ndim != 1 or k_grid.ndim != 1:
            raise ValueError("t_grid and k_grid must be 1-d")
        if variance.shape != (t_grid.size, k_grid.size):
            raise ValueError(
                f"variance shape {variance.shape} does not match ({t_grid.size}, {k_grid.size})"
            )
        if np.any(np.diff(t_grid) <= 0.0) or np.any(np.diff(k_grid) <= 0.0):
            raise ValueError("grids must be strictly increasing")
        object.__setattr__(self, "t_grid", t_grid)
        object.__setattr__(self, "k_grid", k_grid)
        object.__setattr__(self, "variance", variance)

    def initial_state(self) -> State:
        """Empty. The whole variance grid is on the model, not in the state."""
        return {}

    def price(self, F, K, T, df=1.0, kind="call", *, state: State | None = None):
        """Cash price, the Monte Carlo mean. The standard error is not on this array."""
        return self.price_with_error(F, K, T, df, kind, state=state).value

    def price_with_error(
        self,
        F,
        K,
        T,
        df=1.0,
        kind="call",
        *,
        state: State | None = None,
        n_paths: int | None = None,
        seed: int | np.random.SeedSequence | None = None,
    ) -> PriceResult:
        """Mean and standard error at one maturity. Strikes share the paths."""
        started = time.perf_counter()
        n_paths = self.n_paths if n_paths is None else int(n_paths)
        seed = self.seed if seed is None else seed
        T = float(T)
        K_in = np.asarray(K, dtype=float)
        K_flat = np.atleast_1d(K_in)
        if T == 0.0:
            value = _intrinsic(float(F), K_flat, float(df), kind)
            return PriceResult(
                value=value.reshape(K_in.shape),
                stderr=np.zeros(K_in.shape),
                n_evals=0,
                wall_time=time.perf_counter() - started,
            )
        steps = max(int(self.n_steps), 1)
        times = np.linspace(0.0, T, steps + 1)
        paths = self.simulate(n_paths, times, seed=seed, state0=state)
        value, stderr = estimate_terminal(
            paths.Z[:, -1],
            float(F),
            K_flat,
            float(df),
            kind,
            antithetic=False,
            control_variate=False,
        )
        return PriceResult(
            value=value.reshape(K_in.shape),
            stderr=stderr.reshape(K_in.shape),
            n_evals=int(paths.Z.shape[0]),
            wall_time=time.perf_counter() - started,
        )

    def simulate(
        self,
        n_paths: int,
        times: np.ndarray,
        *,
        seed: int | np.random.SeedSequence,
        state0: State | None = None,
    ) -> Paths:
        """Unit-mean ``Z``. One normal per step. See the module note."""
        if state0:
            raise ValueError(f"LocalVol has an empty state, got keys {sorted(state0)}")
        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or times.size < 1 or times[0] != 0.0:
            raise ValueError("times must be 1-d and start at 0")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("times must be strictly increasing")
        n_paths = int(n_paths)
        rng = generator(spawn(seed, 1)[0])
        Z, invalid, clipped = _log_euler(
            self.t_grid,
            self.k_grid,
            self.variance,
            times,
            n_paths,
            rng,
        )
        return Paths(times=times, Z=Z, state={"invalid": invalid, "clipped": clipped})


def _slopes(surface, T, k):
    """Analytic slopes, or finite differences when the surface returns None.

    ``None`` means the surface has no closed form. A ``TypeError`` from a
    bad argument is not that signal and is not caught here.
    """
    method = getattr(surface, "variance_slopes", None)
    if method is not None:
        slopes = method(T, k)
        if slopes is not None:
            return slopes
    return _finite_difference_slopes(surface, T, k)


def _finite_difference_slopes(surface, T, k):
    """Central differences of ``total_variance``. Steps match the mixin."""
    T, k = as_axes(T, k)
    h = 1e-4
    h2 = 1e-3
    w = surface.total_variance(T, k)
    w_k = (surface.total_variance(T, k + h) - surface.total_variance(T, k - h)) / (2.0 * h)
    w_kk = (surface.total_variance(T, k + h2) - 2.0 * w + surface.total_variance(T, k - h2)) / h2**2
    up = T + h
    down = np.maximum(T - h, h * 0.5)
    w_T = (surface.total_variance(up, k) - surface.total_variance(down, k)) / (up - down)[:, None]
    return w, w_k, w_kk, w_T


def _log_euler(t_grid, k_grid, variance, times, n_paths, rng):
    """Bilinear local variance, log-Euler. Non-positive cells add no shock.

    Coordinates outside the grid are clipped to the edge before the
    lookup. ``clipped`` counts those samples. ``invalid`` counts samples
    whose cell is not a strictly positive finite variance.
    """
    interpolant = RegularGridInterpolator(
        (t_grid, k_grid),
        variance,
        bounds_error=False,
        fill_value=np.nan,
    )
    log_z = np.zeros(n_paths, dtype=float)
    Z = np.empty((n_paths, times.size), dtype=float)
    Z[:, 0] = 1.0
    invalid = np.zeros(n_paths, dtype=int)
    clipped = np.zeros(n_paths, dtype=int)
    query = np.empty((n_paths, 2), dtype=float)
    t_lo = float(t_grid[0])
    t_hi = float(t_grid[-1])
    k_lo = float(k_grid[0])
    k_hi = float(k_grid[-1])
    for i in range(1, times.size):
        dt = float(times[i] - times[i - 1])
        t_left = float(times[i - 1])
        off_t = t_left < t_lo or t_left > t_hi
        off_k = (log_z < k_lo) | (log_z > k_hi)
        clipped += off_t | off_k
        query[:, 0] = min(max(t_left, t_lo), t_hi)
        query[:, 1] = np.clip(log_z, k_lo, k_hi)
        sampled = np.asarray(interpolant(query), dtype=float)
        usable = np.isfinite(sampled) & (sampled > 0.0)
        invalid += ~usable
        local = np.where(usable, sampled, 0.0)
        shock = rng.standard_normal(n_paths)
        log_z = log_z - 0.5 * local * dt + np.sqrt(local * dt) * shock
        Z[:, i] = np.exp(log_z)
    return Z, invalid, clipped


def _intrinsic(F: float, K: np.ndarray, df: float, kind) -> np.ndarray:
    call = np.maximum(F - K, 0.0)
    put = np.maximum(K - F, 0.0)
    if isinstance(kind, str):
        if kind == "call":
            return df * call
        if kind == "put":
            return df * put
        raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    kinds = np.asarray(kind)
    return df * np.where(kinds == "call", call, put)
