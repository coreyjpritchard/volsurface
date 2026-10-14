"""Regime-switching Bergomi at H = 0.5. DESIGN section 22.

The chain is a continuous-time Markov chain, independent of the Brownian
drivers. It switches ``nu`` only. ``xi_level`` and ``both`` raise. The
variance is

    v_t = xi0(t) exp(nu_{r_t} sqrt(2H) Y_t - nu_{r_t}^2 t^{2H} / 2).

``Y`` has variance ``t^{2H} / (2H)``, so ``E[v_t | chain] = xi0(t)``. At
``H = 0.5``, ``Y`` is Brownian and this module draws it. Any other ``H``
asks ``create("rbergomi")`` for ``driver_y``. A missing plugin raises.
This file does not simulate the hybrid scheme, and a stand-in driver is
not labelled rBergomi.

Whether ``Z`` stays a martingale when ``nu`` switches is a hypothesis.
The log-Euler step freezes ``v`` at the left of the interval, so each
step is a martingale increment given the past. The test is the z-rule.

Draw order, one child of the run seed for each stream
(:func:`volsurface.markets.seeds.stream_seeds`). The streams are separate
generators, so draws in one do not move another.

- regime: one uniform per path for the initial state; one normal per path
  for the time-0 observation; then, at each later time, one uniform per
  path for the transition and one normal per path for the observation.
- vol: every increment of ``Y``, in time order, when ``H = 0.5``. When
  ``H`` is not 0.5 the plugin consumes this child and this module draws
  nothing from it.
- spot: one normal per path per step, the shock orthogonal to ``Y``.
  ``dW = rho dY + sqrt(1 - rho^2) dW_perp``.
- surrogate: spawned and not drawn.

The spot update is the left-point exponential,
``Z <- Z exp(sqrt(v) dW - v dt / 2)``.

The forward variance stored in ``state["xi"]`` is, at ``H = 0.5``,

    xi_t(u) = xi0(u) sum_r P(r_u = r | r_t) exp(nu_r Y_t - nu_r^2 t / 2).

That is ``E[v_u | Y_t, r_t]``. The filter observation is
``y_t | r_t ~ Normal(obs_mean[r_t], obs_sigma^2)``, drawn with the regime
stream. ``regime_prob`` is the Hamilton filter of ``y``. It does not see
the future of ``y``.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.linalg import expm

from volsurface import bs
from volsurface.core.types import MarketPaths, RegimeSpec, SurfaceBatch
from volsurface.markets.filter import hamilton_filter
from volsurface.markets.pricers import integrate_forward_variance
from volsurface.markets.seeds import DRAW_ORDER, stream_seeds
from volsurface.mc.rng import as_seed_sequence, generator
from volsurface.plugins import create


class RegimeSwitchingBergomi:
    """Two or more regimes, switching vol-of-vol. See the module note."""

    def __init__(
        self,
        spec: RegimeSpec,
        *,
        xi0: float = 0.04,
        rho: float = -0.4,
        H: float = 0.5,
        obs_means: np.ndarray | None = None,
        obs_sigma: float = 0.25,
        forward: float = 100.0,
        rate: float = 0.0,
        smile_k: np.ndarray | None = None,
    ):
        self.spec = spec
        self.xi0 = float(xi0)
        self.rho = float(rho)
        self.H = float(H)
        self.obs_sigma = float(obs_sigma)
        self.forward = float(forward)
        self.rate = float(rate)
        self.smile_k = (
            np.linspace(-0.2, 0.2, 9) if smile_k is None else np.asarray(smile_k, dtype=float)
        )
        nus = [float(item["nu"]) for item in spec.params]
        self.obs_means = np.asarray(nus if obs_means is None else obs_means, dtype=float)

    def simulate_market(
        self,
        n_paths: int,
        times: np.ndarray,
        *,
        seed: int | np.random.SeedSequence,
        state0=None,
    ) -> MarketPaths:
        """Paths, the chain, the filter, and the forward-variance curve."""
        spec = self._checked_spec()
        grid = _times(times)
        n_paths = int(n_paths)
        if n_paths < 1:
            raise ValueError(f"n_paths must be positive, got {n_paths}")
        if state0:
            raise ValueError(f"state0 is not used, got keys {sorted(state0)}")
        streams = stream_seeds(seed)
        regime_rng = generator(streams["regime"])
        spot_rng = generator(streams["spot"])
        nus = np.array([float(item["nu"]) for item in spec.params], dtype=float)
        generator_q = np.asarray(spec.generator, dtype=float)
        p0 = np.asarray(spec.p0, dtype=float)
        n_regimes = nus.size
        regime = np.empty((n_paths, grid.size), dtype=np.int8)
        regime[:, 0] = _categorical(regime_rng, np.broadcast_to(p0, (n_paths, n_regimes)).copy())
        Y = self._driver(n_paths, grid, streams["vol"])
        Z = np.ones((n_paths, grid.size))
        v = np.empty((n_paths, grid.size))
        observed = np.empty((n_paths, grid.size))
        v[:, 0] = self._variance(grid[0], Y[:, 0], regime[:, 0], nus)
        observed[:, 0] = self._observation(regime[:, 0], regime_rng)
        rho = self.rho
        orthogonal = math.sqrt(max(0.0, 1.0 - rho * rho))
        steps: list[np.ndarray] = []
        for i in range(1, grid.size):
            dt = float(grid[i] - grid[i - 1])
            step = _stochastic(expm(generator_q * dt))
            steps.append(step)
            regime[:, i] = _categorical(regime_rng, step[regime[:, i - 1]])
            dY = Y[:, i] - Y[:, i - 1]
            dW = rho * dY + orthogonal * math.sqrt(dt) * spot_rng.standard_normal(n_paths)
            left = np.maximum(v[:, i - 1], 0.0)
            Z[:, i] = Z[:, i - 1] * np.exp(np.sqrt(left) * dW - 0.5 * left * dt)
            v[:, i] = self._variance(grid[i], Y[:, i], regime[:, i], nus)
            observed[:, i] = self._observation(regime[:, i], regime_rng)
        filtered = hamilton_filter(
            observed,
            steps if steps else np.eye(n_regimes),
            self.obs_means,
            np.full(n_regimes, self.obs_sigma**2),
            p0,
        )
        xi = _forward_curve(self.xi0, grid, Y, regime, nus, generator_q, self.H)
        parent = as_seed_sequence(seed)
        return MarketPaths(
            times=grid,
            Z=Z,
            state={"v": v, "Y": Y, "y": observed, "xi": xi, "xi_tenor": grid.copy()},
            F0=_flat(self.forward),
            df=_discount(self.rate),
            regime=regime,
            regime_prob=filtered,
            meta={
                "model": "RegimeSwitchingBergomi",
                "H": self.H,
                "rho": self.rho,
                "xi0": self.xi0,
                "seed_entropy": parent.entropy,
                "spawn_key": tuple(parent.spawn_key),
                "draw_order": DRAW_ORDER,
                "surrogate_drawn": False,
                "roughvol": self.H != 0.5 and _is_roughvol(self._plugin),
                "label": "Brownian" if self.H == 0.5 else type(self._plugin).__name__,
            },
        )

    def surface_at(self, paths: MarketPaths, i_t: int) -> SurfaceBatch:
        """Black calls on the integrated forward variance. ``k`` is pathwise log-moneyness."""
        tenors = np.asarray(paths.state["xi_tenor"], dtype=float)
        now = float(paths.times[i_t])
        future = tenors[tenors > now + 1e-15]
        if future.size == 0:
            future = tenors[-1:]
        if future.size > 7:
            future = future[np.linspace(0, future.size - 1, 7).astype(int)]
        xi = np.asarray(paths.state["xi"][:, i_t, :], dtype=float)
        k = self.smile_k
        strikes = np.exp(k)
        call = np.empty((paths.Z.shape[0], future.size, k.size))
        for j, maturity in enumerate(future):
            total = integrate_forward_variance(xi, tenors, now, float(maturity))
            vol = np.sqrt(np.maximum(total, 0.0))
            for a, strike in enumerate(strikes):
                call[:, j, a] = bs.price(1.0, float(strike), 1.0, vol, 1.0)
        return SurfaceBatch(T=np.asarray(future, dtype=float), k=k.copy(), call=call)

    def _checked_spec(self) -> RegimeSpec:
        spec = self.spec
        if spec.switch != "nu":
            raise ValueError(f"this phase switches nu only, got {spec.switch!r}")
        if not math.isfinite(self.rho) or abs(self.rho) > 1.0:
            raise ValueError(f"rho must lie in [-1, 1], got {self.rho}")
        if self.obs_sigma <= 0.0:
            raise ValueError(f"obs_sigma must be positive, got {self.obs_sigma}")
        generator_q = np.asarray(spec.generator, dtype=float)
        p0 = np.asarray(spec.p0, dtype=float)
        n_regimes = len(spec.params)
        if generator_q.shape != (n_regimes, n_regimes):
            raise ValueError(
                f"generator shape {generator_q.shape} does not match {n_regimes} regimes"
            )
        if not np.allclose(generator_q.sum(axis=1), 0.0, atol=1e-10):
            raise ValueError("each generator row must sum to 0")
        if np.any(generator_q[~np.eye(n_regimes, dtype=bool)] < -1e-12):
            raise ValueError("an off-diagonal generator entry is negative")
        if p0.shape != (n_regimes,) or np.any(p0 < 0.0) or not np.isclose(p0.sum(), 1.0):
            raise ValueError("p0 must be a probability vector, one entry per regime")
        if self.obs_means.shape != (n_regimes,):
            raise ValueError("obs_means must have one entry per regime")
        for item in spec.params:
            if "nu" not in item or float(item["nu"]) <= 0.0:
                raise ValueError("each regime needs a positive nu")
        if self.xi0 <= 0.0:
            raise ValueError(f"xi0 must be positive, got {self.xi0}")
        return spec

    def _driver(self, n_paths: int, times: np.ndarray, seed) -> np.ndarray:
        if self.H == 0.5:
            return _brownian(n_paths, times, seed)
        try:
            plugin = create("rbergomi")
        except KeyError as exc:
            raise RuntimeError(
                "H other than 0.5 uses the roughvol plugin's driver Y, and that plugin "
                "is not installed. This module does not simulate the hybrid scheme."
            ) from exc
        if not hasattr(plugin, "driver_y"):
            raise RuntimeError(
                "the rbergomi plugin has no driver_y. This module does not simulate Y itself."
            )
        self._plugin = plugin
        return np.asarray(plugin.driver_y(n_paths, times, seed=seed, H=self.H), dtype=float)

    def _variance(self, t: float, Y: np.ndarray, regime: np.ndarray, nus: np.ndarray) -> np.ndarray:
        nu = nus[regime]
        scale = math.sqrt(2.0 * self.H)
        clock = float(t) ** (2.0 * self.H)
        return self.xi0 * np.exp(nu * scale * Y - 0.5 * nu * nu * clock)

    def _observation(self, regime: np.ndarray, rng) -> np.ndarray:
        shock = rng.standard_normal(regime.shape[0])
        return self.obs_means[regime] + self.obs_sigma * shock


def _is_roughvol(plugin) -> bool:
    name = type(plugin).__name__
    return name == "RoughBergomi" and type(plugin).__module__.startswith("roughvol")


def _times(times: np.ndarray) -> np.ndarray:
    grid = np.asarray(times, dtype=float)
    if grid.ndim != 1 or grid.size < 1 or grid[0] != 0.0:
        raise ValueError("times must be 1-d and start at 0")
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("times must be strictly increasing")
    return grid


def _categorical(rng, probs: np.ndarray) -> np.ndarray:
    """One draw per row. ``probs`` has shape ``(m, R)`` and rows sum to 1."""
    cdf = np.cumsum(probs, axis=1)
    cdf[:, -1] = 1.0
    u = rng.random(probs.shape[0])
    return np.argmax(u[:, None] < cdf, axis=1).astype(np.int8)


def _brownian(n_paths: int, times: np.ndarray, seed) -> np.ndarray:
    """Standard Brownian motion. ``Var(Y_t) = t``, which is ``t^{2H} / (2H)`` at ``H = 0.5``."""
    rng = generator(seed)
    Y = np.zeros((n_paths, times.size))
    for i in range(1, times.size):
        dt = float(times[i] - times[i - 1])
        Y[:, i] = Y[:, i - 1] + math.sqrt(dt) * rng.standard_normal(n_paths)
    return Y


def _forward_curve(xi0, tenors, Y, regime, nus, generator_q, H) -> np.ndarray:
    """``E[v_u | Y_t, r_t]`` on the time grid.

    The mixture over future regimes is the ``H = 0.5`` case.
    """
    m, n = Y.shape
    curve = np.zeros((m, n, n))
    scale = math.sqrt(2.0 * H)
    for i in range(n):
        clock = float(tenors[i]) ** (2.0 * H)
        nu = nus[regime[:, i]]
        level = np.exp(nu * scale * Y[:, i] - 0.5 * nu * nu * clock)
        if H == 0.5:
            basis = np.exp(nus[None, :] * Y[:, i, None] - 0.5 * (nus**2)[None, :] * clock)
        else:
            basis = None
        for j in range(i, n):
            if H == 0.5 and j > i:
                law = _stochastic(expm(generator_q * float(tenors[j] - tenors[i])))
                probs = law[regime[:, i]]
                curve[:, i, j] = xi0 * np.sum(probs * basis, axis=1)
            else:
                curve[:, i, j] = xi0 * level
    return curve


def _stochastic(matrix: np.ndarray) -> np.ndarray:
    """Clip a rounding error and restore row sums of 1."""
    clipped = np.maximum(np.asarray(matrix, dtype=float), 0.0)
    return clipped / clipped.sum(axis=1, keepdims=True)


def _flat(level: float):
    def curve(t):
        return np.full(np.shape(np.asarray(t, dtype=float)), level, dtype=float)

    return curve


def _discount(rate: float):
    def curve(t):
        return np.exp(-rate * np.asarray(t, dtype=float))

    return curve
