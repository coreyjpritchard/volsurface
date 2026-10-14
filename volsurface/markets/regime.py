"""Regime-switching Bergomi. DESIGN section 22.

The chain is a continuous-time Markov chain, independent of the Brownian
drivers. It switches ``nu`` only. ``xi_level`` and ``both`` raise.

A switch acts on increments only. The new ``nu`` multiplies new Volterra
increments. It does not rescale the integral already accumulated, so ``v``
has no jump at a switch. With ``nu_s`` the vol of vol in force at time ``s``,

    I_t(u) = int_0^t nu_s (u - s)^{H - 1/2} dW_s,
    v_t = xi0(t) exp(sqrt(2H) I_t(t) - H int_0^t nu_s^2 (t - s)^{2H - 1} ds).

On a grid, the increment over ``(t_{i-1}, t_i]`` uses the ``nu`` of the
regime at ``t_{i-1}``. The regime recorded at ``t_i`` applies to later
increments. At ``H = 1/2`` the kernel is 1 and the step is

    v_i = v_{i-1} exp(nu_{i-1} dW_i - nu_{i-1}^2 dt_i / 2),

with ``v_0 = xi0``. Constant ``nu`` reduces to the compensator
``nu^2 t^{2H} / 2``.

``H = 1/2`` is drawn here, as Brownian increments. Any other ``H`` calls
``volterra_components`` on the rBergomi plugin. That method returns the
hybrid-scheme components ``(dW, I_1, ..., I_kappa)`` on each step. This
module multiplies the whole vector of a step by the ``nu`` in force on
that step, then places ``I_k`` at lag ``k`` and the tail with roughvol's
power-kernel weights. ``dW`` alone is the covariance regression of the
kernel onto the Wiener increment. At ``H = 0.1`` that regression keeps
about ``2H / (H + 1/2)^2 = 5/9`` of ``Var(I_1)``, and ``E[v]`` falls short
of ``xi0``. A missing plugin, or a plugin without ``volterra_components``,
raises. This file does not simulate the hybrid scheme, and a stand-in is
not labelled rBergomi.

The interface:

    volterra_components(n_paths, times, *, seed, H) -> components

``times`` has shape ``(n_steps + 1,)``, starts at 0, and is equally spaced.
``components`` has shape ``(n_paths, n_steps, kappa + 1)``. Last-axis index
0 is ``dW``, with variance equal to the step length. Index ``k >= 1`` is
``I_k``, the exact integral of the power kernel at lag ``k``. ``seed`` is
already the volatility child. The draw is the one ``hybrid_scheme_rng``
makes from ``generator(seed)``: do not spawn another child. ``kappa`` is
the plugin's, 1 or 2. The kernel weights are not returned. The tail uses
``roughvol.volterra.power_kernel_weights`` for this ``H``, the step count,
the horizon and ``kappa``. Its first ``kappa`` entries are 0. The
contribution of step ``j`` that lands at time index ``j + k`` is
``nu_j * components[:, j, k]`` when ``k <= kappa``, and
``nu_j * w[k - 1] * components[:, j, 0]`` when ``k > kappa``. Column 0 of
the sum is 0. The compensator stays the exact one,
``sum_j nu_j^2 ((t - t_j)^{2H} - (t - t_{j+1})^{2H}) / 2``.

Every price is computed under the trader's information. The forward
variance is the mixture of the regime-conditional curves, weighted by the
filtered probabilities. It is not computed from the true label ``r_t``.
At ``H = 1/2`` each regime-conditional expectation equals
``v_t xi0(u) / xi0(t)``, so the mixture equals that common curve.
``state["xi"]`` stores the mixture. ``VarianceSwapPricer`` integrates it.
``surface_at`` prices the call by Monte Carlo under the same mixture.
Each regime holds ``nu`` fixed. The variance level is that regime's
forward curve ``xi_t(u)``: at ``H = 1/2`` it is flat at ``v_t``, and at
any other ``H`` it is ``v_t`` times the constant-``nu`` rough ratio. The
future noise is a fresh kernel, independent of the past given that
curve. The past Volterra integral sits in ``xi_t(u)`` and is not drawn
again. Weights are ``regime_prob`` at the clock. The true label is not
an argument. At ``H = 1/2`` the variance step is the exponential
martingale and the plugin is not called. Any other ``H`` reads
``state["dW"]`` and restarts ``volterra_components`` from the clock.

The same shocks drive a Black path whose variance is the deterministic
left forward curve. The control is that Black price. Its mean is the
formula, not the sample mean, and the coefficient is the sample
``cov / var``. One terminal is used for every strike. ``n_mc`` defaults
to 1024 and must be at least 2. The returned standard error is the
sample standard deviation of the residual divided by ``sqrt(n_mc)``.
Fitting the coefficient on the same sample makes that error slightly
optimistic. ``total_variance`` is the controlled log contract
``-2 E[log Z]`` on those paths, which is the variance-swap strike, and
``total_variance_se`` is its standard error.

The Monte Carlo seed is a ``SeedSequence`` with the path's entropy and
``spawn_key + (i_t, 4, regime)``. It is not one of the four simulation
children. At ``H = 1/2`` each step draws ``dW`` for every path and every
copy, then the orthogonal shock. At any other ``H`` the plugin consumes
the vol child and the spot child draws the orthogonal shock. The unit
forward is 1 and the strike is ``exp(k)``.

The Hamilton filter reads one-step log realised variance of the simulated
spot,

    y_i = log( (log Z_i - log Z_{i-1})^2 / dt_i ).

There is no observation at ``t = 0``, and the filter probability there is
``p0``. The emission is Gaussian. The mean is the same in every regime,
``log(xi0) - gamma - log(2)``, which is ``E[log(xi0 * chi^2_1)]``, because
the return's variance is the spot variance and not ``nu``. The variance in
regime ``j`` is ``pi^2 / 2 + nu_j^2``. ``pi^2 / 2`` is about 4.93. For
``nu`` of 0.4 and 1.1 the extra term is 0.16 and 1.21, so the two
emissions are close and the filter does not beat the unconditional base
rate. A trailing window of the same series was tried. The chi-squared
noise in a window that still sits inside one regime visit dominates
``nu^2`` times the window length, and the hit rate stays near that base
rate. The observation is therefore still the one-step log realised
variance. A price from this information does not determine ``r_t``. The
test is a two-sided band on the hit rate minus the base rate.

Whether ``Z`` stays a martingale when ``nu`` switches is a hypothesis.
The log-Euler step freezes ``v`` at the left of the interval, so each
step is a martingale increment given the past. The test is the z-rule.

Draw order, one child of the run seed for each stream
(:func:`volsurface.markets.seeds.stream_seeds`). The streams are separate
generators, so draws in one do not move another.

- regime: one uniform per path for the initial state, then one uniform
  per path at each later time for the transition. The filter observation
  is computed from ``Z``. This stream does not draw it.
- vol: every Brownian increment, in time order, when ``H = 1/2``. When
  ``H`` is not 1/2 the plugin consumes this child inside
  ``volterra_components`` and this module draws nothing from it.
- spot: one normal per path per step, the shock orthogonal to ``dW``.
  ``dW_spot = rho dW + sqrt(1 - rho^2) dW_perp``.
- surrogate: spawned and not drawn.

The spot update is the left-point exponential,
``Z <- Z exp(sqrt(v) dW_spot - v dt / 2)``.
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
from volsurface.mc.rng import as_seed_sequence, generator, spawn
from volsurface.plugins import create

# E[log(chi^2_1)] = -gamma - log(2). Var(log(chi^2_1)) = trigamma(1/2) = pi^2 / 2.
_LOG_CHI2_MEAN = -float(np.euler_gamma) - math.log(2.0)
_LOG_CHI2_VAR = math.pi**2 / 2.0


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
        obs_sigma: float = 0.0,
        forward: float = 100.0,
        rate: float = 0.0,
        smile_k: np.ndarray | None = None,
        n_mc: int = 1024,
    ):
        self.spec = spec
        self.xi0 = float(xi0)
        self.rho = float(rho)
        self.H = float(H)
        self.obs_means = None if obs_means is None else np.asarray(obs_means, dtype=float)
        self.obs_sigma = float(obs_sigma)
        self.forward = float(forward)
        self.rate = float(rate)
        self.smile_k = (
            np.linspace(-0.2, 0.2, 9) if smile_k is None else np.asarray(smile_k, dtype=float)
        )
        self.n_mc = int(n_mc)
        if self.n_mc < 2:
            raise ValueError(f"n_mc must be at least 2, got {self.n_mc}")
        self._plugin = None

    def simulate_market(
        self,
        n_paths: int,
        times: np.ndarray,
        *,
        seed: int | np.random.SeedSequence,
        state0=None,
    ) -> MarketPaths:
        """Paths, the chain, the filter, and the trader's forward variance."""
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
        if self.H == 0.5:
            dW = np.diff(_brownian(n_paths, grid, streams["vol"]), axis=1)
            components = None
        else:
            components = self._rough_components(n_paths, grid, streams["vol"])
            dW = np.asarray(components[..., 0], dtype=float)
        steps: list[np.ndarray] = []
        for i in range(1, grid.size):
            dt = float(grid[i] - grid[i - 1])
            step = _stochastic(expm(generator_q * dt))
            steps.append(step)
            regime[:, i] = _categorical(regime_rng, step[regime[:, i - 1]])
        nu_step = nus[regime[:, :-1]]
        if components is None:
            weights = _kernel_weights(0.5, grid, grid)
            v = increment_variance(self.H, nu_step, self.xi0, grid, dW, weights)
        else:
            v = hybrid_increment_variance(self.H, nu_step, self.xi0, grid, components)
        Z = np.ones((n_paths, grid.size))
        rho = self.rho
        orthogonal = math.sqrt(max(0.0, 1.0 - rho * rho))
        for i in range(1, grid.size):
            dt = float(grid[i] - grid[i - 1])
            shock = rho * dW[:, i - 1] + orthogonal * math.sqrt(dt) * spot_rng.standard_normal(
                n_paths
            )
            left = np.maximum(v[:, i - 1], 0.0)
            Z[:, i] = Z[:, i - 1] * np.exp(np.sqrt(left) * shock - 0.5 * left * dt)
        observed = _log_realised_variance(Z, grid)
        means, variances = self._emission(nus)
        filtered = filter_log_realised_variance(observed, steps, means, variances, p0)
        Y = np.zeros((n_paths, grid.size))
        Y[:, 1:] = np.cumsum(dW, axis=1)
        xi = trader_forward_variance(self.H, nus, v, filtered, self.xi0, grid, dW)
        parent = as_seed_sequence(seed)
        plugin = self._plugin
        return MarketPaths(
            times=grid,
            Z=Z,
            state={"v": v, "Y": Y, "dW": dW, "y": observed, "xi": xi, "xi_tenor": grid.copy()},
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
                "roughvol": self.H != 0.5 and _is_roughvol(plugin),
                "label": "Brownian" if self.H == 0.5 else type(plugin).__name__,
            },
        )

    def surface_at(self, paths: MarketPaths, i_t: int) -> SurfaceBatch:
        """Monte Carlo calls mixed over the filtered regime probabilities.

        Each regime is simulated with that regime's ``nu`` held fixed.
        The variance level is the regime-conditional forward curve
        ``xi_t(u)``, and the future kernel is restarted from the clock.
        The draws do not read ``paths.regime``. Weights are ``regime_prob``
        at the clock. Two calls on the same paths use the same seed, so
        the prices match.
        """
        if paths.regime_prob is None:
            raise ValueError("surface_at mixes filtered regime probabilities")
        clock = int(i_t)
        times = np.asarray(paths.times, dtype=float)
        indices = _expiry_indices(times.size, clock)
        v_now = np.asarray(paths.state["v"][:, clock], dtype=float)
        nus = np.array([float(item["nu"]) for item in self.spec.params], dtype=float)
        probs = np.asarray(paths.regime_prob[:, clock, :], dtype=float)
        if probs.shape != (v_now.shape[0], nus.size):
            raise ValueError(
                f"regime_prob shape {probs.shape} does not match "
                f"{(v_now.shape[0], nus.size)} paths and regimes"
            )
        k = np.asarray(self.smile_k, dtype=float)
        strikes = np.exp(k)
        m = v_now.shape[0]
        if int(indices[-1]) == clock:
            call = _intrinsic_calls(m, strikes)
            return SurfaceBatch(
                T=times[indices].copy(),
                k=k.copy(),
                call=call,
                stderr=np.zeros_like(call),
                total_variance=np.zeros((m, 1)),
                total_variance_se=np.zeros((m, 1)),
            )
        dW = _forward_driver(paths, self.H)
        call = np.zeros((m, indices.size, k.size))
        stderr = np.zeros_like(call)
        total = np.zeros((m, indices.size))
        total_se = np.zeros_like(total)
        for r, nu in enumerate(nus):
            seed = _surface_seed(paths, clock, r)
            xi = _regime_forward(self.H, float(nu), v_now, self.xi0, times, dW, clock)
            column, column_se, vs, vs_se = self._regime_calls(
                float(nu), xi, times, clock, indices, strikes, seed
            )
            weight = probs[:, r]
            call += weight[:, None, None] * column
            stderr += (weight[:, None, None] * column_se) ** 2
            total += weight[:, None] * vs
            total_se += (weight[:, None] * vs_se) ** 2
        return SurfaceBatch(
            T=times[indices].copy(),
            k=k.copy(),
            call=call,
            stderr=np.sqrt(stderr),
            total_variance=total,
            total_variance_se=np.sqrt(total_se),
        )

    def _regime_calls(self, nu, xi, times, i_t, indices, strikes, seed):
        """Controlled calls for one constant ``nu``.

        Returns call, stderr, log-contract and its standard error.
        """
        if self.H == 0.5:
            return _half_calls(nu, xi, times, i_t, indices, strikes, seed, self.n_mc, self.rho)
        return self._rough_calls(nu, xi, times, i_t, indices, strikes, seed)

    def _rough_calls(self, nu, xi, times, i_t, indices, strikes, seed):
        """Restarted hybrid factor on the regime's forward curve ``xi``.

        The plugin draws the vol child. ``xi`` already holds the path's
        past, so the factor is a fresh kernel with mean near 1.
        """
        m = xi.shape[0]
        n_mc = self.n_mc
        last = int(indices[-1])
        n_steps = last - i_t
        grid = np.asarray(times[i_t : last + 1], dtype=float)
        _equal_steps(grid)
        grid = grid - float(grid[0])
        vol_seed, spot_seed = spawn(seed, 2)
        n_sim = m * n_mc
        components = self._rough_components(n_sim, grid, vol_seed)
        nu_step = np.full((n_sim, n_steps), float(nu))
        factor = hybrid_increment_variance(self.H, nu_step, 1.0, grid, components)
        level = np.repeat(np.maximum(xi[:, i_t : last + 1], 0.0), n_mc, axis=0)
        variance = level * factor
        rng = generator(spot_seed)
        log_z = np.zeros(n_sim)
        log_z_bs = np.zeros(n_sim)
        rho = self.rho
        ortho = math.sqrt(max(0.0, 1.0 - rho * rho))
        stored_z: dict[int, np.ndarray] = {}
        stored_bs: dict[int, np.ndarray] = {}
        wanted = {int(idx) for idx in indices}
        for step in range(n_steps):
            dt = float(grid[step + 1] - grid[step])
            dW = np.asarray(components[:, step, 0], dtype=float)
            dW_perp = math.sqrt(dt) * rng.standard_normal(n_sim)
            left = np.maximum(variance[:, step], 0.0)
            left_bs = level[:, step]
            shock = rho * dW + ortho * dW_perp
            log_z = log_z + np.sqrt(left) * shock - 0.5 * left * dt
            log_z_bs = log_z_bs + np.sqrt(left_bs) * shock - 0.5 * left_bs * dt
            done = i_t + step + 1
            if done in wanted:
                stored_z[done] = log_z.reshape(m, n_mc).copy()
                stored_bs[done] = log_z_bs.reshape(m, n_mc).copy()
        return _controlled_surface(xi, times, i_t, indices, strikes, stored_z, stored_bs)

    def _checked_spec(self) -> RegimeSpec:
        spec = self.spec
        if spec.switch != "nu":
            raise ValueError(f"this phase switches nu only, got {spec.switch!r}")
        if not math.isfinite(self.rho) or abs(self.rho) > 1.0:
            raise ValueError(f"rho must lie in [-1, 1], got {self.rho}")
        if self.obs_sigma < 0.0:
            raise ValueError(f"obs_sigma must be non-negative, got {self.obs_sigma}")
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
        if self.obs_means is not None and self.obs_means.shape != (n_regimes,):
            raise ValueError("obs_means must have one entry per regime")
        for item in spec.params:
            if "nu" not in item or float(item["nu"]) <= 0.0:
                raise ValueError("each regime needs a positive nu")
        if self.xi0 <= 0.0:
            raise ValueError(f"xi0 must be positive, got {self.xi0}")
        if self.H <= 0.0 or self.H > 0.5:
            raise ValueError(f"H must lie in (0, 1/2], got {self.H}")
        return spec

    def _emission(self, nus: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Gaussian emission of log realised variance. Means default to one value."""
        if self.obs_means is None:
            means = np.full(nus.shape, math.log(self.xi0) + _LOG_CHI2_MEAN)
        else:
            means = np.asarray(self.obs_means, dtype=float)
        variances = np.full(nus.shape, _LOG_CHI2_VAR) + nus**2 + self.obs_sigma**2
        return means, variances

    def _rough_components(self, n_paths: int, times: np.ndarray, seed) -> np.ndarray:
        """Hybrid components ``(dW, I_1, ..., I_kappa)``. ``H = 1/2`` does not call this."""
        try:
            plugin = create("rbergomi")
        except KeyError as exc:
            raise RuntimeError(
                "H other than 1/2 uses roughvol's volterra_components, and that plugin "
                "is not installed. This module does not simulate the hybrid scheme."
            ) from exc
        if not hasattr(plugin, "volterra_components"):
            raise RuntimeError(
                "the rbergomi plugin has no volterra_components. "
                "dW alone drops the near field I_1, ..., I_kappa, so E[v] falls short "
                "of xi0. This module does not simulate the hybrid scheme."
            )
        self._plugin = plugin
        components = np.asarray(
            plugin.volterra_components(n_paths, times, seed=seed, H=self.H), dtype=float
        )
        n_steps = times.size - 1
        if components.ndim != 3 or components.shape[:2] != (n_paths, n_steps):
            raise ValueError(
                f"volterra_components returned shape {components.shape}, "
                f"expected {(n_paths, n_steps, 'kappa + 1')}"
            )
        if components.shape[2] < 2:
            raise ValueError(
                "volterra_components must include the near field I_1, ..., I_kappa. "
                "dW alone keeps only part of Var(Y), and E[v] falls short of xi0."
            )
        return components


def _is_roughvol(plugin) -> bool:
    if plugin is None:
        return False
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


def _step_compensator(H: float, nu_step: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Exact compensator, shape ``(m, n_times)``.

    Entry ``(path, i)`` is ``sum_j nu_j^2 ((t_i - t_j)^{2H} - (t_i - t_{j+1})^{2H}) / 2``
    over steps ``j`` that finish at or before ``t_i``. Constant ``nu`` reduces
    to ``nu^2 t^{2H} / 2``. This is the variance of the integral, not the
    hybrid scheme's sample variance.
    """
    grid = np.asarray(times, dtype=float)
    nu = np.asarray(nu_step, dtype=float)
    two_h = 2.0 * float(H)
    left = grid[None, :] - grid[:-1, None]
    right = grid[None, :] - grid[1:, None]
    gap = np.maximum(left, 0.0) ** two_h - np.maximum(right, 0.0) ** two_h
    gap = np.where(right >= -1e-14, gap, 0.0)
    return (0.5 * nu**2) @ gap


def increment_variance(
    H: float,
    nu_step: np.ndarray,
    xi0: float,
    times: np.ndarray,
    dW: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Spot variance when ``nu`` multiplies each new increment only.

    ``nu_step`` has shape ``(m, n_steps)`` and is the vol of vol in force on
    each step, the regime at the left endpoint. ``dW`` has the same shape.
    ``weights`` has shape ``(n_steps, n_times)``. A switch changes later
    rows of ``nu_step`` and leaves the variance already accumulated as it was.
    At ``H`` other than 1/2 the market uses :func:`hybrid_increment_variance`,
    because these weights are the regression of the kernel onto ``dW`` and
    they drop the near-field variance.
    """
    hurst = float(H)
    increments = np.asarray(dW, dtype=float)
    nu = np.asarray(nu_step, dtype=float)
    grid = np.asarray(times, dtype=float)
    kernel = np.asarray(weights, dtype=float)
    if increments.ndim != 2:
        raise ValueError(f"dW must have shape (m, n_steps), got {increments.shape}")
    m, n_steps = increments.shape
    if nu.shape != increments.shape:
        raise ValueError(f"nu_step shape {nu.shape} does not match dW {increments.shape}")
    if grid.size != n_steps + 1 or kernel.shape != (n_steps, grid.size):
        raise ValueError(
            f"weights shape {kernel.shape} does not match times {grid.size} and {n_steps} steps"
        )
    scale = math.sqrt(2.0 * hurst)
    level = float(xi0)
    # weight[k, i] is already 0 once step k finishes after tenor i, so the
    # matrix product only accumulates increments that have happened.
    integral = (nu * increments) @ kernel
    variance = level * np.exp(scale * integral - _step_compensator(hurst, nu, grid))
    variance[:, 0] = level
    return variance


def hybrid_increment_variance(
    H: float,
    nu_step: np.ndarray,
    xi0: float,
    times: np.ndarray,
    components: np.ndarray,
) -> np.ndarray:
    """Spot variance from hybrid components ``(dW, I_1, ..., I_kappa)``.

    ``components`` has shape ``(m, n_steps, kappa + 1)``. Each step's vector
    is multiplied by the ``nu`` in force on that step. ``I_k`` lands at lag
    ``k``. The tail uses roughvol's power-kernel weights, whose first
    ``kappa`` entries are 0, so the near field is not added twice. The
    compensator is :func:`_step_compensator`.
    """
    hurst = float(H)
    nu = np.asarray(nu_step, dtype=float)
    grid = np.asarray(times, dtype=float)
    pieces = np.asarray(components, dtype=float)
    if pieces.ndim != 3:
        raise ValueError(f"components must have shape (m, n_steps, kappa + 1), got {pieces.shape}")
    m, n_steps, n_comp = pieces.shape
    if nu.shape != (m, n_steps) or grid.size != n_steps + 1:
        raise ValueError(
            f"nu_step {nu.shape} and times {grid.size} do not match components {pieces.shape}"
        )
    if n_comp < 2:
        raise ValueError(
            "components must include the near field I_1, ..., I_kappa. "
            "dW alone keeps only part of Var(Y), and E[v] falls short of xi0."
        )
    kappa = n_comp - 1
    weights = _hybrid_tail_weights(hurst, n_steps, float(grid[-1]), kappa)
    driver = _assemble_driver(nu, pieces, weights)
    level = float(xi0)
    scale = math.sqrt(2.0 * hurst)
    variance = level * np.exp(scale * driver - _step_compensator(hurst, nu, grid))
    variance[:, 0] = level
    return variance


def _hybrid_tail_weights(H: float, n_steps: int, horizon: float, kappa: int) -> np.ndarray:
    """Power-kernel tail weights. The first ``kappa`` lags are 0."""
    try:
        from roughvol.volterra import power_kernel_weights
    except ImportError as exc:
        raise RuntimeError(
            "H other than 1/2 uses roughvol's power-kernel weights for the hybrid tail. "
            "This module does not build those weights."
        ) from exc
    return np.asarray(power_kernel_weights(H, n_steps, horizon, kappa=kappa), dtype=float)


def _assemble_driver(nu: np.ndarray, pieces: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Nu-weighted Volterra integral. Column 0 is 0. Shape ``(m, n_times)``."""
    _m, n_steps, n_comp = pieces.shape
    kappa = n_comp - 1
    if weights.shape != (n_steps,):
        raise ValueError(f"tail weights must have shape {(n_steps,)}, got {weights.shape}")
    driver = np.zeros((nu.shape[0], n_steps + 1))
    # w[k - 1] is the weight at lag k. The first kappa lags are the near field.
    scaled = nu * pieces[:, :, 0]
    for lag in range(kappa + 1, n_steps + 1):
        weight = float(weights[lag - 1])
        if weight == 0.0:
            continue
        length = n_steps - lag + 1
        driver[:, lag : lag + length] += weight * scaled[:, :length]
    for lag in range(1, kappa + 1):
        length = n_steps - lag + 1
        if length <= 0:
            continue
        block = nu[:, :length] * pieces[:, :length, lag]
        driver[:, lag : lag + length] += block
    return driver


def trader_forward_variance(
    H: float,
    nus: np.ndarray,
    v: np.ndarray,
    regime_prob: np.ndarray,
    xi0: float,
    times: np.ndarray,
    dW: np.ndarray,
) -> np.ndarray:
    """Forward variance mixed over ``regime_prob``. Shape ``(m, n, n)``.

    The true regime label is not an argument. At ``H = 1/2`` every regime
    has the same conditional expectation, ``v_t`` when ``xi0`` is flat, so
    the mixture does not depend on the weights. At any other ``H`` the
    curve in the tenor is the constant-``nu`` rough ratio for each regime,
    anchored at the path's ``v_t``, then mixed. Behind the clock the entry
    is NaN. On the clock it is ``v_t``.
    """
    variance = np.asarray(v, dtype=float)
    grid = np.asarray(times, dtype=float)
    m, n = variance.shape
    if grid.size != n:
        raise ValueError(f"times length {grid.size} does not match v shape {variance.shape}")
    ahead = np.triu(np.ones((n, n), dtype=bool))
    if float(H) == 0.5:
        return np.where(ahead[None, :, :], variance[:, :, None], np.nan)
    vol_of_vol = np.asarray(nus, dtype=float).reshape(-1)
    probs = np.asarray(regime_prob, dtype=float)
    if probs.shape != (m, n, vol_of_vol.size):
        raise ValueError(
            f"regime_prob shape {probs.shape} does not match {(m, n, vol_of_vol.size)}"
        )
    mixed = np.zeros((m, n, n))
    increments = np.asarray(dW, dtype=float)
    for r, nu in enumerate(vol_of_vol):
        component = bergomi_forward_variance(float(H), float(nu), xi0, grid, increments)
        diag = np.diagonal(component, axis1=1, axis2=2)
        ratio = np.where(ahead[None, :, :], component / diag[:, :, None], 0.0)
        mixed += probs[:, :, r][:, :, None] * variance[:, :, None] * ratio
    return np.where(ahead[None, :, :], mixed, np.nan)


def filter_log_realised_variance(
    y: np.ndarray,
    steps: list[np.ndarray],
    means: np.ndarray,
    variances: np.ndarray,
    p0: np.ndarray,
) -> np.ndarray:
    """Hamilton filter of log realised variance. Shape ``(m, n_times, R)``.

    Column 0 of ``y`` is unused: there is no return at time 0, and the
    returned probability there is ``p0``. Each later column is the log
    realised variance on the step that ends at that time. That return is
    emitted by the regime at the left endpoint. The probability stored at
    the right endpoint is the updated belief, moved one transition forward.
    A later return does not enter.
    """
    series = np.asarray(y, dtype=float)
    if series.ndim != 2:
        raise ValueError(f"y must have shape (m, n_times), got {series.shape}")
    m, n_times = series.shape
    prior = np.asarray(p0, dtype=float).reshape(-1)
    out = np.empty((m, n_times, prior.size))
    out[:, 0, :] = prior
    n_steps = n_times - 1
    if n_steps == 0:
        return out
    if len(steps) != n_steps:
        raise ValueError(f"expected {n_steps} transition matrices, got {len(steps)}")
    observations = series[:, 1:]
    transition = np.eye(prior.size) if n_steps == 1 else steps[:-1]
    filtered = hamilton_filter(observations, transition, means, variances, prior)
    for i in range(1, n_times):
        out[:, i, :] = filtered[:, i - 1, :] @ steps[i - 1]
    return out


def _log_realised_variance(Z: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Log of ``(d log Z)^2 / dt``. Column 0 is NaN."""
    y = np.full(Z.shape, np.nan)
    dt = np.diff(np.asarray(times, dtype=float))
    log_ret = np.log(Z[:, 1:]) - np.log(Z[:, :-1])
    realised = log_ret**2 / dt[None, :]
    y[:, 1:] = np.log(np.maximum(realised, 1e-300))
    return y


def _expiry_indices(n_times: int, i_t: int) -> np.ndarray:
    """Later clock indices, at most seven, spread through to the last time."""
    if i_t < 0 or i_t >= n_times:
        raise ValueError(f"i_t must lie in [0, {n_times}), got {i_t}")
    future = np.arange(i_t + 1, n_times)
    if future.size == 0:
        return np.array([i_t], dtype=int)
    if future.size > 7:
        pick = np.linspace(0, future.size - 1, 7).astype(int)
        future = future[pick]
    return future


def _surface_seed(paths: MarketPaths, i_t: int, regime: int) -> np.random.SeedSequence:
    """Monte Carlo seed for one clock and one regime. Not a simulation child."""
    meta = paths.meta
    if "seed_entropy" not in meta or "spawn_key" not in meta:
        raise ValueError("surface_at draws from paths.meta seed_entropy and spawn_key")
    key = tuple(int(part) for part in meta["spawn_key"]) + (int(i_t), 4, int(regime))
    return np.random.SeedSequence(entropy=meta["seed_entropy"], spawn_key=key)


def _intrinsic_calls(n_paths: int, strikes: np.ndarray) -> np.ndarray:
    """Unit-forward intrinsic value, shape ``(m, 1, n_k)``."""
    intrinsic = np.maximum(1.0 - strikes, 0.0)
    return np.broadcast_to(intrinsic, (n_paths, 1, strikes.size)).copy()


def _forward_driver(paths: MarketPaths, H: float) -> np.ndarray | None:
    """``state["dW"]`` when the forward curve depends on it. ``H = 1/2`` skips it."""
    if float(H) == 0.5:
        return None
    if "dW" not in paths.state:
        raise ValueError(
            "surface_at at H other than 1/2 reads state['dW'] to build the "
            "regime-conditional forward variance xi_t(u). A flat anchor at v_t "
            "drops that curve."
        )
    return np.asarray(paths.state["dW"], dtype=float)


def _regime_forward(H, nu, v_now, xi0, times, dW, i_t) -> np.ndarray:
    """Regime-conditional forward variance. Shape ``(m, n_times)``.

    At ``H = 1/2`` the curve is flat at ``v_now``. At any other ``H`` it is
    ``v_now`` times the constant-``nu`` rough ratio ``xi_t(u) / xi_t(t)``.
    Entries before the clock are NaN. The true regime label is not an argument.
    """
    m = int(v_now.shape[0])
    out = np.full((m, times.size), np.nan)
    if float(H) == 0.5:
        out[:, i_t:] = np.maximum(v_now, 0.0)[:, None]
        return out
    curve = bergomi_forward_variance(float(H), float(nu), xi0, times, dW)
    level = curve[:, i_t, i_t]
    ratio = curve[:, i_t, i_t:] / level[:, None]
    out[:, i_t:] = np.maximum(v_now, 0.0)[:, None] * ratio
    return out


def _equal_steps(grid: np.ndarray) -> None:
    """The hybrid plugin draws on an equally spaced grid."""
    dt = np.diff(np.asarray(grid, dtype=float))
    if dt.size == 0:
        return
    span = max(1.0, float(grid[-1] - grid[0]))
    if float(np.max(np.abs(dt - dt[0]))) > 1e-10 * span:
        raise ValueError(
            "surface_at at H other than 1/2 needs equally spaced times "
            "from the clock to the last expiry"
        )


def _half_calls(nu, xi, times, i_t, indices, strikes, seed, n_mc, rho):
    """Exponential-martingale variance at ``H = 1/2``, with a Black control."""
    m = int(xi.shape[0])
    rng = generator(seed)
    multiplier = np.ones((m, n_mc))
    log_z = np.zeros((m, n_mc))
    log_z_bs = np.zeros((m, n_mc))
    ortho = math.sqrt(max(0.0, 1.0 - rho * rho))
    wanted = {int(i) for i in indices}
    stored_z: dict[int, np.ndarray] = {}
    stored_bs: dict[int, np.ndarray] = {}
    last = int(indices[-1])
    for i in range(i_t + 1, last + 1):
        dt = float(times[i] - times[i - 1])
        scale = math.sqrt(dt)
        dW = scale * rng.standard_normal((m, n_mc))
        dW_perp = scale * rng.standard_normal((m, n_mc))
        level = np.maximum(xi[:, i - 1], 0.0)
        left = level[:, None] * multiplier
        shock = rho * dW + ortho * dW_perp
        log_z = log_z + np.sqrt(left) * shock - 0.5 * left * dt
        log_z_bs = log_z_bs + np.sqrt(level)[:, None] * shock - 0.5 * level[:, None] * dt
        multiplier = multiplier * np.exp(nu * dW - 0.5 * nu * nu * dt)
        if i in wanted:
            stored_z[i] = log_z.copy()
            stored_bs[i] = log_z_bs.copy()
    return _controlled_surface(xi, times, i_t, indices, strikes, stored_z, stored_bs)


def _controlled_surface(xi, times, i_t, indices, strikes, stored_z, stored_bs):
    """Black control at the left Riemann of ``xi``, one terminal for every strike."""
    m = xi.shape[0]
    call = np.empty((m, indices.size, strikes.size))
    stderr = np.empty_like(call)
    total = np.empty((m, indices.size))
    total_se = np.empty_like(total)
    clock = float(times[i_t])
    for j, idx in enumerate(indices):
        maturity = float(times[int(idx)])
        integrated = integrate_forward_variance(xi, times, clock, maturity)
        tau = maturity - clock
        z = stored_z[int(idx)]
        z_bs = stored_bs[int(idx)]
        y = np.maximum(np.exp(z)[:, :, None] - strikes, 0.0)
        x = np.maximum(np.exp(z_bs)[:, :, None] - strikes, 0.0)
        known = _black_calls(integrated, tau, strikes)
        call[:, j, :], stderr[:, j, :] = _control_means(y, x, known)
        total[:, j], total_se[:, j] = _controlled_log_contract(z, z_bs, integrated)
    return call, stderr, total, total_se


def _black_calls(total: np.ndarray, tau: float, strikes: np.ndarray) -> np.ndarray:
    """Unit-forward Black calls. ``total`` is the left Riemann, shape ``(m,)``."""
    if tau <= 0.0:
        intrinsic = np.maximum(1.0 - strikes, 0.0)
        return np.broadcast_to(intrinsic, (total.shape[0], strikes.size)).copy()
    sigma = np.sqrt(np.maximum(total, 0.0) / tau)
    return np.asarray(bs.price(1.0, strikes[None, :], tau, sigma[:, None], 1.0), dtype=float)


def _control_means(y: np.ndarray, x: np.ndarray, known: np.ndarray):
    """Controlled mean and standard error.

    ``y`` and ``x`` have shape ``(m, n_mc, n_k)``. ``known`` has shape
    ``(m, n_k)`` and is the mean of ``x``, not the sample mean. The
    coefficient is the sample ``cov(y, x) / var(x)``. A tiny variance sets
    it to 0. The standard error divides the residual's sample standard
    deviation by ``sqrt(n_mc)``. Fitting the coefficient on the same sample
    makes that error slightly optimistic.
    """
    n_mc = y.shape[1]
    y_centred = y - y.mean(axis=1, keepdims=True)
    x_centred = x - x.mean(axis=1, keepdims=True)
    var_x = np.sum(x_centred * x_centred, axis=1) / (n_mc - 1)
    cov = np.sum(y_centred * x_centred, axis=1) / (n_mc - 1)
    beta = np.zeros_like(cov)
    usable = var_x > 1e-18
    beta[usable] = cov[usable] / var_x[usable]
    residual = y - beta[:, None, :] * (x - known[:, None, :])
    mean = residual.mean(axis=1)
    stderr = residual.std(axis=1, ddof=1) / math.sqrt(n_mc)
    return mean, stderr


def _controlled_log_contract(log_z, log_z_bs, total):
    """``-2 E[log Z]`` controlled by the Black total variance."""
    mean, stderr = _control_means(
        -2.0 * log_z[:, :, None], -2.0 * log_z_bs[:, :, None], total[:, None]
    )
    return mean[:, 0], stderr[:, 0]


def bergomi_forward_variance(H: float, nu, xi0, times: np.ndarray, dW: np.ndarray) -> np.ndarray:
    """Closed form of the rough Bergomi forward variance. Shape ``(m, n + 1, n + 1)``.

    ``dW`` has shape ``(m, n)`` and ``times`` has length ``n + 1``. ``nu`` is a
    scalar, shape ``(m,)``, or shape ``(m, n + 1)``. In the last case the vol
    of vol at the clock is used for every later tenor. ``xi0`` is a positive
    float or a callable of the tenor grid.

    For ``u > t``,

        I_t(u) = int_0^t (u - s)^{H - 1/2} dW_s,
        xi_t(u) = xi0(u) exp(nu sqrt(2H) I_t(u) - nu^2 (u^{2H} - (u - t)^{2H}) / 2).

    On the clock the entry is ``v_t``. Behind the clock it is NaN. At
    ``H = 1/2`` the kernel is 1, so ``I_t(u) = W_t`` and the exponent does
    not depend on ``u``. This is the constant-``nu`` curve. A regime path
    uses :func:`increment_variance` and :func:`trader_forward_variance`.
    """
    grid = np.asarray(times, dtype=float)
    increments = np.asarray(dW, dtype=float)
    if increments.ndim != 2 or increments.shape[1] != grid.size - 1:
        raise ValueError(f"dW must have shape (m, {grid.size - 1}), got {increments.shape}")
    m = increments.shape[0]
    nu_arr = np.asarray(nu, dtype=float)
    if nu_arr.ndim == 0:
        nu_clock = np.full((m, grid.size), float(nu_arr))
    elif nu_arr.shape == (m,):
        nu_clock = np.broadcast_to(nu_arr[:, None], (m, grid.size)).copy()
    elif nu_arr.shape == (m, grid.size):
        nu_clock = np.asarray(nu_arr, dtype=float)
    else:
        raise ValueError(f"nu shape {nu_arr.shape} does not match {m} paths")
    if callable(xi0):
        level = np.asarray(xi0(grid), dtype=float)
        if level.shape != grid.shape:
            raise ValueError(f"xi0 returned shape {level.shape}, expected {grid.shape}")
    else:
        level = np.full(grid.shape, float(xi0))
    weights = _kernel_weights(float(H), grid, grid)
    stacked = increments[:, :, None] * weights[None, :, :]
    integral = np.zeros((m, grid.size, grid.size))
    integral[:, 1:, :] = np.cumsum(stacked, axis=1)
    gap = grid[None, :] - grid[:, None]
    tol = 1e-12 * max(1.0, float(grid[-1]))
    ahead = gap > tol
    safe = np.where(ahead, gap, 1.0)
    u_power = np.zeros_like(grid)
    positive = grid > 0.0
    u_power[positive] = grid[positive] ** (2.0 * float(H))
    nu_b = nu_clock[:, :, None]
    gap_power = np.zeros_like(safe)
    gap_power[ahead] = safe[ahead] ** (2.0 * float(H))
    compensator = np.where(
        ahead[None, :, :],
        0.5 * nu_b**2 * (u_power[None, None, :] - gap_power[None, :, :]),
        0.0,
    )
    scale = math.sqrt(2.0 * float(H))
    xi = level[None, None, :] * np.exp(nu_b * scale * integral - compensator)
    diagonal = np.arange(grid.size)
    y_diag = integral[:, diagonal, diagonal]
    clock_power = np.zeros_like(grid)
    clock_power[positive] = grid[positive] ** (2.0 * float(H))
    variance = level[None, :] * np.exp(
        nu_clock * scale * y_diag - 0.5 * nu_clock**2 * clock_power[None, :]
    )
    xi[:, diagonal, diagonal] = variance
    behind = gap < -tol
    if np.any(behind):
        xi = np.where(behind[None, :, :], np.nan, xi)
    return xi


def _kernel_weights(H: float, times: np.ndarray, tenors: np.ndarray) -> np.ndarray:
    """``int_{step} (u - s)^{H - 1/2} ds / dt``. Shape ``(n_steps, n_tenors)``.

    A step that finishes after the tenor has weight 0. At ``H = 1/2`` the
    kernel is 1 on every step that finishes at or before the tenor.
    """
    alpha = H - 0.5
    t0 = times[:-1]
    t1 = times[1:]
    dt = t1 - t0
    left = tenors[None, :] - t0[:, None]
    right = tenors[None, :] - t1[:, None]
    if abs(alpha) < 1e-12:
        weights = np.ones((t0.size, tenors.size))
    else:
        power = alpha + 1.0
        integral = (np.maximum(left, 0.0) ** power - np.maximum(right, 0.0) ** power) / power
        weights = integral / dt[:, None]
    return np.where(right >= -1e-14, weights, 0.0)


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
