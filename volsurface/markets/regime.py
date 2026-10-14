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
``volterra_increments`` on the rBergomi plugin. That method returns the
driving Wiener increments. This module applies the kernel weights itself
and puts ``nu`` inside the sum. A missing plugin, or a plugin without that
method, raises. This file does not simulate the hybrid scheme, and a
stand-in is not labelled rBergomi.

The Volterra interface this module calls, which roughvol does not yet
provide:

    volterra_increments(n_paths, times, *, seed, H) -> dW

``times`` has shape ``(n_steps + 1,)``, starts at 0, and is strictly
increasing. ``dW`` has shape ``(n_paths, n_steps)``. Entry ``[i, k]`` is
``W(times[k + 1]) - W(times[k])`` for path ``i``, with variance equal to
the step length. ``seed`` is already the volatility child. The draw is the
one ``hybrid_scheme_rng`` makes from ``generator(seed)``: do not spawn
another child. On a uniform grid that is the ``dW`` behind
``RoughBergomi.simulate`` at this plugin's ``kappa``. Returning only the
summed factor ``Y_t = int (t - s)^{H - 1/2} dW_s`` is not enough, because
``nu`` has to sit inside the integral. The kernel weights are the
antiderivative already in :func:`_kernel_weights`, the same weights as
``roughvol``'s forward-curve step weights. They do not need to be returned.

Every price is computed under the trader's information. The forward
variance is the mixture of the regime-conditional curves, weighted by the
filtered probabilities. It is not computed from the true label ``r_t``.
At ``H = 1/2`` each regime-conditional expectation equals
``v_t xi0(u) / xi0(t)``, so the mixture equals that common curve.
``state["xi"]`` stores the mixture. ``VarianceSwapPricer`` integrates it.
``surface_at`` prices each regime's Black call and mixes those prices with
``regime_prob``.

The Hamilton filter reads log realised variance of the simulated spot,

    y_i = log( (log Z_i - log Z_{i-1})^2 / dt_i ).

There is no observation at ``t = 0``, and the filter probability there is
``p0``. The emission is Gaussian, the same mean in every regime,
``log(xi0) - gamma - log(2)``, which is ``E[log(xi0 * chi^2_1)]``. The
variance in regime ``j`` is ``pi^2 / 2 + nu_j^2``, the variance of
``log(chi^2_1)`` plus a stated ``nu_j^2`` term. A price from this
information does not determine ``r_t``.

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
  ``volterra_increments`` and this module draws nothing from it.
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
from volsurface.mc.rng import as_seed_sequence, generator
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
        dW, weights = self._increments(n_paths, grid, streams["vol"])
        steps: list[np.ndarray] = []
        for i in range(1, grid.size):
            dt = float(grid[i] - grid[i - 1])
            step = _stochastic(expm(generator_q * dt))
            steps.append(step)
            regime[:, i] = _categorical(regime_rng, step[regime[:, i - 1]])
        nu_step = nus[regime[:, :-1]]
        v = increment_variance(self.H, nu_step, self.xi0, grid, dW, weights)
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
        """Black calls mixed over the filtered regime probabilities.

        Each regime is priced from its own forward curve, anchored at the
        path's variance. The curves do not read ``paths.regime``. At
        ``H = 1/2`` the curves agree, so the mixture is that common price.
        """
        if paths.regime_prob is None:
            raise ValueError("surface_at mixes filtered regime probabilities")
        probs = np.asarray(paths.regime_prob[:, i_t, :], dtype=float)
        tenors = np.asarray(paths.state["xi_tenor"], dtype=float)
        now = float(paths.times[i_t])
        future = tenors[tenors > now + 1e-15]
        if future.size == 0:
            future = tenors[-1:]
        if future.size > 7:
            future = future[np.linspace(0, future.size - 1, 7).astype(int)]
        nus = np.array([float(item["nu"]) for item in self.spec.params], dtype=float)
        dW = np.asarray(paths.state["dW"], dtype=float)
        v_now = np.asarray(paths.state["v"][:, i_t], dtype=float)
        k = self.smile_k
        strikes = np.exp(k)
        call = np.zeros((paths.Z.shape[0], future.size, k.size))
        for r, nu in enumerate(nus):
            weight = probs[:, r]
            row = _anchored_row(self.H, float(nu), v_now, self.xi0, paths.times, dW, i_t)
            for j, maturity in enumerate(future):
                total = integrate_forward_variance(row, tenors, now, float(maturity))
                vol = np.sqrt(np.maximum(total, 0.0))
                for a, strike in enumerate(strikes):
                    call[:, j, a] += weight * bs.price(1.0, float(strike), 1.0, vol, 1.0)
        return SurfaceBatch(T=np.asarray(future, dtype=float), k=k.copy(), call=call)

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

    def _increments(self, n_paths: int, times: np.ndarray, seed):
        """Wiener increments and kernel weights. ``H = 1/2`` does not call the plugin."""
        if self.H == 0.5:
            dW = np.diff(_brownian(n_paths, times, seed), axis=1)
            return dW, _kernel_weights(0.5, times, times)
        try:
            plugin = create("rbergomi")
        except KeyError as exc:
            raise RuntimeError(
                "H other than 1/2 uses roughvol's volterra_increments, and that plugin "
                "is not installed. This module does not simulate the hybrid scheme."
            ) from exc
        if not hasattr(plugin, "volterra_increments"):
            raise RuntimeError(
                "the rbergomi plugin has no volterra_increments. "
                "This module needs the Wiener increments dW, not a summed driver Y, "
                "and it does not simulate the hybrid scheme."
            )
        self._plugin = plugin
        dW = np.asarray(
            plugin.volterra_increments(n_paths, times, seed=seed, H=self.H), dtype=float
        )
        if dW.shape != (n_paths, times.size - 1):
            raise ValueError(
                f"volterra_increments returned shape {dW.shape}, "
                f"expected {(n_paths, times.size - 1)}"
            )
        return dW, _kernel_weights(self.H, times, times)


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
    two_h = 2.0 * hurst
    level = float(xi0)
    # weight[k, i] is already 0 once step k finishes after tenor i, so the
    # matrix product only accumulates increments that have happened.
    integral = (nu * increments) @ kernel
    left = grid[None, :] - grid[:-1, None]
    right = grid[None, :] - grid[1:, None]
    gap = np.maximum(left, 0.0) ** two_h - np.maximum(right, 0.0) ** two_h
    gap = np.where(right >= -1e-14, gap, 0.0)
    compensator = (0.5 * nu**2) @ gap
    variance = level * np.exp(scale * integral - compensator)
    variance[:, 0] = level
    return variance


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


def _anchored_row(
    H: float,
    nu: float,
    v_now: np.ndarray,
    xi0: float,
    times: np.ndarray,
    dW: np.ndarray,
    i_t: int,
) -> np.ndarray:
    """One regime's forward curve at one clock time, anchored at ``v_now``."""
    grid = np.asarray(times, dtype=float)
    current = np.asarray(v_now, dtype=float)
    row = np.full((current.size, grid.size), np.nan)
    if float(H) == 0.5:
        row[:, i_t:] = current[:, None]
        return row
    component = bergomi_forward_variance(float(H), float(nu), xi0, grid, dW)[:, i_t, :]
    clock = component[:, i_t]
    future = np.arange(grid.size) >= i_t
    row[:, future] = current[:, None] * component[:, future] / clock[:, None]
    return row


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
