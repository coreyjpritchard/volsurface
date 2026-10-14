"""SABR (Hagan, Kumar, Lesniewski, Woodward 2002).

The forward and its volatility, under the forward measure::

    dF = alpha * F^beta dW
    d(alpha) = nu * alpha dZ,    corr(dW, dZ) = rho dt

``price`` is Black's formula at the Hagan lognormal implied volatility.
The at-the-money strike uses the closed limit, so the code does not
evaluate ``z / x(z)`` at ``z = 0``. For any other strike

    z = (nu / alpha) * (F K)^((1 - beta) / 2) * log(F / K)
    x(z) = log( (sqrt(1 - 2 rho z + z^2) + z - rho) / (1 - rho) )

and ``z / x(z)`` multiplies the leading term. ``|z| <= 1e-6`` uses
``1 - rho z / 2``, which matches that ratio through order ``z``. ``beta``
is a parameter. It is not eliminated.

``simulate`` is log-Euler on the unit-mean forward ``Z``, with
``dZ = a Z^beta dW`` and ``da = nu a dZ_vol``. Draw order, one child of
the seed: at each time after 0, ``n_paths`` standard normals for the
forward, then ``n_paths`` standard normals for the volatility. The
volatility shock is ``rho Z_F + sqrt(1 - rho^2) Z_vol``. ``euler_price``
sets the initial ``a`` to ``alpha * F^(beta - 1)``, which is the
coefficient in the ``Z`` equation when the contract forward is ``F``.
For ``beta = 1`` that factor is 1.

For ``beta < 1`` the continuous process can reach 0 and must stay there.
Log-Euler of a positive number stays positive, so the step also forms the
normal predictor ``Z + a Z^beta sqrt(dt) Z_F``. If that predictor is not
strictly positive, ``Z`` is set to 0 and later steps leave it there.
``beta = 1`` does not absorb: 0 is not attainable, and the log-Euler step
is the exact lognormal transition for a volatility that is constant on
the step.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from volsurface import bs
from volsurface.core.params import from_unconstrained, to_unconstrained
from volsurface.core.types import ParamSpec, Paths, PriceResult, State
from volsurface.mc.grid import estimate_terminal
from volsurface.mc.rng import generator, spawn
from volsurface.plugins import register

# |z| at or below this uses the first-order expansion of z/x(z). At z = 0
# the ratio is exactly 1. The linear term is -rho z / 2.
_Z_SERIES = 1e-6


def _spec() -> list[ParamSpec]:
    return [
        ParamSpec("alpha", lower=1e-4, upper=2.0, default=0.2, transform="log"),
        ParamSpec("beta", lower=0.0, upper=1.0, default=1.0, transform="identity"),
        ParamSpec("rho", lower=-0.999, upper=0.999, default=-0.3, transform="logit"),
        ParamSpec("nu", lower=1e-8, upper=4.0, default=0.3, transform="log"),
    ]


def free_spec() -> list[ParamSpec]:
    """``alpha``, ``rho``, ``nu``. Calibration holds ``beta`` fixed."""
    names = {"alpha", "rho", "nu"}
    return [spec for spec in _spec() if spec.name in names]


@register("sabr")
@dataclass(frozen=True)
class SABR:
    """Hagan's SABR. ``price`` is the 2002 lognormal formula. See the module note.

    ``n_paths``, ``n_steps`` and ``seed`` are the defaults for
    :func:`euler_price`. The formula does not read them.
    """

    alpha: float
    beta: float
    rho: float
    nu: float
    n_paths: int = 8_000
    n_steps: int = 32
    seed: int = 0

    def __post_init__(self) -> None:
        if self.alpha <= 0.0:
            raise ValueError(f"alpha must be positive, got {self.alpha}")
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError(f"beta must lie in [0, 1], got {self.beta}")
        if not -1.0 < self.rho < 1.0:
            raise ValueError(f"rho must lie in (-1, 1), got {self.rho}")
        if self.nu < 0.0:
            raise ValueError(f"nu must be non-negative, got {self.nu}")
        if self.n_paths < 2:
            raise ValueError(f"n_paths must be at least 2, got {self.n_paths}")
        if self.n_steps < 1:
            raise ValueError(f"n_steps must be at least 1, got {self.n_steps}")

    def initial_state(self) -> State:
        """The volatility ``alpha``. ``beta``, ``rho`` and ``nu`` stay on the model."""
        return {"alpha": np.array(self.alpha)}

    def _alpha(self, state: State | None) -> float:
        if state is None or "alpha" not in state:
            return float(self.alpha)
        value = np.asarray(state["alpha"], dtype=float)
        if value.shape == () or value.size == 1:
            return float(value.reshape(()))
        raise ValueError(
            "SABR.price uses one volatility. "
            f"Got state['alpha'] with shape {value.shape}."
        )

    def implied_vol(self, F, K, T, *, state: State | None = None) -> np.ndarray:
        """Black implied vol of the Hagan formula. Shape of ``K``.

        ``T = 0`` returns ``alpha / F^(1 - beta)``, the leading factor with
        the maturity correction switched off.
        """
        alpha = self._alpha(state)
        return hagan_implied_vol(F, K, T, alpha, self.beta, self.rho, self.nu)

    def price(self, F, K, T, df=1.0, kind="call", *, state: State | None = None):
        """Cash price: Black at :meth:`implied_vol`.

        A scalar ``state['alpha']`` replaces ``alpha``.
        """
        sigma = self.implied_vol(F, K, T, state=state)
        return bs.price(F, K, T, sigma, df, kind)

    def simulate(
        self, n_paths: int, times: np.ndarray, *,
        seed: int | np.random.SeedSequence, state0: State | None = None,
    ) -> Paths:
        """Unit-mean ``Z`` for a unit forward. See the module note for ``F != 1``.

        ``state0['alpha']`` is the initial coefficient ``a`` in
        ``dZ = a Z^beta dW``. Omitted, it is ``self.alpha``, which is the
        right coefficient only when the forward equals 1 or ``beta`` is 1.
        """
        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or times.size < 1 or times[0] != 0.0:
            raise ValueError("times must be 1-d and start at 0")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("times must be strictly increasing")
        n_paths = int(n_paths)
        if state0 is None or "alpha" not in state0:
            a0 = float(self.alpha)
        else:
            raw = np.asarray(state0["alpha"], dtype=float)
            if raw.shape != () and raw.size != 1:
                raise ValueError(f"state0['alpha'] must be a scalar, got shape {raw.shape}")
            a0 = float(raw.reshape(()))
        if a0 <= 0.0:
            raise ValueError(f"initial alpha must be positive, got {a0}")
        rng = generator(spawn(seed, 1)[0])
        Z, alpha = _log_euler(
            a0, self.beta, self.rho, self.nu, times, n_paths, rng,
        )
        return Paths(times=times, Z=Z, state={"alpha": alpha})

    @classmethod
    def param_spec(cls) -> list[ParamSpec]:
        return _spec()

    def to_unconstrained(self) -> np.ndarray:
        values = {
            "alpha": self.alpha,
            "beta": self.beta,
            "rho": self.rho,
            "nu": self.nu,
        }
        return to_unconstrained(values, _spec())

    @classmethod
    def from_unconstrained(cls, x: np.ndarray) -> SABR:
        values = from_unconstrained(np.asarray(x, dtype=float), _spec())
        return cls(**values)


def hagan_implied_vol(F, K, T, alpha: float, beta: float, rho: float, nu: float) -> np.ndarray:
    """Hagan's lognormal Black vol, with the at-the-money limit in closed form.

    ``F`` is one forward. ``K`` is an array of strikes. ``T`` is one maturity.
    Strikes must be positive.
    """
    forward = float(F)
    if forward <= 0.0:
        raise ValueError(f"F must be positive, got {forward}")
    if alpha <= 0.0:
        raise ValueError(f"alpha must be positive, got {alpha}")
    maturity = float(T)
    strike_in = np.asarray(K, dtype=float)
    strike = np.atleast_1d(strike_in).astype(float)
    if np.any(strike <= 0.0):
        raise ValueError("K must be positive")
    one_minus = 1.0 - float(beta)
    log_moneyness = np.log(forward / strike)
    atm = np.abs(log_moneyness) <= 1e-12
    leading_atm = alpha / forward**one_minus
    correction_atm = _correction(forward, forward, alpha, beta, rho, nu)
    if maturity == 0.0:
        sigma = np.full(strike.shape, leading_atm)
        return sigma.reshape(strike_in.shape)
    sigma_atm = leading_atm * (1.0 + correction_atm * maturity)
    fk_half = (forward * strike) ** (0.5 * one_minus)
    z = (nu / alpha) * fk_half * log_moneyness
    numerator = alpha * _z_over_x(z, rho)
    denominator = fk_half * (
        1.0
        + (one_minus**2) / 24.0 * log_moneyness**2
        + (one_minus**4) / 1920.0 * log_moneyness**4
    )
    correction = _correction_array(forward, strike, alpha, beta, rho, nu)
    sigma = (numerator / denominator) * (1.0 + correction * maturity)
    sigma = np.where(atm, sigma_atm, sigma)
    return sigma.reshape(strike_in.shape)


def euler_price(
    model: SABR, F, K, T, df=1.0, kind="call", *,
    state: State | None = None, n_paths: int | None = None,
    n_steps: int | None = None, seed: int | np.random.SeedSequence | None = None,
) -> PriceResult:
    """Log-Euler price and its standard error. Strikes share the paths.

    The initial volatility in the ``Z`` equation is ``alpha * F^(beta - 1)``.
    ``state['alpha']``, when given, is that SABR ``alpha`` before the scaling.
    """
    started = time.perf_counter()
    n_paths = model.n_paths if n_paths is None else int(n_paths)
    n_steps = model.n_steps if n_steps is None else int(n_steps)
    seed = model.seed if seed is None else seed
    if n_steps < 1:
        raise ValueError(f"n_steps must be at least 1, got {n_steps}")
    forward = float(F)
    maturity = float(T)
    strike_in = np.asarray(K, dtype=float)
    strike = np.atleast_1d(strike_in)
    if maturity == 0.0:
        value = _intrinsic(forward, strike, float(df), kind)
        return PriceResult(
            value=value.reshape(strike_in.shape),
            stderr=np.zeros(strike_in.shape, dtype=float),
            n_evals=0,
            wall_time=time.perf_counter() - started,
        )
    alpha = model._alpha(state)
    scaled = alpha * forward ** (model.beta - 1.0)
    times = np.linspace(0.0, maturity, n_steps + 1)
    paths = model.simulate(
        n_paths, times, seed=seed, state0={"alpha": np.array(scaled)},
    )
    value, stderr = estimate_terminal(
        paths.Z[:, -1], forward, strike, float(df), kind,
        antithetic=False, control_variate=False,
    )
    return PriceResult(
        value=value.reshape(strike_in.shape),
        stderr=stderr.reshape(strike_in.shape),
        n_evals=int(paths.Z.shape[0]),
        wall_time=time.perf_counter() - started,
    )


def _correction(
    forward: float, strike: float, alpha: float, beta: float, rho: float, nu: float,
) -> float:
    """The bracket that multiplies ``T`` in Hagan's formula, at one strike."""
    one_minus = 1.0 - beta
    fk_half = (forward * strike) ** (0.5 * one_minus)
    fk_full = (forward * strike) ** one_minus
    return float(
        (one_minus**2) / 24.0 * alpha**2 / fk_full
        + 0.25 * rho * beta * nu * alpha / fk_half
        + (2.0 - 3.0 * rho**2) / 24.0 * nu**2
    )


def _correction_array(forward, strike, alpha, beta, rho, nu):
    one_minus = 1.0 - float(beta)
    fk_half = (forward * strike) ** (0.5 * one_minus)
    fk_full = (forward * strike) ** one_minus
    return (
        (one_minus**2) / 24.0 * alpha**2 / fk_full
        + 0.25 * rho * beta * nu * alpha / fk_half
        + (2.0 - 3.0 * rho**2) / 24.0 * nu**2
    )


def _z_over_x(z: np.ndarray, rho: float) -> np.ndarray:
    """``z / x(z)``. Exactly 1 at ``z = 0``. First-order expansion for small ``z``."""
    z = np.asarray(z, dtype=float)
    small = np.abs(z) <= _Z_SERIES
    series = 1.0 - 0.5 * rho * z
    disc = np.maximum(1.0 - 2.0 * rho * z + z * z, 0.0)
    numer = np.sqrt(disc) + z - rho
    ratio = numer / (1.0 - rho)
    # A round-off that pushes the argument below 1 is still a valid x near 0,
    # and that region is already on the series. The floor only protects log.
    log_arg = np.maximum(ratio, 1e-300)
    x = np.log(log_arg)
    with np.errstate(divide="ignore", invalid="ignore"):
        exact = np.where(np.abs(x) > 0.0, z / x, 1.0)
    return np.where(small, series, exact)


def _log_euler(a0, beta, rho, nu, times, n_paths, rng):
    """Log-Euler. For ``beta < 1``, a non-positive normal predictor absorbs at 0."""
    Z = np.empty((n_paths, times.size), dtype=float)
    alpha = np.empty((n_paths, times.size), dtype=float)
    Z[:, 0] = 1.0
    alpha[:, 0] = a0
    level = np.ones(n_paths, dtype=float)
    vol = np.full(n_paths, a0, dtype=float)
    absorb = float(beta) < 1.0
    for i in range(1, times.size):
        dt = float(times[i] - times[i - 1])
        sqrt_dt = np.sqrt(dt)
        shock_f = rng.standard_normal(n_paths)
        shock_v = rng.standard_normal(n_paths)
        mixed = rho * shock_f + np.sqrt(1.0 - rho * rho) * shock_v
        # Both updates read the volatility at the left edge of the step.
        left = vol
        alive = level > 0.0
        diffusion = np.zeros(n_paths, dtype=float)
        diffusion[alive] = left[alive] * np.power(level[alive], beta) * sqrt_dt * shock_f[alive]
        killed = alive & absorb & (level + diffusion <= 0.0)
        stay = alive & ~killed
        updated = np.zeros(n_paths, dtype=float)
        if np.any(stay):
            local = left[stay] * np.power(level[stay], beta - 1.0)
            log_level = (
                np.log(level[stay]) - 0.5 * local * local * dt + local * sqrt_dt * shock_f[stay]
            )
            updated[stay] = np.exp(log_level)
        level = updated
        vol = np.exp(np.log(left) - 0.5 * nu * nu * dt + nu * sqrt_dt * mixed)
        Z[:, i] = level
        alpha[:, i] = vol
    return Z, alpha


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
