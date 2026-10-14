"""Heston as a :class:`~volsurface.core.protocols.Model`.

Dynamics, under the forward measure::

    dF / F = sqrt(v) dW
    dv    = kappa (theta - v) dt + xi sqrt(v) dZ,    corr(dW, dZ) = rho dt

``price`` is the Lewis (2000) integral in :mod:`volsurface.heston`. That is
the reference. ``cos_price`` is the Fang and Oosterlee (2008) cosine
expansion: the put is expanded, and the call comes back by put-call parity.
The put payoff is bounded by the strike. The call payoff grows with the
forward, so the same truncated interval leaves a larger cosine error on the
call. Parity adds ``df (F - K)`` and uses ``E[F_T] = F``, which the
characteristic function already satisfies (``phi(-i) = 1``).

The cosine interval is ``c1 ± L sqrt(c2 + sqrt(c4))`` with ``L = 10`` and
``N = 256`` terms. Those cumulants are :meth:`Heston.cumulants`. A negative
fourth cumulant, or one larger than ``80 c2²``, is replaced by zero before
the square root. The degree-8 stencil that produces ``c4`` does that when
``xi`` is small enough that the law is nearly Gaussian and the fourth
derivative is round-off. The replacement widens nothing: the interval is
then the Gaussian one.

``simulate`` is Euler full truncation (Lord, Koekkoek and van Dijk 2010) on
the caller's time grid. Draw order, one child of the seed for the whole run:
at each time after 0, a vector of ``n_paths`` standard normals for the
forward (``Z_S``), then a vector of ``n_paths`` standard normals for the
variance (``Z_v``). The variance shock is ``rho Z_S + sqrt(1 - rho²) Z_v``.
The variance that enters the drift and the diffusion is ``max(v, 0)``. The
value written into the next column is not floored, so a negative variance is
visible and is truncated on the following step. ``Z`` starts at 1. No other
draws are made.

The variance state is the scalar ``v`` that the Fourier formulas accept. A
vector of variances, one per path, is what ``simulate`` writes. ``price`` and
``cos_price`` still take one variance.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from volsurface import heston as heston_f
from volsurface.core.params import from_unconstrained, to_unconstrained
from volsurface.core.types import ParamSpec, Paths, PriceResult, State
from volsurface.mc.grid import estimate_terminal
from volsurface.mc.rng import generator, spawn
from volsurface.plugins import register

# Fang and Oosterlee truncation. L = 10 is their reported Heston choice.
# N = 256 prices the T = 1 reference to 1e-7. See docs/DESIGN.md section 14.
COS_L = 10.0
COS_N = 256
# Kurtosis c4 / c2^2 above this is the cumulant stencil, not a fat tail.
_C4_KURTOSIS_CAP = 80.0


def _spec() -> list[ParamSpec]:
    return [
        ParamSpec("v0", lower=1e-6, upper=2.0, default=0.04, transform="log"),
        ParamSpec("kappa", lower=1e-3, upper=20.0, default=1.5, transform="log"),
        ParamSpec("theta", lower=1e-6, upper=2.0, default=0.04, transform="log"),
        ParamSpec("xi", lower=1e-6, upper=5.0, default=0.5, transform="log"),
        ParamSpec("rho", lower=-0.999, upper=0.999, default=-0.7, transform="logit"),
    ]


@register("heston")
@dataclass(frozen=True)
class Heston:
    """Heston (1993). ``price`` is Lewis. ``cos_price`` is the cosine expansion.

    ``n_paths``, ``n_steps`` and ``seed`` are the defaults for
    :func:`euler_price`. The Fourier methods do not read them.
    """

    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float
    n_paths: int = 8_000
    n_steps: int = 64
    seed: int = 0

    def initial_state(self) -> State:
        return {"v": np.array(self.v0)}

    def feller_ratio(self) -> float:
        """``2 kappa theta / xi²``. Below 1 the variance can reach 0. Not a constraint."""
        return float(2.0 * self.kappa * self.theta / (self.xi * self.xi))

    def _variance(self, state: State | None) -> float:
        if state is None or "v" not in state:
            return float(self.v0)
        v = np.asarray(state["v"], dtype=float)
        if v.shape == () or v.size == 1:
            return float(v.reshape(()))
        raise ValueError(
            "Heston.price uses the Fourier formula, which takes one variance. "
            f"Got state['v'] with shape {v.shape}."
        )

    def price(self, F, K, T, df=1.0, kind="call", *, state: State | None = None):
        """Cash Heston price by the Lewis integral. A scalar ``state['v']`` replaces ``v0``."""
        v0 = self._variance(state)
        return heston_f.price(F, K, T, v0, self.kappa, self.theta, self.xi, self.rho, df, kind)

    def cos_price(
        self, F, K, T, df=1.0, kind="call", *, state: State | None = None,
        n_terms: int = COS_N, truncation: float = COS_L,
    ):
        """Cash price by the cosine expansion of the put, then put-call parity.

        ``n_terms`` is ``N`` and ``truncation`` is ``L`` in section 14 of
        ``docs/DESIGN.md``. The default is ``N = 256``, ``L = 10``.
        """
        v0 = self._variance(state)
        return _cos_cash(
            self, float(F), K, float(T), float(df), kind, v0,
            n_terms=int(n_terms), truncation=float(truncation),
        )

    def char_func(self, u, T: float, *, state: State | None = None) -> np.ndarray:
        """``E[exp(i u log(F_T / F(T)))]`` in the trap-free Gatheral form.

        ``F(T)`` is today's forward. ``F_T`` is the random terminal forward.

        ``u`` may be complex. The branch is the one Albrecher, Mayer, Schoutens
        and Tistaert (2007) show stays continuous: see :func:`volsurface.heston.char_func`.
        """
        v0 = self._variance(state)
        return heston_f.char_func(u, float(T), v0, self.kappa, self.theta, self.xi, self.rho)

    def cumulants(self, T: float, *, state: State | None = None) -> tuple[float, float, float]:
        """``(c1, c2, c4)`` of ``X_T = log(F_T / F(T))``.

        The variance is an autonomous square-root process, so

            E[v_t] = theta + (v0 - theta) exp(-kappa t),
            c1 = E[X_T] = -1/2 * integral_0^T E[v_t] dt
               = (1 - exp(-kappa T)) (theta - v0) / (2 kappa) - theta T / 2.

        ``c2`` and ``c4`` are derivatives of the cumulant generating function
        ``K(t) = log E[exp(t X_T)] = log phi(-i t)`` at ``t = 0``. They are
        read from a degree-8 polynomial on the stencil ``t = -4h, ..., 4h``
        with ``h = 0.01``. That stencil reproduces the closed form for ``c1``.
        A second stencil at ``h = 0.02``, and a central difference, are the
        checks on ``c2`` and ``c4`` (see the protocol test). A cosine expansion
        only needs the numbers to place a grid.
        """
        v0 = self._variance(state)
        kappa, theta = self.kappa, self.theta
        c1 = (1.0 - np.exp(-kappa * T)) * (theta - v0) / (2.0 * kappa) - 0.5 * theta * T
        c2, c4 = _higher_cumulants(self, float(T), v0)
        return (float(c1), c2, c4)

    def simulate(
        self, n_paths: int, times: np.ndarray, *,
        seed: int | np.random.SeedSequence, state0: State | None = None,
    ) -> Paths:
        """Unit-mean ``Z`` and the variance path. See the module note for the draws."""
        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or times.size < 1 or times[0] != 0.0:
            raise ValueError("times must be 1-d and start at 0")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("times must be strictly increasing")
        n_paths = int(n_paths)
        if n_paths < 1:
            raise ValueError(f"n_paths must be positive, got {n_paths}")
        v0 = _initial_variance(self, n_paths, state0)
        # One child for the whole run, shared by every strike priced off these paths.
        rng = generator(spawn(seed, 1)[0])
        Z, variance = _euler_full_truncation(
            times, v0, self.kappa, self.theta, self.xi, self.rho, rng,
        )
        return Paths(times=times, Z=Z, state={"v": variance})

    @classmethod
    def param_spec(cls) -> list[ParamSpec]:
        return _spec()

    def to_unconstrained(self) -> np.ndarray:
        values = {
            "v0": self.v0,
            "kappa": self.kappa,
            "theta": self.theta,
            "xi": self.xi,
            "rho": self.rho,
        }
        return to_unconstrained(values, _spec())

    @classmethod
    def from_unconstrained(cls, x: np.ndarray) -> Heston:
        values = from_unconstrained(np.asarray(x, dtype=float), _spec())
        return cls(**values)


def euler_price(
    model: Heston, F, K, T, df=1.0, kind="call", *,
    state: State | None = None, n_paths: int | None = None,
    n_steps: int | None = None, seed: int | np.random.SeedSequence | None = None,
) -> PriceResult:
    """Euler full-truncation price and its standard error, one maturity, one set of paths.

    ``n_paths``, ``n_steps`` and ``seed`` default to the fields on ``model``.
    The standard error is the sample standard deviation of the discounted
    payoffs divided by ``sqrt(n_paths)``. Strikes share the paths, so the
    errors are one draw. The Euler bias does not shrink as ``n_paths`` grows.
    """
    started = time.perf_counter()
    n_paths = model.n_paths if n_paths is None else int(n_paths)
    n_steps = model.n_steps if n_steps is None else int(n_steps)
    seed = model.seed if seed is None else seed
    if n_steps < 1:
        raise ValueError(f"n_steps must be at least 1, got {n_steps}")
    T = float(T)
    K_in = np.asarray(K, dtype=float)
    K_flat = np.atleast_1d(K_in)
    if T == 0.0:
        value = _intrinsic(float(F), K_flat, float(df), kind)
        return PriceResult(
            value=value.reshape(K_in.shape),
            stderr=np.zeros(K_in.shape, dtype=float),
            n_evals=0,
            wall_time=time.perf_counter() - started,
        )
    times = np.linspace(0.0, T, n_steps + 1)
    paths = model.simulate(n_paths, times, seed=seed, state0=state)
    value, stderr = estimate_terminal(
        paths.Z[:, -1], float(F), K_flat, float(df), kind,
        antithetic=False, control_variate=False,
    )
    return PriceResult(
        value=value.reshape(K_in.shape),
        stderr=stderr.reshape(K_in.shape),
        n_evals=int(paths.Z.shape[0]),
        wall_time=time.perf_counter() - started,
    )


def cos_interval(model: Heston, T: float, truncation: float = COS_L, *, state=None):
    """``(a, b, c4_used)`` for the cosine grid. ``c4_used`` may have been replaced by 0."""
    c1, c2, c4 = model.cumulants(float(T), state=state)
    c2_use = max(float(c2), 0.0)
    c4_use = float(c4)
    cap = _C4_KURTOSIS_CAP * max(c2_use, 1e-12) ** 2
    if not np.isfinite(c4_use) or c4_use < 0.0 or c4_use > cap:
        c4_use = 0.0
    half = float(truncation) * np.sqrt(c2_use + np.sqrt(c4_use))
    if half <= 0.0:
        half = 1e-8
    return float(c1) - half, float(c1) + half, c4_use


def _higher_cumulants(model: Heston, T: float, v0: float) -> tuple[float, float]:
    """``(c2, c4)`` from a polynomial fit to ``log phi(-i t)`` around 0."""
    h = 0.01
    t = np.arange(-4, 5) * h
    # phi(-i t) = E[exp(t X)]. log of that is the cumulant generating function.
    psi = np.log(model.char_func(-1j * t, T, state={"v": np.array(v0)})).real
    coeff = np.polyfit(t, psi, 8)
    poly = np.poly1d(coeff)
    # The Taylor coefficient of t^n / n! is the nth cumulant, so c_n = d^n K/dt^n at 0.
    c2 = float(poly.deriv(2)(0.0))
    c4 = float(poly.deriv(4)(0.0))
    return c2, c4


def _cos_cash(model, F, K, T, df, kind, v0, *, n_terms: int, truncation: float):
    K_in = np.asarray(K, dtype=float)
    K_flat = np.atleast_1d(K_in).astype(float)
    if T <= 0.0:
        value = _intrinsic(F, K_flat, df, kind)
        return value.reshape(K_in.shape)
    state = {"v": np.array(v0)}
    a, b, _ = cos_interval(model, T, truncation, state=state)
    omega = np.arange(n_terms) * np.pi / (b - a)
    phi = model.char_func(omega, T, state=state)
    weight = np.real(phi * np.exp(-1j * omega * a))
    weight[0] *= 0.5
    log_k = np.log(K_flat / F)
    upper = np.clip(log_k, a, b)
    chi, psi = _chi_psi(omega, a, upper)
    payoff = np.exp(log_k)[:, None] * psi - chi
    coefficients = 2.0 / (b - a) * payoff
    put_norm = coefficients @ weight
    put = df * F * put_norm
    call = put + df * (F - K_flat)
    if isinstance(kind, str):
        chosen = call if kind == "call" else put
        if kind not in ("call", "put"):
            raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
    else:
        kinds = np.asarray(kind)
        if kinds.shape != K_flat.shape or not np.all((kinds == "call") | (kinds == "put")):
            raise ValueError(f"kind must be 'call' or 'put' with shape {K_flat.shape}")
        chosen = np.where(kinds == "call", call, put)
    return np.asarray(chosen, dtype=float).reshape(K_in.shape)


def _chi_psi(omega: np.ndarray, a: float, upper: np.ndarray):
    """``int_a^upper e^y cos`` and ``int_a^upper cos``, shape ``(n_strikes, N)``.

    These are ``chi`` and ``psi`` in Fang and Oosterlee, with the lower limit
    at the left edge of the cosine interval. An empty interval (``upper == a``)
    integrates to zero, which is the out-of-the-money put whose strike sits
    below every point the density was given.
    """
    span = upper[:, None] - a
    frequency = omega[None, :]
    exp_upper = np.exp(upper)[:, None]
    exp_a = np.exp(a)
    cos_span = np.cos(frequency * span)
    sin_span = np.sin(frequency * span)
    chi = (cos_span * exp_upper - exp_a + frequency * sin_span * exp_upper) / (
        1.0 + frequency * frequency
    )
    psi = np.empty_like(chi)
    zero = np.abs(omega) < 1e-14
    psi[:, zero] = span
    live = ~zero
    psi[:, live] = sin_span[:, live] / frequency[:, live]
    return chi, psi


def _initial_variance(model: Heston, n_paths: int, state0: State | None) -> np.ndarray:
    if state0 is None or "v" not in state0:
        return np.full(n_paths, model.v0)
    v = np.asarray(state0["v"], dtype=float)
    if v.shape == () or v.size == 1:
        return np.full(n_paths, float(v.reshape(())))
    if v.shape != (n_paths,):
        raise ValueError(f"state['v'] must be a scalar or shape {(n_paths,)}, got {v.shape}")
    return v.copy()


def _euler_full_truncation(times, v0, kappa, theta, xi, rho, rng):
    """Full truncation. See the module note for the draw order and the floor."""
    n_paths = int(v0.shape[0])
    n_steps = int(times.size - 1)
    Z = np.empty((n_paths, times.size), dtype=float)
    variance = np.empty((n_paths, times.size), dtype=float)
    Z[:, 0] = 1.0
    variance[:, 0] = v0
    log_z = np.zeros(n_paths, dtype=float)
    v = v0.copy()
    corr = float(np.sqrt(max(1.0 - rho * rho, 0.0)))
    for i in range(n_steps):
        dt = float(times[i + 1] - times[i])
        shock_s = rng.standard_normal(n_paths)
        shock_v = rng.standard_normal(n_paths)
        mixed = rho * shock_s + corr * shock_v
        floor = np.maximum(v, 0.0)
        step = np.sqrt(dt)
        log_z = log_z - 0.5 * floor * dt + np.sqrt(floor) * step * shock_s
        v = v + kappa * (theta - floor) * dt + xi * np.sqrt(floor) * step * mixed
        Z[:, i + 1] = np.exp(log_z)
        variance[:, i + 1] = v
    return Z, variance


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
    if not np.all((kinds == "call") | (kinds == "put")):
        raise ValueError(f"kind must be 'call' or 'put', got {kinds}")
    return df * np.where(kinds == "call", call, put)
