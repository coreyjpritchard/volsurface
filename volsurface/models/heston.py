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

The cosine interval is ``c1 ± L sqrt(c2 + sqrt(c4))`` with ``L = 10``
when the fourth cumulant is usable. Those cumulants are
:meth:`Heston.cumulants`. The stencil step is ``f / sqrt(c2)``, with
``f`` in ``{0.001, 0.002, 0.005}``, halved while ``phi(-i t)`` is not
finite. When that set does not agree, a finer set ``{0.0005, 0.001, 0.002}``
is tried. ``L`` then doubles, and ``N`` is chosen again, until two
successive prices agree to ``1e-7`` or the next width would pass 160 or
65536 terms. The price kept is the narrower of the two that agreed. If
they never agree, the widest width whose last doubling of ``N`` moved the
price by at most ``2.5e-4`` is kept, a ``RuntimeWarning`` is emitted, and
the returned array has ``converged`` false. Past that point the term cap,
not the tail, is the error. Long-dated, very-high-kurtosis slices can hit
that cap; use Lewis there. A large positive ``c4`` is kept inside the
formula.
``N`` is at least 256 and doubles until successive prices agree or ``N``
reaches 65536. Passing ``n_terms`` skips both adaptations and uses the
given count at the given ``L``.

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

import math
import time
import warnings
from dataclasses import dataclass

import numpy as np

from volsurface import heston as heston_f
from volsurface.core.params import from_unconstrained, to_unconstrained
from volsurface.core.types import ParamSpec, Paths, PriceResult, State
from volsurface.mc.grid import estimate_terminal
from volsurface.mc.rng import generator, spawn
from volsurface.plugins import register

# Fang and Oosterlee truncation. L = 10 is their reported Heston choice.
# COS_N is the smallest term count. Wide intervals use more. See section 14.
COS_L = 10.0
COS_N = 256
_COS_N_MAX = 65_536
# Successive cosine prices must agree by this much before N stops growing.
_COS_AGREE = 1e-7
# A wider L is accepted only when the last doubling of N moved the price by
# at most this much. Beyond it, the 65536-term cap dominates the tail.
_COS_N_STEP = 2.5e-4
# Neighbouring stencil steps. c4 is used only when both agree.
_C2_AGREE = 1e-3
_C4_AGREE = 1e-3
# Step = factor / sqrt(c2). 0.01 and wider leave the local expansion once
# kurtosis is large, while phi(-i t) is still finite. 0.0005 is round-off
# on ordinary maturities, so it is used only when the wider three disagree.
_STEP_FACTORS = (0.001, 0.002, 0.005)
_STEP_FACTORS_FINE = (0.0005, 0.001, 0.002)
# When c4 is unusable, L doubles until prices agree or this cap is hit.
_COS_L_MAX = 160.0


class CosPrice(np.ndarray):
    """Cash cosine price.

    ``converged`` is true when two successive widths agreed, when ``n_terms``
    was passed, or when ``T <= 0``. It is false when the loop stops because
    the next width would pass ``L = 160`` or 65536 terms.
    """

    def __new__(cls, values, *, converged: bool):
        obj = np.array(values, dtype=float, copy=True).view(cls)
        obj.converged = bool(converged)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.converged = bool(getattr(obj, "converged", True))


def _as_cos_price(values, shape, *, converged: bool) -> CosPrice:
    return CosPrice(np.asarray(values, dtype=float).reshape(shape), converged=converged)


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
        self,
        F,
        K,
        T,
        df=1.0,
        kind="call",
        *,
        state: State | None = None,
        n_terms: int | None = None,
        truncation: float = COS_L,
    ):
        """Cash price by the cosine expansion of the put, then put-call parity.

        ``n_terms`` is ``N`` and ``truncation`` is ``L`` in section 14 of
        ``docs/DESIGN.md``. ``n_terms=None`` chooses ``N`` from the interval
        and raises ``L`` until successive prices agree. ``L`` defaults to 10.
        The result is a :class:`CosPrice`. ``converged`` is false, and a
        ``RuntimeWarning`` is emitted, when the loop stops at the term cap
        or the ``L`` cap without two widths agreeing.
        """
        v0 = self._variance(state)
        return _cos_cash(
            self,
            float(F),
            K,
            float(T),
            float(df),
            kind,
            v0,
            n_terms=None if n_terms is None else int(n_terms),
            truncation=float(truncation),
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
        ``K(t) = log E[exp(t X_T)] = log phi(-i t)`` at ``t = 0``. The step
        is ``f / sqrt(c2)``. A first pass uses ``sqrt(-2 c1)`` as that scale,
        and a second pass uses the ``c2`` the first pass kept. A degree-8 fit
        is kept only when two neighbouring steps agree, and the step is halved
        while ``phi(-i t)`` is not finite. ``c4`` is ``nan`` when those steps
        disagree. The cosine price then raises ``L`` until the price settles,
        instead of trusting round-off. The protocol test checks the kept
        values against a second stencil and a central difference.
        """
        v0 = self._variance(state)
        kappa, theta = self.kappa, self.theta
        c1 = (1.0 - np.exp(-kappa * T)) * (theta - v0) / (2.0 * kappa) - 0.5 * theta * T
        c2, c4 = _higher_cumulants(self, float(T), v0, float(c1))
        return (float(c1), c2, c4)

    def simulate(
        self,
        n_paths: int,
        times: np.ndarray,
        *,
        seed: int | np.random.SeedSequence,
        state0: State | None = None,
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
            times,
            v0,
            self.kappa,
            self.theta,
            self.xi,
            self.rho,
            rng,
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
    model: Heston,
    F,
    K,
    T,
    df=1.0,
    kind="call",
    *,
    state: State | None = None,
    n_paths: int | None = None,
    n_steps: int | None = None,
    seed: int | np.random.SeedSequence | None = None,
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


def cos_interval(model: Heston, T: float, truncation: float = COS_L, *, state=None):
    """``(a, b, c4_used)`` for one truncation ``L``.

    ``c4_used`` is 0 when the fourth cumulant was unusable. The half-width
    is then the Gaussian ``L sqrt(c2)``. :meth:`Heston.cos_price` still
    doubles ``L`` until successive prices agree, with or without ``c4``.
    A finite positive ``c4`` is not capped.
    """
    c1, c2, c4 = model.cumulants(float(T), state=state)
    return _interval_from_cumulants(c1, c2, c4, truncation)


def _interval_from_cumulants(c1: float, c2: float, c4: float, truncation: float):
    c2_use = _variance_scale(c1, c2)
    c4_use = float(c4)
    if not np.isfinite(c4_use) or c4_use < 0.0:
        half = float(truncation) * math.sqrt(c2_use)
        c4_use = 0.0
    else:
        half = float(truncation) * math.sqrt(c2_use + math.sqrt(c4_use))
    if half <= 0.0:
        half = 1e-8
    return float(c1) - half, float(c1) + half, c4_use


def _variance_scale(c1: float, c2: float) -> float:
    """Positive stand-in for ``c2`` when the stencil did not return one."""
    if np.isfinite(c2) and c2 > 0.0:
        return float(c2)
    proxy = -2.0 * float(c1)
    if proxy > 0.0:
        return proxy
    return 1e-8


def _higher_cumulants(model: Heston, T: float, v0: float, c1: float) -> tuple[float, float]:
    """``(c2, c4)`` from degree-8 fits with step ``f / sqrt(c2)``.

    ``c4`` is ``nan`` when neighbouring steps do not agree. ``c2`` is still
    returned when those steps agree on the second derivative alone.
    """
    scale = math.sqrt(max(-2.0 * float(c1), 1e-18))
    c2, c4 = _stencil_passes(model, T, v0, scale, _STEP_FACTORS)
    if np.isfinite(c2) and c2 > 0.0 and not (np.isfinite(c4) and c4 > 0.0):
        fine_c2, fine_c4 = _stencil_at_scale(model, T, v0, math.sqrt(c2), _STEP_FACTORS_FINE)
        if np.isfinite(fine_c4) and fine_c4 > 0.0:
            c4 = fine_c4
            if np.isfinite(fine_c2) and fine_c2 > 0.0:
                c2 = fine_c2
    return c2, c4


def _stencil_passes(model, T, v0, scale, factors) -> tuple[float, float]:
    c2, c4 = _stencil_at_scale(model, T, v0, scale, factors)
    if np.isfinite(c2) and c2 > 0.0:
        refined_c2, refined_c4 = _stencil_at_scale(model, T, v0, math.sqrt(c2), factors)
        if np.isfinite(refined_c2) and refined_c2 > 0.0:
            return refined_c2, refined_c4
    return c2, c4


def _stencil_at_scale(
    model: Heston, T: float, v0: float, scale: float, factors
) -> tuple[float, float]:
    rows: list[tuple[float, float]] = []
    seen: list[float] = []
    for factor in factors:
        step = factor / scale
        fitted = None
        for _ in range(24):
            fitted = _cgf_poly(model, T, v0, step)
            if fitted is not None and fitted[0] > 0.0:
                break
            fitted = None
            step *= 0.5
        if fitted is None:
            continue
        # Halving can land on a step already used. That pair is not a check.
        if any(abs(step - prev) <= 1e-12 * max(step, prev, 1.0) for prev in seen):
            continue
        seen.append(step)
        rows.append(fitted)
    return _agreeing_cumulants(rows)


def _agreeing_cumulants(rows: list[tuple[float, float]]) -> tuple[float, float]:
    """Keep the finest pair that agrees. A coarser pair must not replace it."""
    c2_kept = float("nan")
    c4_kept = float("nan")
    for left, right in zip(rows, rows[1:], strict=False):
        c2_left, c4_left = left
        c2_right, c4_right = right
        if c2_left <= 0.0 or c2_right <= 0.0:
            continue
        if abs(c2_left - c2_right) > _C2_AGREE * max(c2_left, c2_right):
            continue
        if not np.isfinite(c2_kept):
            c2_kept = c2_left
        if np.isfinite(c4_kept) or c4_left <= 0.0 or c4_right <= 0.0:
            continue
        if abs(c4_left - c4_right) <= _C4_AGREE * max(c4_left, c4_right):
            c4_kept = c4_left
    if not np.isfinite(c2_kept):
        for c2_row, _c4_row in rows:
            if np.isfinite(c2_row) and c2_row > 0.0:
                c2_kept = c2_row
                break
    return c2_kept, c4_kept


def _cgf_poly(model: Heston, T: float, v0: float, step: float):
    """``(c2, c4)`` at one step, or None if ``phi(-i t)`` is not usable."""
    t = np.arange(-4, 5, dtype=float) * step
    phi = np.asarray(model.char_func(-1j * t, T, state={"v": np.array(v0)}))
    if phi.shape != t.shape or not np.all(np.isfinite(phi)) or np.any(np.abs(phi) == 0.0):
        return None
    psi = np.log(phi).real
    if not np.all(np.isfinite(psi)):
        return None
    coeff = np.polyfit(t, psi, 8)
    if not np.all(np.isfinite(coeff)):
        return None
    poly = np.poly1d(coeff)
    c2 = float(poly.deriv(2)(0.0))
    c4 = float(poly.deriv(4)(0.0))
    if not np.isfinite(c2) or not np.isfinite(c4):
        return None
    return c2, c4


def _n_from_width(left: float, right: float, c2: float) -> int:
    """Smallest power of two at least 256 that resolves the cosine interval."""
    width = max(right - left, 1e-8)
    scale = max(math.sqrt(max(c2, 0.0)), 1e-8)
    raw = 32.0 * width / scale
    n_terms = COS_N
    while n_terms < raw and n_terms < _COS_N_MAX:
        n_terms *= 2
    return n_terms


def _cos_cash(model, F, K, T, df, kind, v0, *, n_terms: int | None, truncation: float):
    K_in = np.asarray(K, dtype=float)
    K_flat = np.atleast_1d(K_in).astype(float)
    if T <= 0.0:
        value = _intrinsic(F, K_flat, df, kind)
        return _as_cos_price(value, K_in.shape, converged=True)
    state = {"v": np.array(v0)}
    c1, c2, c4 = model.cumulants(float(T), state=state)
    scale = _variance_scale(c1, c2)
    if n_terms is not None:
        left, right, _used = _interval_from_cumulants(c1, c2, c4, truncation)
        chosen, _step = _adapt_n(
            model, F, K_flat, T, df, kind, state, left, right, scale, n_terms=n_terms
        )
        return _as_cos_price(chosen, K_in.shape, converged=True)
    # Double L until the price stops moving. Keep the narrower of two widths
    # that agree. If the next width is unresolved, keep the widest one whose
    # last doubling of N still moved the price by at most _COS_N_STEP, and
    # say so: the term cap, not the tail, is the remaining error.
    level = float(truncation)
    previous = None
    kept = None
    while True:
        left, right, _used = _interval_from_cumulants(c1, c2, c4, level)
        chosen, last_step = _adapt_n(
            model, F, K_flat, T, df, kind, state, left, right, scale, n_terms=None
        )
        if previous is not None and float(np.max(np.abs(chosen - previous))) <= _COS_AGREE:
            return _as_cos_price(previous, K_in.shape, converged=True)
        if kept is None or last_step <= _COS_N_STEP:
            kept = chosen
        next_level = level * 2.0
        if next_level > _COS_L_MAX or _width_unresolved(c1, c2, c4, next_level, scale):
            warnings.warn(
                "COS hit the term cap or the L cap before two widths agreed. "
                "Use Lewis for this long-dated, high-kurtosis slice.",
                RuntimeWarning,
                stacklevel=3,
            )
            return _as_cos_price(kept, K_in.shape, converged=False)
        previous = chosen
        level = next_level


def _width_unresolved(c1: float, c2: float, c4: float, truncation: float, scale: float) -> bool:
    left, right, _used = _interval_from_cumulants(c1, c2, c4, truncation)
    return _n_from_width(left, right, scale) >= _COS_N_MAX


def _adapt_n(model, F, K_flat, T, df, kind, state, left, right, scale, *, n_terms: int | None):
    """Price, and how far the last doubling of ``N`` moved it."""
    if n_terms is not None:
        price = _cos_at(model, F, K_flat, T, df, kind, state, left, right, int(n_terms))
        return price, 0.0
    count = _n_from_width(left, right, scale)
    previous = None
    chosen = None
    # Inf until a second N is priced. A single N on the cap is not converged.
    last_step = math.inf
    while True:
        chosen = _cos_at(model, F, K_flat, T, df, kind, state, left, right, count)
        if previous is not None:
            last_step = float(np.max(np.abs(chosen - previous)))
            if last_step <= _COS_AGREE:
                break
        if count >= _COS_N_MAX:
            break
        previous = chosen
        count *= 2
    return chosen, last_step


def _cos_at(model, F, K_flat, T, df, kind, state, left, right, n_terms: int):
    omega = np.arange(n_terms) * np.pi / (right - left)
    phi = model.char_func(omega, T, state=state)
    weight = np.real(phi * np.exp(-1j * omega * left))
    weight[0] *= 0.5
    log_k = np.log(K_flat / F)
    upper = np.clip(log_k, left, right)
    chi, psi = _chi_psi(omega, left, upper)
    payoff = np.exp(log_k)[:, None] * psi - chi
    coefficients = 2.0 / (right - left) * payoff
    put_norm = coefficients @ weight
    put = df * F * put_norm
    call = put + df * (F - K_flat)
    if isinstance(kind, str):
        if kind not in ("call", "put"):
            raise ValueError(f"kind must be 'call' or 'put', got {kind!r}")
        return call if kind == "call" else put
    kinds = np.asarray(kind)
    if kinds.shape != K_flat.shape or not np.all((kinds == "call") | (kinds == "put")):
        raise ValueError(f"kind must be 'call' or 'put' with shape {K_flat.shape}")
    return np.where(kinds == "call", call, put)


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
