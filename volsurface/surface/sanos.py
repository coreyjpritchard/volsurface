"""SANOS normalised call: a mixture of Black calls on a discrete marginal.

``docs/DESIGN.md`` section 18. At expiry ``T_j``

``C(K) = sum_i q_j^i Call(K^i, K, eta * V_j)``.

``eta = 0`` is piecewise linear in the strike. Between expiries the call is
the blend in the paper's equation (4). ``time_interp="linear"`` uses the
straight weight in calendar time. ``"pchip"`` uses a PCHIP of at-the-money
total variance (Remark 2.13).
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.special import ndtr

from volsurface.core.types import Curves, Domain
from volsurface.surface.base import SurfaceMixin, as_axes

_SQRT_2PI = np.sqrt(2.0 * np.pi)


def black_call(forward, strike, variance) -> np.ndarray:
    """Undiscounted Black call. ``variance`` is total variance. Zero variance is the payoff."""
    forward = np.asarray(forward, dtype=float)
    strike = np.asarray(strike, dtype=float)
    variance = np.asarray(variance, dtype=float)
    intrinsic = np.maximum(forward - strike, 0.0)
    sd = np.sqrt(np.maximum(variance, 0.0))
    tiny = sd <= 1e-14
    safe = np.maximum(sd, 1e-300)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = np.log(np.maximum(forward, 1e-300) / np.maximum(strike, 1e-300)) / safe + 0.5 * safe
        d2 = d1 - safe
        smooth = forward * ndtr(d1) - strike * ndtr(d2)
    # A non-positive strike is the forward: the call pays the atom.
    return np.where(strike <= 0.0, forward, np.where(tiny, intrinsic, smooth))


def _normal_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / _SQRT_2PI


class SanosSurface(SurfaceMixin):
    """One fitted SANOS surface. See the module note.

    ``strikes[j]`` and ``weights[j]`` are the model strikes and the projected
    weights at expiry ``T[j]``. ``variance[j]`` is the ATM total variance
    ``V_j``. ``eta`` is the smoothing weight.
    """

    def __init__(
        self,
        T: np.ndarray,
        strikes: list[np.ndarray],
        weights: list[np.ndarray],
        variance: np.ndarray,
        eta: float,
        market: Curves,
        domain: Domain,
        *,
        time_interp: str = "linear",
    ):
        self.T = np.asarray(T, dtype=float)
        self.strikes = [np.asarray(s, dtype=float) for s in strikes]
        self.weights = [np.asarray(w, dtype=float) for w in weights]
        self.variance = np.asarray(variance, dtype=float)
        self.eta = float(eta)
        self.market = market
        self.domain = domain
        if time_interp not in ("linear", "pchip"):
            raise ValueError(f"time_interp must be 'linear' or 'pchip', got {time_interp!r}")
        self.time_interp = time_interp
        self._atm_w = self._atm_total_variance()
        self._pchip = None
        if time_interp == "pchip" and self.T.size >= 2:
            # Flat steps are not strictly increasing. Nudge ties so PCHIP accepts them.
            w = self._atm_w.copy()
            for j in range(1, w.size):
                if w[j] <= w[j - 1]:
                    w[j] = w[j - 1] + 1e-12
            self._pchip = PchipInterpolator(self.T, w, extrapolate=False)

    def call(self, T, k) -> np.ndarray:
        T, k = as_axes(T, k)
        pillars = self._pillar_calls(k)
        out = np.empty((T.size, k.size))
        for i, t in enumerate(T):
            out[i] = self._blend(float(t), pillars)
        return out

    def dcall_dk(self, T, k, h: float = 1e-4) -> np.ndarray:
        T, k = as_axes(T, k)
        pillars = [self._pillar_dcall_dk(j, k) for j in range(self.T.size)]
        out = np.empty((T.size, k.size))
        for i, t in enumerate(T):
            out[i] = self._blend(float(t), pillars)
        return out

    def d2call_dk2(self, T, k, h: float = 1e-3) -> np.ndarray:
        T, k = as_axes(T, k)
        pillars = [self._pillar_d2(j, k) for j in range(self.T.size)]
        out = np.empty((T.size, k.size))
        for i, t in enumerate(T):
            out[i] = self._blend(float(t), pillars)
        return out

    def dcall_dT(self, T, k, h: float = 1e-4) -> np.ndarray:
        """Time slope from ``alpha'(T)``. At a pillar the more negative side is kept."""
        T, k = as_axes(T, k)
        pillars = self._pillar_calls(k)
        out = np.empty((T.size, k.size))
        for i, t in enumerate(T):
            out[i] = self._time_slope(float(t), pillars)
        return out

    def density(self, T, k) -> np.ndarray:
        T, k = as_axes(T, k)
        if self.eta == 0.0 or np.all(self.variance <= 0.0):
            return np.zeros((T.size, k.size))
        pillars = [self._pillar_density(j, k) for j in range(self.T.size)]
        out = np.empty((T.size, k.size))
        for i, t in enumerate(T):
            out[i] = self._blend(float(t), pillars)
        return out

    def local_var(self, T, k, h: float = 1e-4) -> np.ndarray:
        """Dupire ratio from the analytic derivatives. NaN where ``eta`` is 0."""
        if self.eta == 0.0:
            T, k = as_axes(T, k)
            return np.full((T.size, k.size), np.nan)
        return super().local_var(T, k, h=h)

    def extrapolated(self, T, k) -> np.ndarray:
        return super().extrapolated(T, k)

    def _kernel_variance(self, j: int) -> float:
        return float(self.eta * self.variance[j])

    def _pillar_calls(self, k: np.ndarray) -> list[np.ndarray]:
        x = np.exp(k)
        return [
            self.weights[j]
            @ black_call(self.strikes[j][:, None], x[None, :], self._kernel_variance(j))
            for j in range(self.T.size)
        ]

    def _pillar_dcall_dk(self, j: int, k: np.ndarray) -> np.ndarray:
        """``dC/dk = x dC/dx``. The strike derivative of a Black call is ``-N(d2)``."""
        x = np.exp(k)
        v = self._kernel_variance(j)
        if v <= 1e-14:
            atoms = self.strikes[j][:, None]
            alive = atoms > x[None, :]
            slope_dx = -(self.weights[j][:, None] * alive).sum(axis=0)
            return slope_dx * x
        sd = np.sqrt(v)
        forward = self.strikes[j][:, None]
        d2 = np.log(np.maximum(forward, 1e-300) / x[None, :]) / sd - 0.5 * sd
        slope_dx = -(self.weights[j][:, None] * ndtr(d2)).sum(axis=0)
        return slope_dx * x

    def _pillar_d2(self, j: int, k: np.ndarray) -> np.ndarray:
        """Second derivative in ``k``, from the mixture density identity."""
        dk = self._pillar_dcall_dk(j, k)
        phi = self._pillar_density(j, k)
        return np.exp(k) * phi + dk

    def _pillar_density(self, j: int, k: np.ndarray) -> np.ndarray:
        v = self._kernel_variance(j)
        if v <= 1e-14:
            return np.zeros(k.shape)
        sd = np.sqrt(v)
        mean = np.log(np.maximum(self.strikes[j], 1e-300)) - 0.5 * v
        z = (k[None, :] - mean[:, None]) / sd
        return (self.weights[j][:, None] * _normal_pdf(z) / sd).sum(axis=0)

    def _atm_total_variance(self) -> np.ndarray:
        """Black total variance of the model call at ``K = 1``."""
        from volsurface import bs

        out = np.empty(self.T.size)
        for j in range(self.T.size):
            price = float(
                black_call(self.strikes[j], np.array([1.0]), self._kernel_variance(j))
                @ self.weights[j]
            )
            iv = bs.implied_vol(price, 1.0, 1.0, 1.0, 1.0, "call")
            # The inversion uses T = 1, so the returned vol squared is the total variance.
            out[j] = float(iv) ** 2 if np.isfinite(iv) else 0.0
        return out

    def _bracket(self, t: float) -> tuple[int, float, float]:
        """Left index, blend weight on the right pillar, and ``d alpha / d t``."""
        times = self.T
        if t <= times[0] or times.size == 1:
            return 0, 0.0, 0.0
        if t >= times[-1]:
            return int(times.size - 1), 0.0, 0.0
        j = int(np.searchsorted(times, t, side="right") - 1)
        dt = float(times[j + 1] - times[j])
        if self.time_interp == "linear" or self._pchip is None:
            alpha = (t - float(times[j])) / dt
            return j, float(alpha), 1.0 / dt
        w_left = float(self._atm_w[j])
        w_right = float(self._atm_w[j + 1])
        gap = w_right - w_left
        w_t = float(self._pchip(t))
        slope = float(self._pchip(t, nu=1))
        if abs(gap) < 1e-14:
            alpha = (t - float(times[j])) / dt
            return j, float(alpha), 1.0 / dt
        # Remark 2.13: alpha from the ATM call as a function of total variance.
        c_t = float(black_call(1.0, 1.0, w_t))
        c_left = float(black_call(1.0, 1.0, w_left))
        c_right = float(black_call(1.0, 1.0, w_right))
        alpha = (c_t - c_left) / (c_right - c_left)
        # dCall/dw at the money is the total-variance vega n(d1)/2.
        vega_w = float(_normal_pdf(np.array([0.5 * np.sqrt(max(w_t, 0.0))]))[0])
        if w_t > 1e-14:
            vega_w = vega_w / (2.0 * np.sqrt(w_t))
        alpha_prime = vega_w * slope / (c_right - c_left)
        return j, float(np.clip(alpha, 0.0, 1.0)), float(alpha_prime)

    def _blend(self, t: float, pillars: list[np.ndarray]) -> np.ndarray:
        j, alpha, _ = self._bracket(t)
        if alpha == 0.0 or j == self.T.size - 1:
            return pillars[j]
        return (1.0 - alpha) * pillars[j] + alpha * pillars[j + 1]

    def _time_slope(self, t: float, pillars: list[np.ndarray]) -> np.ndarray:
        times = self.T
        if times.size == 1 or t < float(times[0]) or t > float(times[-1]):
            return np.zeros(pillars[0].shape)
        on_pillar = np.where(np.isclose(times, t, rtol=0.0, atol=1e-14))[0]
        if on_pillar.size and 0 < int(on_pillar[0]) < times.size - 1:
            j = int(on_pillar[0])
            span_left = float(times[j] - times[j - 1])
            span_right = float(times[j + 1] - times[j])
            _, _, a_left = self._bracket(float(times[j]) - 1e-8 * span_left)
            _, _, a_right = self._bracket(float(times[j]) + 1e-8 * span_right)
            slope_left = a_left * (pillars[j] - pillars[j - 1])
            slope_right = a_right * (pillars[j + 1] - pillars[j])
            return np.minimum(slope_left, slope_right)
        j, _, alpha_prime = self._bracket(t)
        if j >= times.size - 1:
            return np.zeros(pillars[0].shape)
        return alpha_prime * (pillars[j + 1] - pillars[j])
