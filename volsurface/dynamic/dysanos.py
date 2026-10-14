"""Minimal DYSANOS. DESIGN section 22.

The full v3 baseline is future work: a fitted decoder, a spot copula,
the paper's kappa units and expiry grid, and joint training. The paper
was not read in full. Paper statements below are marked UNVERIFIED.

``kappa`` is per year. The step passed to ``transition_matrix`` is in
years, so a one-day step is ``1/252``. Decoder expiries are business days
divided by 252.

The decoder maps a latent to a nonnegative discrete local volatility and to
increasing total variances, then to a SANOS surface. The default grid is
seven expiries, at 2, 5, 10, 20, 40, 126 and 252 business days, and 21
strikes. An 18 by 200 grid is available as ``wide=True`` and is not the
default. The knots of that wide grid are UNVERIFIED.

Compression is affine, ``u = b + B h``. ``d`` defaults to 20.

The transition is the AR(1)

    h_{t+D} = m + A_D (h_t - m) + G' z,
    A_D = exp(-kappa D),
    kappa = L L' + S + kappa_min I,

with ``kappa_min = 2e-4`` and ``S`` antisymmetric. The eigenvalues of
``A_D`` then have modulus below 1. ``G`` does not depend on ``h``. An
optional scalar OU multiplies ``G`` and is off unless requested. Both the
constant-innovation claim and the OU variant are UNVERIFIED as readings of
the paper.

The fit used here is two-stage: factors, then Gaussian maximum likelihood,
which for this linear transition is ordinary least squares. That is a
stated deviation from joint training of the decoder and the transition.
Joint training is UNVERIFIED.

The spot is drawn from the front-expiry mixture. That draw adds no drift.
The probability-integral transform of the draw under the same mixture is
the spot check. The paper's copula formulae are UNVERIFIED, so this is the
mixture the surface actually prices.

The bound written in the plan is ``(Sigma_min, Sigma_max) = (0.01^2, 3^2)``,
with the exponent flagged for a check against the PDF. The discrete-local-
volatility operator squares ``Sigma``. This code therefore bounds that
``Sigma`` in ``[0.01, 3]``, which is the square root of the written
interval. The exponent is UNVERIFIED. Kernel volatilities lie in
``[0.01, 1]`` and ``eta`` lies in ``[0.01, 0.50]``. Those three intervals
are the plan's experimental values, used as defaults, and they are
UNVERIFIED against the PDF.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.linalg import expm
from scipy.special import ndtr

from volsurface.core.types import Curves, Domain
from volsurface.fit.sanos import project_marginal
from volsurface.mc.rng import generator
from volsurface.surface.sanos import SanosSurface
from volsurface.surface.sanos_dlv import advance

# UNVERIFIED experimental bounds. See the module note for the Sigma exponent.
KAPPA_MIN = 2e-4
SIGMA_MIN = 0.01
SIGMA_MAX = 3.0
VOL_MIN = 0.01
VOL_MAX = 1.0
ETA_MIN = 0.01
ETA_MAX = 0.50
DEFAULT_D = 20
DEFAULT_STRIKES = 21
BUSINESS_DAY = 1.0 / 252.0
DEFAULT_DAYS = (2, 5, 10, 20, 40, 126, 252)
# UNVERIFIED. Eighteen knots standing in for the paper's wide grid.
WIDE_DAYS = (1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 126, 180, 252, 365)


def kappa_matrix(lower: np.ndarray, skew: np.ndarray, kappa_min: float = KAPPA_MIN) -> np.ndarray:
    """``L L' + S + kappa_min I``. ``S`` is replaced by its antisymmetric part."""
    ell = np.asarray(lower, dtype=float)
    raw = np.asarray(skew, dtype=float)
    anti = 0.5 * (raw - raw.T)
    return ell @ ell.T + anti + float(kappa_min) * np.eye(ell.shape[0])


def transition_matrix(
    lower: np.ndarray, skew: np.ndarray, step: float, kappa_min: float = KAPPA_MIN
) -> np.ndarray:
    """``A_D = exp(-kappa D)``."""
    if step <= 0.0:
        raise ValueError(f"step must be positive, got {step}")
    return expm(-kappa_matrix(lower, skew, kappa_min) * float(step))


def max_modulus(lower: np.ndarray, skew: np.ndarray, step: float) -> float:
    """Largest modulus of an eigenvalue of ``A_D``."""
    values = np.linalg.eigvals(transition_matrix(lower, skew, step))
    return float(np.max(np.abs(values)))


def simulate_factors(
    mean: np.ndarray,
    lower: np.ndarray,
    skew: np.ndarray,
    loading: np.ndarray,
    n_steps: int,
    step: float,
    seed: int,
    *,
    scale: np.ndarray | None = None,
) -> np.ndarray:
    """Gaussian AR(1). Draw order: at each date, one standard normal per row of ``loading``.

    The innovation is ``loading.T @ z`` times ``scale[t]`` when a scale path
    is supplied. ``scale`` is the optional OU multiplier. It is not fitted here.
    """
    centre = np.asarray(mean, dtype=float).reshape(-1)
    gain = np.asarray(loading, dtype=float)
    matrix = transition_matrix(lower, skew, step)
    rng = generator(seed)
    out = np.empty((int(n_steps), centre.size))
    current = centre.copy()
    widths = np.ones(int(n_steps)) if scale is None else np.asarray(scale, dtype=float)
    for t in range(int(n_steps)):
        shock = rng.standard_normal(gain.shape[0])
        current = centre + matrix @ (current - centre) + widths[t] * (gain.T @ shock)
        out[t] = current
    return out


def fit_var(factors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gaussian MLE of ``h_{t+1} = m + A (h_t - m) + e``.

    For this linear model the MLE is the least-squares coefficient and the
    residual covariance with divisor ``n - 1`` observations. This is the
    second stage of the two-stage fit. It does not constrain ``A``.
    """
    series = np.asarray(factors, dtype=float)
    if series.ndim != 2 or series.shape[0] < series.shape[1] + 2:
        raise ValueError("need more dates than factors")
    centre = series.mean(axis=0)
    lagged = series[:-1] - centre
    leading = series[1:] - centre
    coef, _, _, _ = np.linalg.lstsq(lagged, leading, rcond=None)
    residual = leading - lagged @ coef
    cov = (residual.T @ residual) / residual.shape[0]
    return centre, coef.T, cov


def bootstrap_bands(
    factors: np.ndarray, n_boot: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Percentile bands for the least-squares ``A`` and the residual covariance.

    Residuals are resampled with replacement. The return is
    ``(low_A, high_A, low_cov, high_cov)`` at the 2.5 and 97.5 percentiles.
    """
    series = np.asarray(factors, dtype=float)
    centre, slope, _ = fit_var(series)
    lagged = series[:-1] - centre
    leading = series[1:] - centre
    residual = leading - lagged @ slope.T
    rng = generator(seed)
    slopes = np.empty((n_boot, slope.shape[0], slope.shape[1]))
    covs = np.empty((n_boot, slope.shape[0], slope.shape[1]))
    n_obs = residual.shape[0]
    fitted = lagged @ slope.T
    for b in range(int(n_boot)):
        draw = rng.integers(0, n_obs, size=n_obs)
        sample = fitted + residual[draw]
        coef, _, _, _ = np.linalg.lstsq(lagged, sample, rcond=None)
        slopes[b] = coef.T
        shock = residual[draw]
        covs[b] = (shock.T @ shock) / n_obs
    return (
        np.quantile(slopes, 0.025, axis=0),
        np.quantile(slopes, 0.975, axis=0),
        np.quantile(covs, 0.025, axis=0),
        np.quantile(covs, 0.975, axis=0),
    )


class Dysanos:
    """Minimal DYSANOS decoder and grids. See the module note."""

    def __init__(
        self,
        *,
        d: int = DEFAULT_D,
        n_strikes: int = DEFAULT_STRIKES,
        wide: bool = False,
        seed: int = 0,
    ):
        if d < 1:
            raise ValueError(f"d must be positive, got {d}")
        self.d = int(d)
        self.wide = bool(wide)
        days = WIDE_DAYS if wide else DEFAULT_DAYS
        strikes = 200 if wide else int(n_strikes)
        if not wide and not 20 <= strikes <= 50:
            raise ValueError(f"n_strikes must lie in [20, 50] on the default grid, got {strikes}")
        self.expiries = np.asarray(days, dtype=float) * BUSINESS_DAY
        log_k = np.linspace(-0.3, 0.3, strikes)
        self.strikes = np.exp(log_k)
        rng = generator(seed)
        # Four controls: eta, local vol, kernel vol, tilt. The width is UNVERIFIED.
        self.b = np.zeros(4)
        self.B = 0.3 * rng.standard_normal((4, self.d))

    def decode(self, latent: np.ndarray) -> SanosSurface:
        """One SANOS surface. ``latent`` has length ``d``."""
        h = np.asarray(latent, dtype=float).reshape(-1)
        if h.size != self.d:
            raise ValueError(f"latent length {h.size} does not match d={self.d}")
        controls = self.b + self.B @ h
        eta = _unit(controls[0], ETA_MIN, ETA_MAX)
        local = _unit(controls[1], 0.05, 0.35)
        vol = _unit(controls[2], VOL_MIN, 0.4)
        if not SIGMA_MIN <= local <= SIGMA_MAX:
            raise RuntimeError("local volatility left the UNVERIFIED Sigma bound")
        variance = np.maximum(vol**2 * self.expiries, 1e-8)
        if np.any(np.diff(variance) <= 0.0):
            raise RuntimeError("kernel variance is not increasing")
        masses = [_front_masses(self.strikes, float(variance[0]))]
        sigma = np.zeros(self.strikes.size)
        sigma[1:-1] = local
        for index in range(1, self.expiries.size):
            gap = float(self.expiries[index] - self.expiries[index - 1])
            masses.append(advance(self.strikes, masses[-1], self.strikes, sigma, gap))
        return _surface(self.expiries, self.strikes, masses, variance, eta)

    def sample_front(self, surface: SanosSurface, n: int, seed: int) -> np.ndarray:
        """Draws from the front-expiry mixture. No drift is added."""
        return _sample_mixture(
            surface.strikes[0],
            surface.weights[0],
            surface.eta * float(surface.variance[0]),
            int(n),
            generator(seed),
        )

    def front_cdf(self, surface: SanosSurface, spots: np.ndarray) -> np.ndarray:
        """Distribution function of the same front mixture."""
        return _mixture_cdf(
            surface.strikes[0],
            surface.weights[0],
            surface.eta * float(surface.variance[0]),
            spots,
        )

    def sample_expiry(self, surface: SanosSurface, index: int, n: int, seed: int) -> np.ndarray:
        """Draws from one expiry's mixture. Used by the held-to-expiry book."""
        return _sample_mixture(
            surface.strikes[index],
            surface.weights[index],
            surface.eta * float(surface.variance[index]),
            int(n),
            generator(seed),
        )


def _unit(value: float, low: float, high: float) -> float:
    weight = 1.0 / (1.0 + math.exp(-float(value)))
    return low + (high - low) * weight


def _front_masses(strikes: np.ndarray, variance: float) -> np.ndarray:
    """Lognormal atoms on the strike grid, projected onto mass 1 and mean 1."""
    log_k = np.log(strikes)
    sd = math.sqrt(variance)
    z = (log_k + 0.5 * variance) / sd
    density = np.exp(-0.5 * z * z) / (sd * math.sqrt(2.0 * math.pi) * strikes)
    raw = density * np.gradient(strikes)
    masses, _ = project_marginal(np.maximum(raw, 0.0), strikes)
    return masses


def _surface(expiries, strikes, masses, variance, eta: float) -> SanosSurface:
    def flat(level: float):
        def curve(t):
            return np.full(np.shape(np.asarray(t, dtype=float)), level, dtype=float)

        return curve

    log_k = np.log(strikes)
    return SanosSurface(
        expiries,
        [strikes for _ in expiries],
        masses,
        variance,
        eta,
        Curves(flat(1.0), flat(1.0)),
        Domain(float(expiries[0]), float(expiries[-1]), float(log_k[0]), float(log_k[-1])),
        time_interp="linear",
    )


def _sample_mixture(strikes, weights, variance: float, n: int, rng) -> np.ndarray:
    index = rng.choice(strikes.size, size=n, p=np.asarray(weights, dtype=float))
    chosen = np.asarray(strikes, dtype=float)[index]
    if variance <= 1e-16:
        return chosen
    shock = rng.standard_normal(n)
    return chosen * np.exp(-0.5 * variance + math.sqrt(variance) * shock)


def _mixture_cdf(strikes, weights, variance: float, spots: np.ndarray) -> np.ndarray:
    values = np.asarray(spots, dtype=float)
    grid = np.asarray(strikes, dtype=float)
    weight = np.asarray(weights, dtype=float)
    if variance <= 1e-16:
        return np.array([(weight[grid <= value]).sum() for value in values])
    sd = math.sqrt(variance)
    centred = np.log(np.maximum(values, 1e-300))[:, None] - np.log(grid)[None, :]
    gap = (centred + 0.5 * variance) / sd
    return ndtr(gap) @ weight
