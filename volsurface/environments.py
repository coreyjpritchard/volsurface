"""Volatility environments for the dashboard: one surface, its Greeks, and its arbitrage.

An environment is a pricing model or a fitted surface seen through one lens: the Black
implied volatility ``sigma(T, k)`` on a grid of expiries ``T`` and log-moneyness
``k = log(K / F)``. Everything here reuses the library's own models and surfaces:

- ``flat``: :class:`~volsurface.models.BlackScholes`.
- ``heston``: :class:`~volsurface.models.Heston`, Lewis (2001) prices.
- ``sabr``: :func:`~volsurface.models.sabr.hagan_implied_vol`, one SABR per expiry.
- ``svi``: raw SVI slices (:class:`~volsurface.surface.svi.SVISurface`) whose parameters
  follow a power law in ``T``. See :func:`svi_environment`.
- ``rbergomi``: the ``rbergomi`` plug-in from the rough-volatility package, Monte Carlo.
- ``spx_svi`` and ``spx_sanos``: the 14 October 2026 SPXW fits, rebuilt from
  ``data/presets/spx_fits.npz`` (written by ``scripts/build_dashboard_presets.py``).

The grid is rectangular in standardised moneyness ``z = k / (sigma_ref sqrt(T))``, so each
expiry is sampled over the same number of standard deviations and the short end is not
all wing. ``k`` is therefore a ``(n_T, n_z)`` array.

Greeks are Black Greeks at each node's own implied vol, per 100 of forward, with zero
rates (``F = 100``, ``df = 1``). That is the sticky-strike convention a trader's risk
system shows. :func:`smile_delta` adds the part of delta that comes from the smile moving
with the forward. Arbitrage is :func:`volsurface.arbitrage.check_surface`, run on each
expiry's strikes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from volsurface import bs
from volsurface.arbitrage import check_surface
from volsurface.core.types import ArbTol, Curves, Domain
from volsurface.models import BlackScholes, Heston
from volsurface.models.sabr import SABR, hagan_implied_vol
from volsurface.surface.model import ModelSurface
from volsurface.surface.svi import RawSVI, SVISurface

NOTIONAL_F = 100.0
SIGMA_REF = 0.2
DEFAULT_T = np.array([7, 14, 30, 61, 91, 182, 273, 365, 547, 730]) / 365.0
DEFAULT_Z = np.linspace(-3.0, 2.5, 45)
GREEKS = ("delta", "gamma", "vega", "vanna", "volga", "theta")
PRESET_PATH = Path(__file__).resolve().parents[1] / "data" / "presets" / "spx_fits.npz"


def _unit(T):
    return np.ones_like(np.asarray(T, dtype=float))


UNIT_CURVES = Curves(F=_unit, df=_unit)


@dataclass(frozen=True)
class Grid:
    """Expiries ``T`` (n_T,) and log-moneyness ``k`` (n_T, n_z) from standardised ``z``."""

    T: np.ndarray
    z: np.ndarray
    sigma_ref: float = SIGMA_REF

    @property
    def k(self) -> np.ndarray:
        return self.z[None, :] * self.sigma_ref * np.sqrt(self.T)[:, None]

    @property
    def strike(self) -> np.ndarray:
        """Strikes per 100 of forward."""
        return NOTIONAL_F * np.exp(self.k)


def default_grid(T=None, z=None) -> Grid:
    return Grid(
        T=np.asarray(DEFAULT_T if T is None else T, dtype=float),
        z=np.asarray(DEFAULT_Z if z is None else z, dtype=float),
    )


@dataclass
class Environment:
    """A named surface. ``surface`` has the :class:`SurfaceMixin` interface.

    ``iv_override`` computes implied vol directly when the model has a formula for it
    (flat, SABR, SVI), which is faster and exact. ``dsigma_dF`` is set only when the
    smile does not simply slide with the forward (SABR with ``beta < 1``).
    """

    name: str
    surface: object
    params: dict = field(default_factory=dict)
    iv_override: object | None = None
    dsigma_dF: object | None = None
    monte_carlo: bool = False
    note: str = ""

    def iv(self, grid: Grid) -> np.ndarray:
        """Implied vol on the grid, shape ``(n_T, n_z)``. NaN where it does not exist."""
        k = grid.k
        out = np.empty_like(k)
        for i, t in enumerate(grid.T):
            if self.iv_override is not None:
                out[i] = self.iv_override(float(t), k[i])
            else:
                out[i] = np.asarray(self.surface.iv([float(t)], k[i]), dtype=float)[0]
        return out


# Environments -------------------------------------------------------------------------


def _domain(t_max: float = 5.0) -> Domain:
    return Domain(t_min=0.0, t_max=t_max, k_min=-3.0, k_max=3.0)


def flat_environment(sigma: float) -> Environment:
    model = BlackScholes(float(sigma))
    return Environment(
        name="Black-Scholes, flat vol",
        surface=ModelSurface(model, UNIT_CURVES, _domain()),
        params={"sigma": sigma},
        iv_override=lambda t, k: np.full(np.shape(k), float(sigma)),
    )


def heston_environment(v0, kappa, theta, xi, rho) -> Environment:
    model = Heston(float(v0), float(kappa), float(theta), float(xi), float(rho))
    return Environment(
        name="Heston",
        surface=ModelSurface(model, UNIT_CURVES, _domain()),
        params={"v0": v0, "kappa": kappa, "theta": theta, "xi": xi, "rho": rho},
        note=f"Feller ratio 2 kappa theta / xi^2 = {model.feller_ratio():.2f}",
    )


def sabr_environment(alpha, beta, rho, nu) -> Environment:
    """SABR with the forward normalised to 1, so ``alpha`` is roughly the ATM vol.

    SABR is a model for one expiry. Using one parameter set at every expiry is the
    simplest term structure and it is not a calibrated surface.
    """
    model = SABR(float(alpha), float(beta), float(rho), float(nu))

    def iv(t, k):
        return hagan_implied_vol(1.0, np.exp(k), t, alpha, beta, rho, nu)

    def dsigma_dF(t, k, h=1e-4):
        # Strike held fixed, forward bumped. Per unit of a forward normalised to 1.
        strike = np.exp(k)
        up = hagan_implied_vol(1.0 + h, strike, t, alpha, beta, rho, nu)
        down = hagan_implied_vol(1.0 - h, strike, t, alpha, beta, rho, nu)
        return (up - down) / (2.0 * h)

    return Environment(
        name="SABR (Hagan 2002), one parameter set for every expiry",
        surface=ModelSurface(model, UNIT_CURVES, _domain()),
        params={"alpha": alpha, "beta": beta, "rho": rho, "nu": nu},
        iv_override=iv,
        dsigma_dF=dsigma_dF,
    )


SVI_T_FLOOR = 0.25  # years; the SVI toy surface is flat in T below three months


def svi_slice(T: float, atm_vol, term_slope, b, rho, m, sigma, skew_decay) -> RawSVI:
    """One raw SVI slice in total variance at expiry ``T``.

    The parameters describe annualised variance ``v(k) = w(k) / T`` at ``T = 1``. Two
    exponents move them with ``T``: the at-the-money vol is ``atm_vol * T^term_slope``
    and the wing slope ``b`` scales like ``T^-skew_decay``, both with ``T`` floored at
    ``SVI_T_FLOOR`` so a negative exponent cannot blow up the short end. ``a`` is whatever
    puts the at-the-money variance where ``atm_vol`` says. The slice is raw SVI exactly; only the
    way its parameters vary with ``T`` is this module's choice.
    """
    t_eff = max(float(T), SVI_T_FLOOR)
    b_t = float(b) * t_eff ** (-float(skew_decay))
    atm_var = (float(atm_vol) * t_eff ** float(term_slope)) ** 2
    a_t = atm_var - b_t * (rho * (-m) + np.sqrt(m * m + sigma * sigma))
    return RawSVI(a=a_t * T, b=b_t * T, rho=float(rho), m=float(m), sigma=float(sigma))


def svi_environment(grid_T, atm_vol, term_slope, b, rho, m, sigma, skew_decay) -> Environment:
    """Raw SVI at every grid expiry, total variance linear in ``T`` between them."""
    pillars = [
        (float(t), svi_slice(float(t), atm_vol, term_slope, b, rho, m, sigma, skew_decay))
        for t in np.asarray(grid_T, dtype=float)
    ]
    surface = SVISurface(pillars, UNIT_CURVES, _domain())
    lowest = min(s.a + s.b * s.sigma * np.sqrt(1.0 - s.rho**2) for _, s in pillars)

    def iv(t, k):
        w = surface.total_variance([t], k)[0]
        with np.errstate(invalid="ignore"):
            return np.where(w > 0.0, np.sqrt(w / t), np.nan)

    note = ""
    if lowest < 0.0:
        note = "Total variance goes negative on some slice (a + b sigma sqrt(1 - rho^2) < 0)."
    return Environment(
        name="SVI (raw slices, power-law term structure)",
        surface=surface,
        params={
            "atm_vol": atm_vol,
            "term_slope": term_slope,
            "b": b,
            "rho": rho,
            "m": m,
            "sigma": sigma,
            "skew_decay": skew_decay,
        },
        iv_override=iv,
        note=note,
    )


def rbergomi_environment(H, eta, rho, sigma0, n_paths=4000, n_steps=64, seed=0) -> Environment:
    """Rough Bergomi from the ``rbergomi`` plug-in, flat forward variance ``sigma0^2``.

    Raises ``KeyError`` when the rough-volatility package is not installed.
    """
    from volsurface.plugins import create

    model = create(
        "rbergomi",
        H=float(H),
        nu=float(eta),
        rho=float(rho),
        xi0=float(sigma0) ** 2,
        n_paths=int(n_paths),
        n_steps=int(n_steps),
        seed=int(seed),
    )
    return Environment(
        name="Rough Bergomi (Monte Carlo)",
        surface=ModelSurface(model, UNIT_CURVES, _domain()),
        params={"H": H, "eta": eta, "rho": rho, "sigma0": sigma0},
        monte_carlo=True,
        note=f"Monte Carlo, {n_paths:,} paths, {n_steps} steps, seed {seed}; no error bars.",
    )


# SPX presets --------------------------------------------------------------------------


def load_spx_presets(path: str | Path = PRESET_PATH) -> dict[str, Environment]:
    """The 14 October 2026 SPXW SVI and SANOS fits, rebuilt from saved parameters.

    Moneyness is normalised to each expiry's forward and prices to ``df F``, so the
    curves are unit curves; the saved implied vols already carry the market's forwards.
    """
    from volsurface.surface.sanos import SanosSurface

    data = np.load(path, allow_pickle=False)
    svi_T = data["svi_T"]
    svi_params = data["svi_params"]  # a, b, rho, m, sigma
    pillars = [(float(t), RawSVI(*map(float, p))) for t, p in zip(svi_T, svi_params, strict=True)]
    k_lo, k_hi = (float(x) for x in data["svi_k_span"])
    svi = SVISurface(pillars, UNIT_CURVES, Domain(svi_T.min(), svi_T.max(), k_lo, k_hi))

    n = int(data["sanos_n"])
    strikes = [data[f"sanos_strikes_{j}"] for j in range(n)]
    weights = [data[f"sanos_weights_{j}"] for j in range(n)]
    s_dom = data["sanos_domain"]
    sanos = SanosSurface(
        data["sanos_T"],
        strikes,
        weights,
        data["sanos_variance"],
        float(data["sanos_eta"]),
        UNIT_CURVES,
        Domain(*map(float, s_dom)),
        time_interp=str(data["sanos_time_interp"]),
    )
    market = {
        "T": data["market_T"],
        "k": data["market_k"],
        "iv": data["market_iv"],
    }
    out = {}
    for key, surface, name in (
        ("spx_svi", svi, "SPX 14 Oct 2026: raw SVI per expiry"),
        ("spx_sanos", sanos, f"SPX 14 Oct 2026: SANOS, eta = {float(data['sanos_eta']):g}"),
    ):
        env = Environment(name=name, surface=surface, note="One snapshot, SPXW mids.")
        env.params = {"market": market, "T_range": (float(svi_T.min()), float(svi_T.max()))}
        out[key] = env
    return out


def spx_grid(env: Environment, n_T: int = 10, z=None) -> Grid:
    """Expiries from the first to the last quoted one, spaced evenly in ``sqrt(T)``."""
    lo, hi = env.params["T_range"]
    lo = max(lo, 3.0 / 365.0)
    T = np.linspace(np.sqrt(lo), np.sqrt(hi), n_T) ** 2
    return default_grid(T=T, z=z)


# Greeks, smile delta, arbitrage -------------------------------------------------------


def greeks_grid(grid: Grid, iv: np.ndarray) -> dict[str, np.ndarray]:
    """Black Greeks at each node's own implied vol, per 100 of forward, zero rates.

    Units, chosen to read like a risk report:

    - ``delta``: call delta, 0 to 1 (put delta is this minus 1).
    - ``gamma``: change in delta per 1 point move of a forward at 100.
    - ``vega``: price change per 1 vol point.
    - ``vanna``: change in delta per 1 vol point.
    - ``volga``: change in vega (per vol point) per 1 vol point.
    - ``theta``: price change per calendar day, forward and vol held fixed.
    """
    F = NOTIONAL_F
    K = grid.strike
    T = grid.T[:, None] * np.ones_like(K)
    sigma = iv
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "delta": bs.delta(F, K, T, sigma, 1.0, "call"),
            "gamma": bs.gamma(F, K, T, sigma),
            "vega": bs.vega(F, K, T, sigma) / 100.0,
            "vanna": bs.vanna(F, K, T, sigma) / 100.0,
            "volga": bs.volga(F, K, T, sigma) / 100.0**2,
            "theta": bs.theta(F, K, T, sigma) / 365.0,
        }


def smile_delta(env: Environment, grid: Grid, iv: np.ndarray, h: float = 1e-3) -> np.ndarray:
    """Call delta when the smile moves with the forward, strike held fixed.

    ``delta_smile = delta_BS + vega * d sigma / d F``. For a smile that slides with the
    forward (``sigma`` a function of ``K / F`` only, which is exact for flat vol, Heston
    and rough Bergomi with a flat forward variance), ``d sigma / d F = -(1/F) d sigma / dk``.
    SABR supplies its own derivative. For a fitted surface this is the sticky-moneyness
    assumption, not a model statement.
    """
    F = NOTIONAL_F
    K = grid.strike
    T = grid.T[:, None] * np.ones_like(K)
    with np.errstate(invalid="ignore", divide="ignore"):
        base = bs.delta(F, K, T, iv, 1.0, "call")
        vega = bs.vega(F, K, T, iv)
    if env.dsigma_dF is not None:
        slope = np.vstack([env.dsigma_dF(float(t), grid.k[i]) / F for i, t in enumerate(grid.T)])
    else:
        up = env.iv(Grid(grid.T, grid.z + h / (grid.sigma_ref * 1.0), grid.sigma_ref))
        down = env.iv(Grid(grid.T, grid.z - h / (grid.sigma_ref * 1.0), grid.sigma_ref))
        dk = 2.0 * h * np.sqrt(grid.T)[:, None]
        slope = -(up - down) / dk / F
    return base + vega * slope


@dataclass(frozen=True)
class ArbitrageMap:
    """Boolean masks ``(n_T, n_z)`` per check, and counts."""

    masks: dict[str, np.ndarray]
    counts: dict[str, int]
    n_checked: int
    note: str

    @property
    def ok(self) -> bool:
        return all(v == 0 for v in self.counts.values())


def arbitrage_map(env: Environment, grid: Grid, tol: ArbTol | None = None) -> ArbitrageMap:
    """:func:`check_surface` on each expiry's strikes, stacked into grid-shaped masks."""
    tol = ArbTol(price=1e-9) if tol is None else tol
    names = ("bounds", "call_spread", "butterfly", "calendar")
    masks = {name: np.zeros(grid.k.shape, dtype=bool) for name in names}
    note = ""
    for i, t in enumerate(grid.T):
        with np.errstate(all="ignore"):
            report = check_surface(env.surface, [float(t)], grid.k[i], tol=tol)
        note = report.note
        for name in names:
            if name in report.masks:
                masks[name][i] = np.asarray(report.masks[name], dtype=bool).reshape(-1)
    counts = {name: int(mask.sum()) for name, mask in masks.items()}
    return ArbitrageMap(masks=masks, counts=counts, n_checked=int(grid.k.size), note=note)


def atm_skew(env: Environment, T, h: float = 1e-3) -> np.ndarray:
    """``d sigma / dk`` at the money, central difference, one value per expiry."""
    T = np.atleast_1d(np.asarray(T, dtype=float))
    k = np.array([-h, h])
    out = np.empty(T.size)
    for i, t in enumerate(T):
        if env.iv_override is not None:
            v = env.iv_override(float(t), k)
        else:
            v = np.asarray(env.surface.iv([float(t)], k), dtype=float)[0]
        out[i] = (v[1] - v[0]) / (2.0 * h)
    return out
