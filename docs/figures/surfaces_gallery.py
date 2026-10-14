"""Implied-vol gallery for the models on the committed SPX book.

The figures use standardised moneyness ``k / sqrt(T)`` on ``[-3, 1.5]``.
Expiries under seven days are left out of the pictures and kept in the
JSON error tables. A cell is drawn only when that expiry quotes the
strike: there is no extrapolated sheet. Implied-vol panels share one
colour scale, ``[0.05, 0.90]``, unless a drawn cell sits a little outside
it, in which case the scale grows to cover the cell. A cell above 1.25
is left off the mesh and marked. Dupire local volatility keeps its own
vertical scale, set to the values actually drawn.

The market surface in the picture is a light Gaussian smooth along
``k / sqrt(T)``. The smooth is display only. The raw quotes are the
points, and the error tables use the unsmoothed interpolation on the
full book, including the short expiries. The error heatmap uses the
same figure domain. Its colour scale is the 98th percentile of the
absolute in-quote error, not the largest cell.

Heston and SABR are fit on the smile, ``|k| <= 0.4`` and at most 11 strikes
an expiry, the same cut as the calibration figures. Raw SVI uses every
quoted strike. The flat Black-Scholes volatility is the median of the
at-the-money implied vol across expiries. The Monte Carlo panel is exact
geometric Brownian motion at that volatility, one seed. The JSON tables
leave a node blank when the price does not clear two standard errors.
The figure draws those wings at the flat volatility, on the same quoted
domain as the market. The scatter that remains is the simulation noise.

Run from the repository root:

    python docs/figures/surfaces_gallery.py

Writes one PNG per surface, ``p2_surface_gallery.png``, and
``p2_surface_iv_error.png`` under ``docs/figures/``. Writes one
self-contained plotly HTML file per surface under ``docs/surfaces/``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Patch

from volsurface import bs, chains
from volsurface.core.market import market_data_from_snapshot
from volsurface.core.types import MarketData
from volsurface.fit.heston import HestonFitter
from volsurface.fit.sabr import SABRFitter
from volsurface.fit.svi import SVIFitter
from volsurface.mc.grid import estimate_terminal
from volsurface.models.gbm import GBM
from volsurface.models.heston import Heston
from volsurface.models.sabr import SABR

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "docs" / "figures"
HTML = ROOT / "docs" / "surfaces"

# Shared by every panel. Matplotlib azim is degrees from the x axis.
# The camera sits on the short-dated side and a little above the sheet.
VIEW_ELEV = 24.0
VIEW_AZIM = -58.0
PLOTLY_EYE = {"x": 1.35, "y": -1.75, "z": 0.95}
CMAP = plt.get_cmap("viridis")
N_K = 41
N_PATHS = 8_000
SEED = 2
# The smile cut used by the Heston and SABR calibration figures.
SMILE_ABS_K = 0.4
SMILE_STRIKES = 11
# Figures omit expiries shorter than a week. The JSON tables keep them.
DISPLAY_MIN_DAYS = 7
DISPLAY_T_MIN = DISPLAY_MIN_DAYS / 365.25
# Common grid in standardised moneyness. The drawn cells are the quoted subset.
M_LO = -3.0
M_HI = 1.5
M_STEP = 0.1
# Implied-vol colour scale. A drawn cell above IV_CAP is marked, not scaled in.
IV_LO = 0.05
IV_HI = 0.90
IV_CAP = 1.25
# Error heatmap. The limit is a high percentile of |error|, not the maximum.
ERROR_PERCENTILE = 98.0
ERROR_KEYS = ("svi", "heston", "sabr")
MARK_HIGH = "#9b2335"
MARK_LOW = "#1f4e79"


@dataclass
class Sheet:
    """One surface on the common grid. ``z`` has shape ``(n_T, n_k)``."""

    key: str
    title: str
    z: np.ndarray
    quantity: str


def quote_pillars(quotes: pd.DataFrame) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """``(T, k, iv)`` per expiry, sorted in calendar time, one row per strike."""
    pillars = []
    for maturity, group in quotes.groupby("T", sort=True):
        group = group.dropna(subset=["iv", "k"]).drop_duplicates("k").sort_values("k")
        if group.empty:
            continue
        pillars.append(
            (
                float(maturity),
                group["k"].to_numpy(dtype=float),
                group["iv"].to_numpy(dtype=float),
            )
        )
    return pillars


def common_axes(
    pillars: list[tuple[float, np.ndarray, np.ndarray]], n_k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Quoted maturities, and ``n_k`` log-moneyness nodes from the widest quotes."""
    if not pillars:
        raise ValueError("no expiry had an implied vol")
    if n_k < 2:
        raise ValueError(f"n_k must be at least 2, got {n_k}")
    maturities = np.array([row[0] for row in pillars], dtype=float)
    k_lo = min(float(row[1][0]) for row in pillars)
    k_hi = max(float(row[1][-1]) for row in pillars)
    if not k_hi > k_lo:
        raise ValueError("quoted log-moneyness has no width")
    return maturities, np.linspace(k_lo, k_hi, n_k)


def extrapolation_mask(
    pillars: list[tuple[float, np.ndarray, np.ndarray]], k: np.ndarray
) -> np.ndarray:
    """True where ``k`` is outside that expiry's own quoted strikes.

    The grid spans every expiry, so a short expiry's wings are inside the
    rectangle and still outside its quotes. Those cells are extrapolations.
    """
    mask = np.ones((len(pillars), k.size), dtype=bool)
    for i, (_maturity, k_obs, _iv) in enumerate(pillars):
        mask[i] = (k < k_obs[0]) | (k > k_obs[-1])
    return mask


def market_iv(pillars: list[tuple[float, np.ndarray, np.ndarray]], k: np.ndarray) -> np.ndarray:
    """Quoted implied vol, linear in ``k``, flat beyond that expiry's strikes."""
    out = np.empty((len(pillars), k.size), dtype=float)
    for i, (_maturity, k_obs, iv_obs) in enumerate(pillars):
        if k_obs.size == 1:
            out[i] = iv_obs[0]
        else:
            out[i] = np.interp(k, k_obs, iv_obs)
    return out


def flat_sigma(pillars: list[tuple[float, np.ndarray, np.ndarray]]) -> float:
    """Median across expiries of the quoted implied vol nearest ``k = 0``."""
    atm = []
    for _maturity, k_obs, iv_obs in pillars:
        atm.append(float(iv_obs[np.argmin(np.abs(k_obs))]))
    return float(np.median(np.asarray(atm, dtype=float)))


def _otm_iv_row(cash: np.ndarray, forward: float, k: np.ndarray, maturity: float, df: float):
    strike = forward * np.exp(k)
    kind = np.where(k >= 0.0, "call", "put")
    return bs.implied_vol(cash, forward, strike, maturity, df, kind)


def _simulate_gbm(sigma: float, maturities: np.ndarray, n_paths: int, seed: int):
    """Exact GBM paths. Column ``i + 1`` is the expiry ``maturities[i]``."""
    model = GBM(sigma=sigma, scheme="exact", antithetic=False, control_variate=False)
    times = np.concatenate(([0.0], np.asarray(maturities, dtype=float)))
    return model.simulate(n_paths, times, seed=seed)


def _gbm_one(data: MarketData, terminal: np.ndarray, maturity: float, k: np.ndarray) -> np.ndarray:
    """Implied vol of one expiry. Blank where the price does not clear two standard errors."""
    forward = float(np.asarray(data.F(maturity), dtype=float))
    discount = float(np.asarray(data.df(maturity), dtype=float))
    strike = forward * np.exp(k)
    # Out of the money: a put when k < 0, a call when k >= 0.
    kind = np.where(k < 0.0, "put", "call")
    value, stderr = estimate_terminal(
        terminal,
        forward,
        strike,
        discount,
        kind,
        antithetic=False,
        control_variate=False,
    )
    iv = _otm_iv_row(value, forward, k, maturity, discount)
    return np.where(value > 2.0 * stderr, iv, np.nan)


def gbm_iv(
    data: MarketData,
    maturities: np.ndarray,
    k: np.ndarray,
    sigma: float,
    *,
    n_paths: int,
    seed: int,
    paths=None,
) -> np.ndarray:
    """Exact GBM at ``sigma``, one set of paths, inverted to implied vol.

    Each expiry uses its own forward. Strikes left of the money are puts.
    A price with no implied vol, which is a wing no path reached, is NaN.
    Pass ``paths`` from :func:`_simulate_gbm` to reuse one simulation.
    """
    if paths is None:
        paths = _simulate_gbm(sigma, maturities, n_paths, seed)
    out = np.empty((maturities.size, k.size), dtype=float)
    for i, maturity in enumerate(maturities):
        out[i] = _gbm_one(data, paths.Z[:, i + 1], float(maturity), k)
    return out


def thin_smile(quotes: pd.DataFrame) -> pd.DataFrame:
    """Up to 11 strikes per expiry inside ``|k| <= 0.4``. Same cut as the fits."""
    kept = []
    for _, group in quotes.groupby("expiry", sort=True):
        group = group[np.abs(group["k"]) <= SMILE_ABS_K].dropna(subset=["iv", "price"])
        if len(group) < 5:
            continue
        if len(group) > SMILE_STRIKES:
            order = np.argsort(group["k"].to_numpy())
            pick = np.linspace(0, len(order) - 1, SMILE_STRIKES).round().astype(int)
            group = group.iloc[order[pick]]
        kept.append(group)
    if not kept:
        raise ValueError("no expiry had five strikes inside |k| <= 0.4")
    return pd.concat(kept, ignore_index=True)


def _smile_book(data: MarketData) -> MarketData:
    return MarketData(
        asof=data.asof,
        quotes=thin_smile(data.quotes),
        F=data.F,
        df=data.df,
        F_err=data.F_err,
        day_count=data.day_count,
    )


def heston_iv(model: Heston, data: MarketData, maturities: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Lewis implied vol. One forward and one discount factor per expiry."""
    out = np.empty((maturities.size, k.size), dtype=float)
    for i, maturity in enumerate(maturities):
        forward = float(np.asarray(data.F(maturity), dtype=float))
        discount = float(np.asarray(data.df(maturity), dtype=float))
        strike = forward * np.exp(k)
        call = np.asarray(
            model.price(forward, strike, float(maturity), discount, "call"), dtype=float
        )
        put = call - discount * (forward - strike)
        cash = np.where(k >= 0.0, call, put)
        out[i] = _otm_iv_row(cash, forward, k, float(maturity), discount)
    return out


def sabr_iv(
    report: pd.DataFrame, data: MarketData, maturities: np.ndarray, k: np.ndarray
) -> np.ndarray:
    """Hagan implied vol, one fitted ``(alpha, rho, nu)`` per expiry. ``beta`` is 1."""
    out = np.empty((maturities.size, k.size), dtype=float)
    fitted_t = report["T"].to_numpy(dtype=float)
    for i, maturity in enumerate(maturities):
        row = int(np.argmin(np.abs(fitted_t - maturity)))
        if abs(float(fitted_t[row]) - float(maturity)) > 1e-8:
            out[i] = np.nan
            continue
        params = report.iloc[row]
        model = SABR(
            alpha=float(params["alpha"]),
            beta=float(params["beta"]),
            rho=float(params["rho"]),
            nu=float(params["nu"]),
        )
        forward = float(np.asarray(data.F(maturity), dtype=float))
        strike = forward * np.exp(k)
        out[i] = np.asarray(model.implied_vol(forward, strike, float(maturity)), dtype=float)
    return out


def local_vol(surface, maturities: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Square root of Dupire local variance. Non-positive variance is left blank."""
    variance = np.asarray(surface.local_var(maturities, k), dtype=float)
    out = np.full(variance.shape, np.nan, dtype=float)
    positive = np.isfinite(variance) & (variance > 0.0)
    out[positive] = np.sqrt(variance[positive])
    return out


def _quote_date(data: MarketData) -> str:
    stamp = pd.Timestamp(data.asof)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("America/New_York")
    return f"{stamp.day} {stamp.strftime('%B')} {stamp.year}"


def standardised_axis() -> np.ndarray:
    """Grid of ``k / sqrt(T)`` from ``M_LO`` to ``M_HI`` in steps of ``M_STEP``."""
    count = int(round((M_HI - M_LO) / M_STEP))
    return np.linspace(M_LO, M_HI, count + 1)


def pillars_for_display(pillars: list[tuple[float, np.ndarray, np.ndarray]]):
    """Expiries of at least seven days. Shorter ones stay in the error tables."""
    return [row for row in pillars if row[0] >= DISPLAY_T_MIN]


def quote_mask(pillars: list[tuple[float, np.ndarray, np.ndarray]], m: np.ndarray) -> np.ndarray:
    """True where ``m * sqrt(T)`` lies inside that expiry's own quoted strikes."""
    mask = np.zeros((len(pillars), m.size), dtype=bool)
    for i, (maturity, k_obs, _iv) in enumerate(pillars):
        k_query = m * np.sqrt(maturity)
        mask[i] = (k_query >= k_obs[0]) & (k_query <= k_obs[-1])
    return mask


def market_on_moneyness(
    pillars: list[tuple[float, np.ndarray, np.ndarray]], m: np.ndarray
) -> np.ndarray:
    """Quoted implied vol, linear in ``k``. Nothing is drawn past the quotes."""
    out = np.full((len(pillars), m.size), np.nan)
    for i, (maturity, k_obs, iv_obs) in enumerate(pillars):
        if k_obs.size < 2:
            continue
        k_query = m * np.sqrt(maturity)
        inside = (k_query >= k_obs[0]) & (k_query <= k_obs[-1])
        out[i, inside] = np.interp(k_query[inside], k_obs, iv_obs)
    return out


def smooth_along_moneyness(z: np.ndarray, sigma_steps: float = 1.0) -> np.ndarray:
    """Light Gaussian along standardised moneyness. NaN cells stay NaN.

    The kernel uses finite neighbours only, and a cell outside the quotes is
    not filled in. ``sigma_steps`` is in grid steps.
    """
    if sigma_steps <= 0.0 or z.size == 0:
        return np.array(z, dtype=float, copy=True)
    radius = int(np.ceil(3.0 * sigma_steps))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma_steps) ** 2)
    finite = np.isfinite(z)
    out = np.full(z.shape, np.nan, dtype=float)
    for j in range(z.shape[1]):
        acc = np.zeros(z.shape[0], dtype=float)
        wsum = np.zeros(z.shape[0], dtype=float)
        for offset, weight in zip(offsets, kernel, strict=True):
            src = j + int(offset)
            if src < 0 or src >= z.shape[1]:
                continue
            ok = finite[:, src]
            acc[ok] += weight * z[ok, src]
            wsum[ok] += weight
        keep = finite[:, j] & (wsum > 0.0)
        out[keep, j] = acc[keep] / wsum[keep]
    return out


def quote_cloud(pillars: list[tuple[float, np.ndarray, np.ndarray]]):
    """Raw quotes on the display expiries, as ``(m, T, iv)``."""
    ms: list[np.ndarray] = []
    ts: list[np.ndarray] = []
    ivs: list[np.ndarray] = []
    for maturity, k_obs, iv_obs in pillars:
        if maturity < DISPLAY_T_MIN:
            continue
        moneyness = k_obs / np.sqrt(maturity)
        keep = (moneyness >= M_LO) & (moneyness <= M_HI) & np.isfinite(iv_obs)
        if not np.any(keep):
            continue
        ms.append(moneyness[keep])
        ts.append(np.full(int(np.sum(keep)), maturity))
        ivs.append(iv_obs[keep])
    if not ms:
        empty = np.array([], dtype=float)
        return empty, empty, empty
    return np.concatenate(ms), np.concatenate(ts), np.concatenate(ivs)


def _rows_of(maturities: np.ndarray, k_rows: np.ndarray, fn) -> np.ndarray:
    """``fn(maturity, k)`` per expiry. ``k`` changes with ``T`` because ``k = m sqrt(T)``."""
    out = np.empty(k_rows.shape, dtype=float)
    for i, maturity in enumerate(maturities):
        out[i] = np.asarray(fn(float(maturity), k_rows[i]), dtype=float).reshape(-1)
    return out


def _apply_quotes(z: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.array(z, dtype=float, copy=True)
    out[~mask] = np.nan
    return out


def _view_m_limits(mask: np.ndarray, m: np.ndarray) -> tuple[float, float]:
    """Frame the quoted columns, with a small pad, inside the sampling grid."""
    columns = np.any(mask, axis=0)
    if not np.any(columns):
        return M_LO, M_HI
    used = m[columns]
    pad = 1.5 * M_STEP
    lo = max(M_LO, float(used[0]) - pad)
    hi = min(M_HI, float(used[-1]) + pad)
    return round(lo, 2), round(hi, 2)


def choose_iv_axis(sheets: list[Sheet]) -> tuple[float, float]:
    """One implied-vol scale. Grow it to cover drawn cells, until ``IV_CAP``.

    A cell above the cap is a blow-up and is marked later. A cell between
    ``IV_HI`` and the cap widens the shared scale so it does not become a cross.
    """
    pieces = []
    for sheet in sheets:
        if sheet.quantity != "implied volatility":
            continue
        finite = sheet.z[np.isfinite(sheet.z)]
        if finite.size:
            pieces.append(finite)
    if not pieces:
        return IV_LO, IV_HI
    values = np.concatenate(pieces)
    kept = values[(values >= 0.0) & (values <= IV_CAP)]
    if kept.size == 0:
        return IV_LO, IV_HI
    lo_data = float(np.min(kept))
    hi_data = float(np.max(kept))
    lo = IV_LO if lo_data >= IV_LO else max(0.0, lo_data - 0.01)
    hi = IV_HI if hi_data <= IV_HI else hi_data + 0.02
    return float(lo), float(hi)


def _local_axis(z: np.ndarray) -> tuple[float, float]:
    """Dupire's own scale, set just above the values drawn on this domain."""
    finite = z[np.isfinite(z)]
    if finite.size == 0:
        return 0.0, 1.0
    hi = float(np.max(finite))
    return 0.0, hi + 0.02 * max(hi, 1.0)


def build_display(
    data: MarketData,
    pillars: list[tuple[float, np.ndarray, np.ndarray]],
    maturities: np.ndarray,
    paths,
    *,
    sigma: float,
    heston: Heston,
    sabr_report: pd.DataFrame,
    svi_surface,
    date: str,
    n_paths: int,
    seed: int,
) -> dict:
    """Surfaces on ``k / sqrt(T)``, expiries of at least seven days, quotes only."""
    kept = pillars_for_display(pillars)
    if not kept:
        raise ValueError("no expiry is at least 7 days")
    m = standardised_axis()
    d_maturities = np.array([row[0] for row in kept], dtype=float)
    positions = [i for i, maturity in enumerate(maturities) if float(maturity) >= DISPLAY_T_MIN]
    if len(positions) != d_maturities.size:
        raise ValueError("display expiries do not match the quote pillars")
    k_rows = m[None, :] * np.sqrt(d_maturities)[:, None]
    mask = quote_mask(kept, m)
    market_raw = market_on_moneyness(kept, m)
    market_smooth = smooth_along_moneyness(market_raw, sigma_steps=1.0)

    def _heston_row(maturity: float, k: np.ndarray) -> np.ndarray:
        return heston_iv(heston, data, np.array([maturity]), k)[0]

    def _sabr_row(maturity: float, k: np.ndarray) -> np.ndarray:
        return sabr_iv(sabr_report, data, np.array([maturity]), k)[0]

    def _svi_row(maturity: float, k: np.ndarray) -> np.ndarray:
        return np.asarray(svi_surface.iv(np.array([maturity]), k), dtype=float)[0]

    def _dupire_row(maturity: float, k: np.ndarray) -> np.ndarray:
        return local_vol(svi_surface, np.array([maturity]), k)[0]

    gbm = np.empty(k_rows.shape, dtype=float)
    for row, src in enumerate(positions):
        gbm[row] = _gbm_one(data, paths.Z[:, src + 1], float(d_maturities[row]), k_rows[row])
    # Same footprint as the market. A wing no path reached is the flat volatility,
    # which is the value the Monte Carlo is estimating. The JSON tables still
    # leave those cells blank.
    on_market = np.isfinite(market_raw)
    flat = np.where(on_market, sigma, np.nan)
    gbm_draw = np.where(on_market, sigma, np.nan)
    resolved = on_market & np.isfinite(gbm)
    gbm_draw[resolved] = gbm[resolved]
    svi = _rows_of(d_maturities, k_rows, _svi_row)
    heston_surface = _rows_of(d_maturities, k_rows, _heston_row)
    sabr_surface = _rows_of(d_maturities, k_rows, _sabr_row)
    dupire = _rows_of(d_maturities, k_rows, _dupire_row)

    smile_note = "fit on |k| <= 0.4"
    sheets = [
        Sheet("market", f"SPX market, display smooth, {date}", market_smooth, "implied volatility"),
        Sheet(
            "black_scholes",
            f"Black-Scholes flat, sigma {sigma:.3f}, {date}",
            flat,
            "implied volatility",
        ),
        Sheet(
            "gbm_mc",
            f"Monte Carlo GBM, {n_paths} paths, seed {seed}, wings at sigma, {date}",
            gbm_draw,
            "implied volatility",
        ),
        Sheet("svi", f"raw SVI, {date}", _apply_quotes(svi, mask), "implied volatility"),
        Sheet(
            "heston",
            f"Heston, {smile_note}, {date}",
            _apply_quotes(heston_surface, mask),
            "implied volatility",
        ),
        Sheet(
            "sabr",
            f"SABR, {smile_note}, {date}",
            _apply_quotes(sabr_surface, mask),
            "implied volatility",
        ),
        Sheet(
            "dupire_localvol",
            f"Dupire local volatility from raw SVI, not implied vol, {date}",
            _apply_quotes(dupire, mask),
            "local volatility",
        ),
    ]
    quote_m, quote_t, quote_iv = quote_cloud(pillars)
    m_lo, m_hi = _view_m_limits(mask, m)
    local_lo, local_hi = _local_axis(sheets[-1].z)
    return {
        "maturities": d_maturities,
        "m": m,
        "k_rows": k_rows,
        "quoted": mask,
        "market_raw": market_raw,
        "sheets": sheets,
        "quote_m": quote_m,
        "quote_t": quote_t,
        "quote_iv": quote_iv,
        "m_view": (m_lo, m_hi),
        "iv_axis": choose_iv_axis(sheets),
        "local_axis": (local_lo, local_hi),
    }


def build_sheets(
    data: MarketData,
    *,
    n_k: int = N_K,
    n_paths: int = N_PATHS,
    seed: int = SEED,
    heston_starts: int = 4,
    sabr_starts: int = 4,
) -> dict:
    """Fit the book and sample every surface on the quoted grid."""
    pillars = quote_pillars(data.quotes)
    maturities, k = common_axes(pillars, n_k)
    extrapolated = extrapolation_mask(pillars, k)
    market = market_iv(pillars, k)
    sigma = flat_sigma(pillars)
    date = _quote_date(data)

    smile = _smile_book(data)
    heston_fit = HestonFitter(n_starts=heston_starts, loss="vega", seed=seed).fit(smile)
    row = heston_fit.slice_report.iloc[0]
    heston = Heston(
        v0=float(row["v0"]),
        kappa=float(row["kappa"]),
        theta=float(row["theta"]),
        xi=float(row["xi"]),
        rho=float(row["rho"]),
    )
    sabr_fit = SABRFitter(beta=1.0, n_starts=sabr_starts, loss="vega", seed=seed).fit(smile)
    svi_fit = SVIFitter().fit(data)

    paths = _simulate_gbm(sigma, maturities, n_paths, seed)
    gbm = gbm_iv(data, maturities, k, sigma, n_paths=n_paths, seed=seed, paths=paths)
    svi = np.asarray(svi_fit.surface.iv(maturities, k), dtype=float)
    heston_surface = heston_iv(heston, data, maturities, k)
    sabr_surface = sabr_iv(sabr_fit.slice_report, data, maturities, k)
    dupire = local_vol(svi_fit.surface, maturities, k)
    flat = np.full_like(market, sigma)

    smile_note = "fit on |k| <= 0.4"
    sheets = [
        Sheet("market", f"SPX market, {date}", market, "implied volatility"),
        Sheet(
            "black_scholes",
            f"Black-Scholes flat, sigma {sigma:.3f}, {date}",
            flat,
            "implied volatility",
        ),
        Sheet(
            "gbm_mc",
            f"Monte Carlo GBM, {n_paths} paths, seed {seed}, {date}",
            gbm,
            "implied volatility",
        ),
        Sheet("svi", f"raw SVI, {date}", svi, "implied volatility"),
        Sheet("heston", f"Heston, {smile_note}, {date}", heston_surface, "implied volatility"),
        Sheet("sabr", f"SABR, {smile_note}, {date}", sabr_surface, "implied volatility"),
        Sheet(
            "dupire_localvol",
            f"Dupire local volatility from raw SVI, not implied vol, {date}",
            dupire,
            "local volatility",
        ),
    ]
    return {
        "date": date,
        "maturities": maturities,
        "k": k,
        "extrapolated": extrapolated,
        "sigma": sigma,
        "sheets": sheets,
        "heston": {
            "v0": heston.v0,
            "kappa": heston.kappa,
            "theta": heston.theta,
            "xi": heston.xi,
            "rho": heston.rho,
            "feller": float(row["feller"]),
            "rmse_price": float(row["rmse_price"]),
        },
        "sabr_report": sabr_fit.slice_report,
        "n_paths": n_paths,
        "seed": seed,
        "display": build_display(
            data,
            pillars,
            maturities,
            paths,
            sigma=sigma,
            heston=heston,
            sabr_report=sabr_fit.slice_report,
            svi_surface=svi_fit.surface,
            date=date,
            n_paths=n_paths,
            seed=seed,
        ),
    }


def _inside_gap(
    model: np.ndarray,
    market: np.ndarray,
    inside: np.ndarray,
    maturities: np.ndarray,
    k: np.ndarray,
) -> dict:
    gap = model - market
    ok = inside & np.isfinite(gap)
    if not np.any(ok):
        return {"n": 0, "rmse": None, "max_abs": None, "mean": None}
    values = gap[ok]
    flat = np.where(ok, np.abs(gap), -1.0)
    loc = np.unravel_index(int(np.argmax(flat)), flat.shape)
    return {
        "n": int(ok.sum()),
        "rmse": float(np.sqrt(np.mean(values**2))),
        "max_abs": float(np.max(np.abs(values))),
        "mean": float(np.mean(values)),
        "max_abs_T": float(maturities[loc[0]]),
        "max_abs_k": float(k[loc[1]]),
        "max_signed": float(gap[loc]),
    }


def summary_numbers(built: dict) -> dict:
    """Gaps against the interpolated market, inside the quotes only."""
    sheets = {sheet.key: sheet for sheet in built["sheets"]}
    market = sheets["market"].z
    k = built["k"]
    extrapolated = built["extrapolated"]
    inside = ~extrapolated
    smile = inside & (np.abs(k)[None, :] <= SMILE_ABS_K)
    quoted = market[inside & np.isfinite(market)]
    numbers = {
        "date": built["date"],
        "n_expiries": int(built["maturities"].size),
        "n_k": int(k.size),
        "T_min": float(built["maturities"][0]),
        "T_max": float(built["maturities"][-1]),
        "k_min": float(k[0]),
        "k_max": float(k[-1]),
        "flat_sigma": built["sigma"],
        "n_extrapolated": int(extrapolated.sum()),
        "n_cells": int(extrapolated.size),
        "heston_fit": built["heston"],
        "n_paths": built["n_paths"],
        "seed": built["seed"],
        "iv_scale": [float(np.min(quoted)), float(np.max(quoted))] if quoted.size else None,
    }
    for key in ("black_scholes", "gbm_mc", "svi", "heston", "sabr"):
        z = sheets[key].z
        finite = z[np.isfinite(z)]
        numbers[key] = {
            "inside": _inside_gap(z, market, inside, built["maturities"], k),
            "smile": _inside_gap(z, market, smile, built["maturities"], k),
            "n_missing": int(np.sum(inside & ~np.isfinite(z))),
            "span": None if finite.size == 0 else float(np.max(finite) - np.min(finite)),
        }
    local = sheets["dupire_localvol"].z
    finite = inside & np.isfinite(local)
    if np.any(finite):
        at_max = np.where(finite, local, -np.inf)
        loc = np.unravel_index(int(np.argmax(at_max)), at_max.shape)
        local_max_t = float(built["maturities"][loc[0]])
        local_max_k = float(k[loc[1]])
        local_min = float(np.min(local[finite]))
        local_max = float(np.max(local[finite]))
    else:
        local_max_t = local_max_k = local_min = local_max = None
    numbers["dupire_localvol"] = {
        "n_finite_inside": int(finite.sum()),
        "min": local_min,
        "max": local_max,
        "max_T": local_max_t,
        "max_k": local_max_k,
        "n_missing_inside": int(np.sum(inside & ~np.isfinite(local))),
    }
    j_atm = int(np.argmin(np.abs(k)))
    atm = sheets["gbm_mc"].z[:, j_atm]
    atm_ok = np.isfinite(atm)
    if np.any(atm_ok):
        atm_gap = atm[atm_ok] - built["sigma"]
        numbers["gbm_atm"] = {
            "k": float(k[j_atm]),
            "n": int(atm_ok.sum()),
            "rmse": float(np.sqrt(np.mean(atm_gap**2))),
            "max_abs": float(np.max(np.abs(atm_gap))),
        }
    else:
        numbers["gbm_atm"] = {"k": float(k[j_atm]), "n": 0, "rmse": None, "max_abs": None}
    report = built["sabr_report"]
    numbers["sabr_fit"] = {
        "n_expiries": int(len(report)),
        "nu_on_4": int(np.sum(report["nu"] > 3.999)),
        "rho_on_lower": int(np.sum(report["rho"] < -0.998)),
        "rmse_median": float(report["rmse_price"].median()),
    }
    display = built["display"]
    omitted = [float(t) for t in built["maturities"] if float(t) < DISPLAY_T_MIN]
    quoted_m = display["m"][np.any(display["quoted"], axis=0)]
    numbers["display"] = {
        "coordinate": "k/sqrt(T)",
        "m_min": M_LO,
        "m_max": M_HI,
        "m_step": M_STEP,
        "min_days": DISPLAY_MIN_DAYS,
        "T_min_year": DISPLAY_T_MIN,
        "T_min_drawn": float(display["maturities"][0]),
        "T_max_drawn": float(display["maturities"][-1]),
        "n_expiries_drawn": int(display["maturities"].size),
        "n_expiries_omitted": len(omitted),
        "omitted_T": omitted,
        "quoted_m_min": None if quoted_m.size == 0 else float(quoted_m[0]),
        "quoted_m_max": None if quoted_m.size == 0 else float(quoted_m[-1]),
        "m_view": [float(display["m_view"][0]), float(display["m_view"][1])],
        "n_quote_points": int(display["quote_iv"].size),
        "market_smooth": "gaussian along k/sqrt(T), sigma one grid step, display only",
        "implied_vol_axis": [float(display["iv_axis"][0]), float(display["iv_axis"][1])],
        "local_vol_axis": [float(display["local_axis"][0]), float(display["local_axis"][1])],
        "clips": _display_clips(display),
        "error_heatmap": summarise_errors(error_gaps(display)),
    }
    return numbers


def clipped_mask(z: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Finite cells outside ``[lo, hi]``. These must not set the axis."""
    return np.isfinite(z) & ((z < lo) | (z > hi))


def layer_z(z: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Copy of ``z`` with every cell outside ``keep`` set to NaN."""
    out = np.array(z, dtype=float, copy=True)
    out[~keep] = np.nan
    return out


def clip_report(
    z: np.ndarray,
    extrapolated: np.ndarray,
    maturities: np.ndarray,
    k: np.ndarray,
    lo: float,
    hi: float,
    n_worst: int = 25,
) -> dict:
    """How many cells sit outside the locked window, and the furthest ones."""
    finite = np.isfinite(z)
    above = finite & (z > hi)
    below = finite & (z < lo)
    clipped = above | below
    excess = np.full(z.shape, -np.inf)
    excess[above] = z[above] - hi
    excess[below] = lo - z[below]
    worst = []
    order = np.argsort(excess, axis=None)[::-1]
    for flat in order:
        if len(worst) >= n_worst:
            break
        i, j = np.unravel_index(int(flat), z.shape)
        if not clipped[i, j]:
            break
        worst.append(
            {
                "T": float(maturities[i]),
                "k": float(k[j]),
                "z": float(z[i, j]),
                "extrapolated": bool(extrapolated[i, j]),
                "side": "above" if above[i, j] else "below",
            }
        )
    raw = z[finite]
    return {
        "lo": float(lo),
        "hi": float(hi),
        "n_clipped": int(clipped.sum()),
        "n_above": int(above.sum()),
        "n_below": int(below.sum()),
        "n_clipped_inside": int((clipped & ~extrapolated).sum()),
        "n_clipped_extrapolated": int((clipped & extrapolated).sum()),
        "raw_min": None if raw.size == 0 else float(np.min(raw)),
        "raw_max": None if raw.size == 0 else float(np.max(raw)),
        "worst": worst,
    }


def _display_clips(display: dict) -> dict:
    """Cells on the figure grid that fall outside the axis actually used."""
    maturities = display["maturities"]
    m = display["m"]
    k_rows = display["k_rows"]
    iv_lo, iv_hi = display["iv_axis"]
    local_lo, local_hi = display["local_axis"]
    clips = {}
    for sheet in display["sheets"]:
        if sheet.quantity == "local volatility":
            lo, hi = local_lo, local_hi
        else:
            lo, hi = iv_lo, iv_hi
        clips[sheet.key] = _clip_on_moneyness(sheet.z, maturities, m, k_rows, lo, hi)
    return clips


def _clip_on_moneyness(
    z: np.ndarray,
    maturities: np.ndarray,
    m: np.ndarray,
    k_rows: np.ndarray,
    lo: float,
    hi: float,
    n_worst: int = 15,
) -> dict:
    finite = np.isfinite(z)
    above = finite & (z > hi)
    below = finite & (z < lo)
    clipped = above | below
    excess = np.full(z.shape, -np.inf)
    excess[above] = z[above] - hi
    excess[below] = lo - z[below]
    worst = []
    order = np.argsort(excess, axis=None)[::-1]
    for flat in order:
        if len(worst) >= n_worst:
            break
        i, j = np.unravel_index(int(flat), z.shape)
        if not clipped[i, j]:
            break
        worst.append(
            {
                "T": float(maturities[i]),
                "m": float(m[j]),
                "k": float(k_rows[i, j]),
                "z": float(z[i, j]),
                "side": "above" if above[i, j] else "below",
            }
        )
    raw = z[finite]
    return {
        "lo": float(lo),
        "hi": float(hi),
        "n_finite": int(finite.sum()),
        "n_clipped": int(clipped.sum()),
        "n_above": int(above.sum()),
        "n_below": int(below.sum()),
        "raw_min": None if raw.size == 0 else float(np.min(raw)),
        "raw_max": None if raw.size == 0 else float(np.max(raw)),
        "worst": worst,
    }


def _face_colours(z: np.ndarray, norm: Normalize, alpha: float = 1.0) -> np.ndarray:
    """One RGBA colour per face. A face is drawn only when all four corners are finite."""
    stack = np.stack([z[:-1, :-1], z[:-1, 1:], z[1:, :-1], z[1:, 1:]], axis=0)
    ok = np.all(np.isfinite(stack), axis=0)
    total = np.sum(np.where(np.isfinite(stack), stack, 0.0), axis=0)
    face_z = np.zeros(ok.shape, dtype=float)
    np.divide(total, 4.0, out=face_z, where=ok)
    colours = CMAP(norm(np.where(ok, face_z, norm.vmin)))
    colours[..., 3] = np.where(ok, alpha, 0.0)
    return colours


def _plot_layer(ax, k, maturities, z: np.ndarray, faces: np.ndarray) -> None:
    if not np.any(np.isfinite(z)):
        return
    kk, tt = np.meshgrid(k, maturities)
    ax.plot_surface(
        kk,
        tt,
        z,
        facecolors=faces,
        rstride=1,
        cstride=1,
        linewidth=0,
        edgecolor=(0, 0, 0, 0),
        antialiased=False,
        shade=False,
    )


def _mark_clipped(ax, k, maturities, z: np.ndarray, lo: float, hi: float, size: float) -> None:
    """Crosses at the window edge. The raw value is not used as a height."""
    finite = np.isfinite(z)
    for mask, height, colour in (
        (finite & (z > hi), hi, MARK_HIGH),
        (finite & (z < lo), lo, MARK_LOW),
    ):
        rows, cols = np.nonzero(mask)
        if rows.size == 0:
            continue
        ax.scatter(
            k[cols],
            maturities[rows],
            np.full(rows.shape, height),
            c=colour,
            marker="x",
            s=size,
            depthshade=False,
            linewidths=0.6,
        )


def _axis_for(display: dict, sheet: Sheet) -> tuple[float, float]:
    if sheet.quantity == "local volatility":
        return display["local_axis"]
    return display["iv_axis"]


def _draw_surface(ax, display: dict, sheet: Sheet, *, points, point_size: float, mark_size: float):
    lo, hi = _axis_for(display, sheet)
    m = display["m"]
    maturities = display["maturities"]
    drawn = layer_z(sheet.z, ~clipped_mask(sheet.z, lo, hi))
    norm = Normalize(vmin=lo, vmax=hi)
    alpha = 0.88 if points is not None else 1.0
    _plot_layer(ax, m, maturities, drawn, _face_colours(drawn, norm, alpha))
    if points is not None and points[0].size:
        ax.scatter(
            points[0],
            points[1],
            points[2],
            c="#8a8a8a",
            s=point_size,
            alpha=0.28,
            depthshade=False,
            linewidths=0,
        )
    if int(clipped_mask(sheet.z, lo, hi).sum()):
        _mark_clipped(ax, m, maturities, sheet.z, lo, hi, size=mark_size)
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    ax.set_xlim(float(display["m_view"][0]), float(display["m_view"][1]))
    ax.set_ylim(float(maturities[0]), float(maturities[-1]))
    ax.set_zlim(lo, hi)
    ax.set_xlabel("k / √T")
    ax.set_ylabel("T")
    ax.set_zlabel(sheet.quantity)
    ax.set_title(sheet.title, fontsize=9)
    ax.tick_params(labelsize=7)


def write_pngs(built: dict, fig_dir: Path) -> None:
    """One PNG per surface, the seven-panel gallery, and the error heatmap."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    display = built["display"]
    points = (display["quote_m"], display["quote_t"], display["quote_iv"])

    for sheet in display["sheets"]:
        fig = plt.figure(figsize=(7.6, 5.8))
        ax = fig.add_subplot(111, projection="3d")
        _draw_surface(
            ax,
            display,
            sheet,
            points=points if sheet.key == "market" else None,
            point_size=8.0,
            mark_size=16.0,
        )
        lo, hi = _axis_for(display, sheet)
        fig.colorbar(
            ScalarMappable(norm=Normalize(vmin=lo, vmax=hi), cmap=CMAP),
            ax=ax,
            fraction=0.046,
            pad=0.08,
            label=f"{sheet.quantity} on [{lo:.2f}, {hi:.2f}]",
        )
        note = "k/√T, inside each expiry's quotes. Expiries under 7 days omitted."
        if sheet.key == "market":
            note = "k/√T, inside quotes. Points are raw quotes. Expiries under 7 days omitted."
        fig.text(0.5, 0.01, note, ha="center", va="bottom", fontsize=8)
        fig.subplots_adjust(left=0.02, right=0.92, bottom=0.08, top=0.92)
        fig.savefig(fig_dir / f"p2_surface_{sheet.key}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

    fig = plt.figure(figsize=(16.5, 9.4))
    for i, sheet in enumerate(display["sheets"]):
        ax = fig.add_subplot(2, 4, i + 1, projection="3d")
        _draw_surface(
            ax,
            display,
            sheet,
            points=None,
            point_size=2.0,
            mark_size=8.0,
        )
    fig.suptitle(
        f"SPX {built['date']}. k/√T, expiries under 7 days omitted, inside the quotes.",
        fontsize=12,
    )
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.92, wspace=0.08, hspace=0.22)
    fig.savefig(fig_dir / "p2_surface_gallery.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    _write_error_heatmap(built, fig_dir)


def error_colour_limit(abs_errors: np.ndarray, percentile: float = ERROR_PERCENTILE) -> float:
    """Symmetric colour limit from a high percentile of ``|error|``, in vol points.

    A handful of large cells must not set the scale. An empty sample falls
    back to 1 vol point.
    """
    if abs_errors.size == 0:
        return 1.0
    limit = float(np.quantile(np.asarray(abs_errors, dtype=float), percentile / 100.0))
    if not np.isfinite(limit) or limit <= 0.0:
        return 1.0
    return limit


def error_gaps(display: dict) -> dict[str, np.ndarray]:
    """Model minus unsmoothed market, in vol points, on the figure grid.

    Cells outside that expiry's quotes, and cells with no finite pair, are NaN.
    """
    sheets = {sheet.key: sheet for sheet in display["sheets"]}
    market = display["market_raw"]
    quoted = display["quoted"]
    gaps = {}
    for key in ERROR_KEYS:
        gap = 100.0 * (sheets[key].z - market)
        gaps[key] = np.where(quoted & np.isfinite(gap), gap, np.nan)
    return gaps


def summarise_errors(gaps: dict[str, np.ndarray]) -> dict:
    """Percentile colour limit and the in-quote RMSE of each panel."""
    panels = {}
    pieces = []
    for key in ERROR_KEYS:
        values = gaps[key][np.isfinite(gaps[key])]
        pieces.append(np.abs(values))
        panels[key] = {
            "n": int(values.size),
            "rmse_vol_points": None if values.size == 0 else float(np.sqrt(np.mean(values**2))),
            "max_abs_vol_points": None if values.size == 0 else float(np.max(np.abs(values))),
            "p98_abs_vol_points": None if values.size == 0 else error_colour_limit(np.abs(values)),
        }
    pooled = np.concatenate(pieces) if pieces else np.array([], dtype=float)
    raw = error_colour_limit(pooled)
    stated = _stated_limit(raw)
    beyond = int(np.sum(pooled > stated)) if pooled.size else 0
    return {
        "percentile": ERROR_PERCENTILE,
        "limit_vol_points": stated,
        "limit_raw": raw,
        "n_errors": int(pooled.size),
        "n_beyond_limit": beyond,
        "panels": panels,
    }


def _limit_text(limit: float) -> str:
    """Digits for the caption. A limit under 0.01 keeps four places so it does not print as 0."""
    text = f"{limit:.2f}"
    if float(text) <= 0.0:
        return f"{limit:.4f}"
    return text


def _stated_limit(raw: float) -> float:
    """Round the colour limit to the digits the caption prints. Never round it to zero."""
    stated = float(_limit_text(raw))
    return stated if stated > 0.0 else raw


def _shade_unquoted(ax, m: np.ndarray, y: np.ndarray, show: np.ndarray) -> None:
    """Pale grey where the error is not drawn. White on the colour scale means zero."""
    if not np.any(~show):
        return
    hidden = np.ma.array(np.ones(show.shape, dtype=float), mask=show)
    ax.pcolormesh(
        m,
        y,
        hidden,
        shading="nearest",
        cmap=ListedColormap(["#f0f0f0"]),
        vmin=0.0,
        vmax=1.0,
        linewidth=0,
        zorder=1,
    )


def _write_error_heatmap(built: dict, fig_dir: Path) -> None:
    """Model minus unsmoothed market, on the same domain as the 3D figures."""
    display = built["display"]
    gaps = error_gaps(display)
    stats = summarise_errors(gaps)
    limit = float(stats["limit_vol_points"])
    titles = {
        "svi": "SVI minus market",
        "heston": "Heston minus market",
        "sabr": "SABR minus market",
    }
    levels = [level for level in (-2.0, -1.0, 1.0, 2.0) if abs(level) < limit]
    m = display["m"]
    maturities = display["maturities"]
    y = np.sqrt(maturities)
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 5.2), sharey=True)
    mesh = None
    for ax, key in zip(axes, ERROR_KEYS, strict=True):
        gap = gaps[key]
        show = np.isfinite(gap)
        ax.set_facecolor("white")
        _shade_unquoted(ax, m, y, show)
        mesh = ax.pcolormesh(
            m,
            y,
            np.ma.array(gap, mask=~show),
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            shading="nearest",
            linewidth=0,
            zorder=2,
        )
        if levels and np.any(show):
            ax.contour(
                m,
                y,
                gap,
                levels=levels,
                colors="#222222",
                linewidths=0.7,
                zorder=3,
            )
        panel = stats["panels"][key]
        rmse = panel["rmse_vol_points"]
        if rmse is None:
            note = f"n = {panel['n']}"
        else:
            note = f"RMSE {rmse:.2f} vol points, n = {panel['n']}"
        ax.set_xlim(float(display["m_view"][0]), float(display["m_view"][1]))
        ax.set_xlabel("standardised moneyness k/√T")
        ax.set_title(f"{titles[key]}\n{note}", fontsize=10)
    _set_sqrt_maturity_ticks(axes[0], maturities)
    extend = "both" if stats["n_beyond_limit"] else "neither"
    fig.colorbar(mesh, ax=axes, fraction=0.046, pad=0.04, label="vol points", extend=extend)
    fig.legend(
        handles=[Patch(facecolor="#f0f0f0", edgecolor="#d0d0d0", label="outside quotes")],
        loc="lower center",
        bbox_to_anchor=(0.45, -0.02),
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        f"SPX {built['date']}. Saturates at ±{_limit_text(limit)} vol points. "
        "Grey is outside the quotes.",
        fontsize=12,
    )
    fig.subplots_adjust(left=0.07, right=0.90, bottom=0.16, top=0.84, wspace=0.16)
    fig.savefig(fig_dir / "p2_surface_iv_error.png", dpi=140, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def _set_sqrt_maturity_ticks(ax, maturities: np.ndarray) -> None:
    """Tick labels are calendar year fractions. Positions are ``sqrt(T)``."""
    candidates = np.array([0.02, 0.05, 0.10, 0.25, 0.50, 1.00])
    lo = float(maturities[0])
    hi = float(maturities[-1])
    ticks = candidates[(candidates >= lo - 1e-12) & (candidates <= hi + 1e-12)]
    if ticks.size == 0:
        ticks = np.array([lo, hi])
    ax.set_yticks(np.sqrt(ticks))
    ax.set_yticklabels([f"{t:.2f}" for t in ticks])
    ax.set_ylabel("maturity T, spaced by √T")


def _colourscale() -> list:
    scale = []
    for stop in (0.0, 0.25, 0.5, 0.75, 1.0):
        red, green, blue, _alpha = CMAP(stop)
        scale.append([stop, f"rgb({int(red * 255)},{int(green * 255)},{int(blue * 255)})"])
    return scale


def _plotly_marks(m, maturities, z: np.ndarray, lo: float, hi: float) -> list:
    """Crosses at the axis edge. Added only when a drawn cell leaves the scale."""
    traces = []
    finite = np.isfinite(z)
    for mask, height, colour, name in (
        (finite & (z > hi), hi, MARK_HIGH, "clipped above"),
        (finite & (z < lo), lo, MARK_LOW, "clipped below"),
    ):
        rows, cols = np.nonzero(mask)
        if rows.size == 0:
            continue
        raw = z[rows, cols]
        traces.append(
            go.Scatter3d(
                x=m[cols],
                y=maturities[rows],
                z=np.full(rows.shape, height),
                mode="markers",
                marker={"size": 3, "color": colour, "symbol": "x"},
                name=name,
                text=[f"raw {value:.4f}" for value in raw],
                hovertemplate=(
                    "k/√T=%{x:.3f}<br>T=%{y:.5f}<br>marked at %{z:.2f}"
                    "<br>%{text}<extra>" + name + "</extra>"
                ),
            )
        )
    return traces


def _plotly_surface(sheet: Sheet, built: dict) -> go.Figure:
    display = built["display"]
    lo, hi = _axis_for(display, sheet)
    m = display["m"]
    maturities = display["maturities"]
    drawn = layer_z(sheet.z, ~clipped_mask(sheet.z, lo, hi))
    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=m,
            y=maturities,
            z=drawn,
            cmin=lo,
            cmax=hi,
            colorscale=_colourscale(),
            colorbar={"title": sheet.quantity},
            name="surface",
            showscale=True,
        )
    )
    if sheet.key == "market" and display["quote_iv"].size:
        fig.add_trace(
            go.Scatter3d(
                x=display["quote_m"],
                y=display["quote_t"],
                z=display["quote_iv"],
                mode="markers",
                marker={"size": 1.5, "color": "rgba(140,140,140,0.35)"},
                name="raw quotes",
                hovertemplate=(
                    "k/√T=%{x:.3f}<br>T=%{y:.4f}<br>quote %{z:.4f}<extra>raw quote</extra>"
                ),
            )
        )
    for trace in _plotly_marks(m, maturities, sheet.z, lo, hi):
        fig.add_trace(trace)
    n_marked = int(clipped_mask(sheet.z, lo, hi).sum())
    marked = f" {n_marked} cells marked." if n_marked else ""
    own = ""
    if sheet.quantity == "local volatility":
        own = " Own vertical scale, local volatility, not implied vol."
    title = (
        f"{sheet.title}.{own}{marked} Standardised moneyness k/√T. Expiries under 7 days omitted."
    )
    if sheet.key == "market":
        title += " Points are raw quotes."
    m_lo, m_hi = display["m_view"]
    fig.update_layout(
        title={"text": title, "font": {"size": 12}},
        margin={"l": 0, "r": 0, "t": 72, "b": 0},
        legend={"x": 0.01, "y": 0.99},
        scene={
            "xaxis": {
                "title": "k / √T",
                "range": [float(m_lo), float(m_hi)],
                "autorange": False,
            },
            "yaxis": {
                "title": "maturity T",
                "range": [float(maturities[0]), float(maturities[-1])],
                "autorange": False,
            },
            "zaxis": {
                "title": sheet.quantity,
                "range": [float(lo), float(hi)],
                "autorange": False,
            },
            "camera": {"eye": PLOTLY_EYE},
            "aspectmode": "manual",
            "aspectratio": {"x": 1.6, "y": 1.15, "z": 0.65},
        },
    )
    return fig


def write_html(built: dict, html_dir: Path) -> None:
    """One self-contained HTML file per surface. Drag rotates the view."""
    html_dir.mkdir(parents=True, exist_ok=True)
    for sheet in built["display"]["sheets"]:
        fig = _plotly_surface(sheet, built)
        text = fig.to_html(
            full_html=True,
            include_plotlyjs=True,
            config={"responsive": True, "scrollZoom": True, "displaylogo": False},
        )
        text = text.replace(
            "<head>",
            '<head><meta name="viewport" content="width=device-width, initial-scale=1">',
            1,
        )
        (html_dir / f"{sheet.key}.html").write_text(text, encoding="utf-8")


def build_and_write(
    data: MarketData,
    *,
    fig_dir: Path = FIG,
    html_dir: Path = HTML,
    n_k: int = N_K,
    n_paths: int = N_PATHS,
    seed: int = SEED,
    heston_starts: int = 4,
    sabr_starts: int = 4,
) -> dict:
    """Sample the surfaces and write the PNG and HTML files. Returns the summary."""
    built = build_sheets(
        data,
        n_k=n_k,
        n_paths=n_paths,
        seed=seed,
        heston_starts=heston_starts,
        sabr_starts=sabr_starts,
    )
    write_pngs(built, fig_dir)
    write_html(built, html_dir)
    numbers = summary_numbers(built)
    (fig_dir / "p2_surface_gallery.json").write_text(
        json.dumps(numbers, indent=2) + "\n", encoding="utf-8"
    )
    return numbers


def main() -> None:
    chain = chains.load_snapshot(chains.latest_snapshot("SPX"))
    data = market_data_from_snapshot(chain)
    numbers = build_and_write(data)
    print(json.dumps(numbers, indent=2))


if __name__ == "__main__":
    main()
