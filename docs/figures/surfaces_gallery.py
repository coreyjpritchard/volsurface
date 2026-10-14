"""Implied-vol gallery for the models on the committed SPX book.

One grid covers the quoted log-moneyness and the quoted expiries. A face is
grey when that expiry does not quote the strike: the value there is an
extrapolation. The six implied-vol panels share a colour scale, a vertical
range, and a view angle. The seventh panel is Dupire local volatility from
the raw SVI fit. It uses the same view and the same colour scale, and its
axis says local volatility, not implied volatility.

Heston and SABR are fit on the smile, ``|k| <= 0.4`` and at most 11 strikes
an expiry, the same cut as the calibration figures. Raw SVI uses every
quoted strike. The flat Black-Scholes volatility is the median of the
at-the-money implied vol across expiries. The Monte Carlo panel is exact
geometric Brownian motion at that volatility, one seed. A node is left
blank when the price does not clear two standard errors, which is a wing
no path reached. The scatter that remains is the simulation noise.

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
from matplotlib.colors import Normalize

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
VIEW_ELEV = 25.0
VIEW_AZIM = -60.0
PLOTLY_EYE = {"x": 1.55, "y": -1.65, "z": 0.85}
CMAP = plt.get_cmap("viridis")
N_K = 41
N_PATHS = 8_000
SEED = 2
# The smile cut used by the Heston and SABR calibration figures.
SMILE_ABS_K = 0.4
SMILE_STRIKES = 11


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


def gbm_iv(
    data: MarketData,
    maturities: np.ndarray,
    k: np.ndarray,
    sigma: float,
    *,
    n_paths: int,
    seed: int,
) -> np.ndarray:
    """Exact GBM at ``sigma``, one set of paths, inverted to implied vol.

    Each expiry uses its own forward. Strikes left of the money are puts.
    A price with no implied vol, which is a wing no path reached, is NaN.
    """
    model = GBM(sigma=sigma, scheme="exact", antithetic=False, control_variate=False)
    times = np.concatenate(([0.0], maturities))
    paths = model.simulate(n_paths, times, seed=seed)
    out = np.empty((maturities.size, k.size), dtype=float)
    for i, maturity in enumerate(maturities):
        forward = float(np.asarray(data.F(maturity), dtype=float))
        discount = float(np.asarray(data.df(maturity), dtype=float))
        strike = forward * np.exp(k)
        # Out of the money: a put when k < 0, a call when k >= 0.
        kind = np.where(k < 0.0, "put", "call")
        value, stderr = estimate_terminal(
            paths.Z[:, i + 1],
            forward,
            strike,
            discount,
            kind,
            antithetic=False,
            control_variate=False,
        )
        iv = _otm_iv_row(value, forward, k, float(maturity), discount)
        # A wing price that is one path in thousands inverts to a nonsense vol.
        # Keep a node only when the price clears two standard errors.
        out[i] = np.where(value > 2.0 * stderr, iv, np.nan)
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

    gbm = gbm_iv(data, maturities, k, sigma, n_paths=n_paths, seed=seed)
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
    return numbers


def _colour_limits(market: np.ndarray, inside: np.ndarray) -> tuple[float, float]:
    finite = market[inside & np.isfinite(market)]
    if finite.size == 0:
        return 0.05, 0.50
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if hi <= lo:
        hi = lo + 1e-3
    return lo, hi


def _face_colours(z: np.ndarray, extrapolated: np.ndarray, norm: Normalize) -> np.ndarray:
    """One RGBA colour per surface face. Grey faces sit outside the quotes."""
    corners = np.stack(
        [
            extrapolated[:-1, :-1],
            extrapolated[:-1, 1:],
            extrapolated[1:, :-1],
            extrapolated[1:, 1:],
        ],
        axis=0,
    )
    face_out = corners.sum(axis=0) >= 3
    stack = np.stack([z[:-1, :-1], z[:-1, 1:], z[1:, :-1], z[1:, 1:]], axis=0)
    count = np.sum(np.isfinite(stack), axis=0)
    total = np.nansum(np.where(np.isfinite(stack), stack, 0.0), axis=0)
    face_z = np.full(count.shape, np.nan, dtype=float)
    np.divide(total, count, out=face_z, where=count > 0)
    colours = CMAP(norm(face_z))
    colours[face_out] = (0.75, 0.75, 0.75, 1.0)
    colours[~np.isfinite(face_z)] = (1.0, 1.0, 1.0, 0.0)
    return colours


def _draw_surface(ax, k, maturities, sheet: Sheet, extrapolated, norm, zlim) -> None:
    kk, tt = np.meshgrid(k, maturities)
    faces = _face_colours(sheet.z, extrapolated, norm)
    ax.plot_surface(
        kk,
        tt,
        np.where(np.isfinite(sheet.z), sheet.z, np.nan),
        facecolors=faces,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False,
    )
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    ax.set_zlim(*zlim)
    ax.set_xlabel("k")
    ax.set_ylabel("T")
    ax.set_zlabel(sheet.quantity)
    ax.set_title(sheet.title, fontsize=9)
    ax.tick_params(labelsize=7)


def write_pngs(built: dict, fig_dir: Path) -> None:
    """One PNG per surface, the seven-panel gallery, and the error heatmap."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    sheets = built["sheets"]
    k = built["k"]
    maturities = built["maturities"]
    extrapolated = built["extrapolated"]
    market = next(sheet for sheet in sheets if sheet.key == "market").z
    inside = ~extrapolated
    vmin, vmax = _colour_limits(market, inside)
    norm = Normalize(vmin=vmin, vmax=vmax)
    iv_zlim = (vmin, vmax)
    local = next(sheet for sheet in sheets if sheet.key == "dupire_localvol")
    local_finite = local.z[np.isfinite(local.z)]
    if local_finite.size:
        local_zlim = (float(np.min(local_finite)), float(np.max(local_finite)))
        if local_zlim[1] <= local_zlim[0]:
            local_zlim = iv_zlim
    else:
        local_zlim = iv_zlim

    for sheet in sheets:
        fig = plt.figure(figsize=(7.4, 5.6))
        ax = fig.add_subplot(111, projection="3d")
        zlim = local_zlim if sheet.quantity == "local volatility" else iv_zlim
        _draw_surface(ax, k, maturities, sheet, extrapolated, norm, zlim)
        fig.colorbar(
            ScalarMappable(norm=norm, cmap=CMAP),
            ax=ax,
            fraction=0.046,
            pad=0.08,
            label="volatility, shared scale",
        )
        fig.tight_layout()
        fig.savefig(fig_dir / f"p2_surface_{sheet.key}.png", dpi=140)
        plt.close(fig)

    fig = plt.figure(figsize=(16.5, 9.2))
    for i, sheet in enumerate(sheets):
        ax = fig.add_subplot(2, 4, i + 1, projection="3d")
        zlim = local_zlim if sheet.quantity == "local volatility" else iv_zlim
        _draw_surface(ax, k, maturities, sheet, extrapolated, norm, zlim)
    fig.suptitle(
        f"SPX {built['date']}. Grey faces are outside that expiry's quotes. "
        "Colour scale is the market implied-vol range.",
        fontsize=11,
    )
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.90, wspace=0.08, hspace=0.22)
    fig.savefig(fig_dir / "p2_surface_gallery.png", dpi=130)
    plt.close(fig)

    _write_error_heatmap(built, fig_dir, inside)


def _write_error_heatmap(built: dict, fig_dir: Path, inside: np.ndarray) -> None:
    sheets = {sheet.key: sheet for sheet in built["sheets"]}
    market = sheets["market"].z
    keys = ("svi", "heston", "sabr")
    gaps = []
    for key in keys:
        gap = 100.0 * (sheets[key].z - market)
        gap = np.ma.array(gap, mask=~inside | ~np.isfinite(gap))
        gaps.append(gap)
    finite = np.concatenate([gap.compressed() for gap in gaps])
    limit = 1.0 if finite.size == 0 else float(np.max(np.abs(finite)))
    if limit == 0.0:
        limit = 1.0
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.6), sharey=True)
    titles = {
        "svi": "SVI minus market",
        "heston": "Heston minus market",
        "sabr": "SABR minus market",
    }
    k = built["k"]
    maturities = built["maturities"]
    for ax, key, gap in zip(axes, keys, gaps, strict=True):
        ax.set_facecolor("0.82")
        mesh = ax.pcolormesh(
            k,
            maturities,
            gap,
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
            shading="nearest",
        )
        ax.set_xlabel("log-moneyness k")
        ax.set_title(titles[key])
    axes[0].set_ylabel("maturity T (year fraction)")
    fig.colorbar(mesh, ax=axes, fraction=0.046, pad=0.04, label="vol points")
    fig.suptitle(
        f"Model minus SPX implied vol, {built['date']}. "
        "Grey cells are outside that expiry's quotes.",
        fontsize=11,
    )
    fig.subplots_adjust(left=0.06, right=0.90, bottom=0.16, top=0.82, wspace=0.18)
    fig.savefig(fig_dir / "p2_surface_iv_error.png", dpi=140)
    plt.close(fig)


def _plotly_surface(sheet: Sheet, built: dict, norm: Normalize) -> go.Figure:
    k = built["k"]
    maturities = built["maturities"]
    extrapolated = built["extrapolated"]
    quoted = np.where(extrapolated, np.nan, sheet.z)
    outside = np.where(extrapolated, sheet.z, np.nan)
    colourscale = []
    for stop in (0.0, 0.25, 0.5, 0.75, 1.0):
        red, green, blue, _alpha = CMAP(stop)
        colourscale.append([stop, f"rgb({int(red * 255)},{int(green * 255)},{int(blue * 255)})"])
    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=k,
            y=maturities,
            z=quoted,
            cmin=float(norm.vmin),
            cmax=float(norm.vmax),
            colorscale=colourscale,
            colorbar={"title": "volatility"},
            name="quoted",
            showscale=True,
        )
    )
    fig.add_trace(
        go.Surface(
            x=k,
            y=maturities,
            z=outside,
            showscale=False,
            colorscale=[[0.0, "rgb(191,191,191)"], [1.0, "rgb(191,191,191)"]],
            name="outside the quotes",
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        title=sheet.title + ". Grey is outside that expiry's quotes.",
        margin={"l": 0, "r": 0, "t": 48, "b": 0},
        scene={
            "xaxis_title": "log-moneyness k",
            "yaxis_title": "maturity T",
            "zaxis_title": sheet.quantity,
            "camera": {"eye": PLOTLY_EYE},
            "aspectmode": "manual",
            "aspectratio": {"x": 1.5, "y": 1.15, "z": 0.7},
        },
    )
    return fig


def write_html(built: dict, html_dir: Path) -> None:
    """One self-contained HTML file per surface. Drag rotates the view."""
    html_dir.mkdir(parents=True, exist_ok=True)
    market = next(sheet for sheet in built["sheets"] if sheet.key == "market").z
    vmin, vmax = _colour_limits(market, ~built["extrapolated"])
    norm = Normalize(vmin=vmin, vmax=vmax)
    for sheet in built["sheets"]:
        fig = _plotly_surface(sheet, built, norm)
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
