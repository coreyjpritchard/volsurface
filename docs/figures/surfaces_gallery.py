"""Implied-vol gallery for the models on the committed SPX book.

One grid covers the quoted log-moneyness and the quoted expiries. Implied-vol
panels share a colour scale, a vertical range, and a view angle. That range
is the market's, plus a margin: ``[0.05, 0.90]``. A cell outside it is left
off the mesh and drawn as a cross at the near edge, so it cannot rescale the
axis. Faces outside that expiry's own quotes are faint, not a solid grey
sheet. Dupire local volatility keeps its own axis, ``[0.00, 2.80]``, and the
same rule: cells outside that window are marked and do not set the scale.

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
# Market quoted IV on this book is about 0.08 to 0.86. The axis is that
# range plus a margin. Cells outside it are marked and do not set the scale.
IV_LO = 0.05
IV_HI = 0.90
# Inside the quotes, Dupire local vol on this book runs to about 2.61.
# The axis stops short of the short-dated extrapolated spikes.
LOCAL_LO = 0.0
LOCAL_HI = 2.80
FAINT_FACE = (0.45, 0.45, 0.45, 0.22)
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
    numbers["display"] = {
        "implied_vol_axis": [IV_LO, IV_HI],
        "local_vol_axis": [LOCAL_LO, LOCAL_HI],
    }
    clips = {}
    for sheet in built["sheets"]:
        lo, hi = _window(sheet.quantity)
        clips[sheet.key] = clip_report(sheet.z, extrapolated, built["maturities"], k, lo, hi)
    numbers["clips"] = clips
    return numbers


def _window(quantity: str) -> tuple[float, float]:
    """Locked vertical range. Implied vol shares one window. Local vol has its own."""
    if quantity == "local volatility":
        return LOCAL_LO, LOCAL_HI
    return IV_LO, IV_HI


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


def _face_colours(z: np.ndarray, norm: Normalize | None, rgba: tuple | None) -> np.ndarray:
    """One RGBA colour per face. A face is drawn only when all four corners are finite."""
    stack = np.stack([z[:-1, :-1], z[:-1, 1:], z[1:, :-1], z[1:, 1:]], axis=0)
    ok = np.all(np.isfinite(stack), axis=0)
    total = np.sum(np.where(np.isfinite(stack), stack, 0.0), axis=0)
    face_z = np.zeros(ok.shape, dtype=float)
    np.divide(total, 4.0, out=face_z, where=ok)
    if rgba is None:
        colours = CMAP(norm(np.where(ok, face_z, norm.vmin)))
        colours[..., 3] = np.where(ok, 1.0, 0.0)
    else:
        colours = np.zeros(face_z.shape + (4,), dtype=float)
        colours[ok] = rgba
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


def _draw_surface(ax, k, maturities, sheet: Sheet, extrapolated, *, caption: bool) -> None:
    lo, hi = _window(sheet.quantity)
    finite = np.isfinite(sheet.z)
    in_window = finite & ~clipped_mask(sheet.z, lo, hi)
    quoted = layer_z(sheet.z, in_window & ~extrapolated)
    faint = layer_z(sheet.z, in_window & extrapolated)
    norm = Normalize(vmin=lo, vmax=hi)
    _plot_layer(ax, k, maturities, faint, _face_colours(faint, None, FAINT_FACE))
    _plot_layer(ax, k, maturities, quoted, _face_colours(quoted, norm, None))
    _mark_clipped(ax, k, maturities, sheet.z, lo, hi, size=14.0 if caption else 7.0)
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    ax.set_xlim(float(k[0]), float(k[-1]))
    ax.set_ylim(float(maturities[0]), float(maturities[-1]))
    ax.set_zlim(lo, hi)
    ax.set_xlabel("k")
    ax.set_ylabel("T")
    ax.set_zlabel(sheet.quantity)
    if caption:
        n_marked = int(clipped_mask(sheet.z, lo, hi).sum())
        ax.set_title(f"{sheet.title}\n[{lo:.2f}, {hi:.2f}], {n_marked} cells marked", fontsize=9)
    else:
        ax.set_title(sheet.title, fontsize=8)
    ax.tick_params(labelsize=7)


def write_pngs(built: dict, fig_dir: Path) -> None:
    """One PNG per surface, the seven-panel gallery, and the error heatmap."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    sheets = built["sheets"]
    k = built["k"]
    maturities = built["maturities"]
    extrapolated = built["extrapolated"]
    inside = ~extrapolated

    for sheet in sheets:
        fig = plt.figure(figsize=(7.4, 5.6))
        ax = fig.add_subplot(111, projection="3d")
        _draw_surface(ax, k, maturities, sheet, extrapolated, caption=True)
        lo, hi = _window(sheet.quantity)
        fig.colorbar(
            ScalarMappable(norm=Normalize(vmin=lo, vmax=hi), cmap=CMAP),
            ax=ax,
            fraction=0.046,
            pad=0.08,
            label=f"{sheet.quantity} on [{lo:.2f}, {hi:.2f}]",
        )
        fig.tight_layout()
        fig.savefig(fig_dir / f"p2_surface_{sheet.key}.png", dpi=140)
        plt.close(fig)

    fig = plt.figure(figsize=(16.5, 9.2))
    for i, sheet in enumerate(sheets):
        ax = fig.add_subplot(2, 4, i + 1, projection="3d")
        _draw_surface(ax, k, maturities, sheet, extrapolated, caption=False)
    fig.suptitle(
        f"SPX {built['date']}. Implied vol is drawn on [{IV_LO:.2f}, {IV_HI:.2f}]. "
        "Crosses mark cells outside that window. Faint faces are outside the quotes. "
        f"Dupire local vol is on its own axis [{LOCAL_LO:.2f}, {LOCAL_HI:.2f}].",
        fontsize=11,
    )
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.88, wspace=0.08, hspace=0.28)
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
        "Grey cells are outside that expiry's quotes. "
        "Gaps use the unclipped model.",
        fontsize=11,
    )
    fig.subplots_adjust(left=0.06, right=0.90, bottom=0.16, top=0.82, wspace=0.18)
    fig.savefig(fig_dir / "p2_surface_iv_error.png", dpi=140)
    plt.close(fig)


def _colourscale() -> list:
    scale = []
    for stop in (0.0, 0.25, 0.5, 0.75, 1.0):
        red, green, blue, _alpha = CMAP(stop)
        scale.append([stop, f"rgb({int(red * 255)},{int(green * 255)},{int(blue * 255)})"])
    return scale


def _plotly_marks(k, maturities, z: np.ndarray, extrapolated, lo: float, hi: float) -> list:
    """Marker traces at the window edge. Hover text carries the raw value."""
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
        where = np.where(extrapolated[rows, cols], "outside the quotes", "inside the quotes")
        traces.append(
            go.Scatter3d(
                x=k[cols],
                y=maturities[rows],
                z=np.full(rows.shape, height),
                mode="markers",
                marker={"size": 3, "color": colour, "symbol": "x"},
                name=name,
                text=[f"raw {value:.4f}, {place}" for value, place in zip(raw, where, strict=True)],
                hovertemplate=(
                    "k=%{x:.4f}<br>T=%{y:.5f}<br>marked at %{z:.2f}<br>%{text}<extra>"
                    + name
                    + "</extra>"
                ),
            )
        )
    return traces


def _plotly_surface(sheet: Sheet, built: dict) -> go.Figure:
    k = built["k"]
    maturities = built["maturities"]
    extrapolated = built["extrapolated"]
    lo, hi = _window(sheet.quantity)
    finite = np.isfinite(sheet.z)
    in_window = finite & ~clipped_mask(sheet.z, lo, hi)
    quoted = layer_z(sheet.z, in_window & ~extrapolated)
    faint = layer_z(sheet.z, in_window & extrapolated)
    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=k,
            y=maturities,
            z=quoted,
            cmin=lo,
            cmax=hi,
            colorscale=_colourscale(),
            colorbar={"title": sheet.quantity},
            name="quoted",
            showscale=True,
        )
    )
    if np.any(np.isfinite(faint)):
        fig.add_trace(
            go.Surface(
                x=k,
                y=maturities,
                z=faint,
                showscale=False,
                opacity=0.22,
                colorscale=[[0.0, "rgb(140,140,140)"], [1.0, "rgb(140,140,140)"]],
                name="outside the quotes",
                hovertemplate=(
                    "k=%{x:.4f}<br>T=%{y:.5f}<br>value %{z:.4f}"
                    "<br>outside the quotes<extra></extra>"
                ),
            )
        )
    for trace in _plotly_marks(k, maturities, sheet.z, extrapolated, lo, hi):
        fig.add_trace(trace)
    own = " Own axis." if sheet.quantity == "local volatility" else ""
    n_marked = int(clipped_mask(sheet.z, lo, hi).sum())
    fig.update_layout(
        title=(
            f"{sheet.title}.{own} Axis [{lo:.2f}, {hi:.2f}], {n_marked} cells marked. "
            "Faint faces are outside the quotes."
        ),
        margin={"l": 0, "r": 0, "t": 64, "b": 0},
        legend={"x": 0.01, "y": 0.99},
        scene={
            "xaxis": {
                "title": "log-moneyness k",
                "range": [float(k[0]), float(k[-1])],
                "autorange": False,
            },
            "yaxis": {
                "title": "maturity T",
                "range": [float(maturities[0]), float(maturities[-1])],
                "autorange": False,
            },
            "zaxis": {
                "title": sheet.quantity,
                "range": [lo, hi],
                "autorange": False,
            },
            "camera": {"eye": PLOTLY_EYE},
            "aspectmode": "manual",
            "aspectratio": {"x": 1.5, "y": 1.15, "z": 0.7},
        },
    )
    return fig


def write_html(built: dict, html_dir: Path) -> None:
    """One self-contained HTML file per surface. Drag rotates the view."""
    html_dir.mkdir(parents=True, exist_ok=True)
    for sheet in built["sheets"]:
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
