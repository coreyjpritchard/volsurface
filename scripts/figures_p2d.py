"""Figures for Dupire local variance on the SPX snapshot and on Heston.

Run from the repository root:

    python scripts/figures_p2d.py

Writes PNG and CSV under docs/figures/, and data/presets/localvol.json.
The snapshot is SPXW mids. Local variance outside the quoted domain is
kept in the wing panel and marked extrapolated. The default dupire call
blanks that wing.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from volsurface import bs, chains
from volsurface.core.market import market_data_from_snapshot
from volsurface.core.types import Curves, Domain
from volsurface.fit.svi import SVIFitter
from volsurface.models.heston import Heston
from volsurface.models.localvol import EXPLODE, dupire
from volsurface.surface.linear import linear_surface
from volsurface.surface.model import ModelSurface
from volsurface.surface.spline import spline_surface
from volsurface.surface_export import write_surface_preset

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"


def _flat(level):
    return lambda t: np.full(np.shape(t), level, dtype=float)


def snapshot_panels(data) -> dict:
    """Linear, spline and SVI local variance of the committed SPX chain.

    The expiry is the one nearest 23 calendar days. The vertical axis is
    symmetric-log so a spike and the at-the-money level share a panel.
    """
    quotes = data.quotes.dropna(subset=["iv", "price", "k"])
    market = Curves(data.F, data.df)
    linear = linear_surface(quotes, market)
    spline = spline_surface(quotes, market)
    svi = SVIFitter().fit(data).surface
    expiries = np.sort(quotes.groupby("expiry")["T"].first().to_numpy())
    target = float(expiries[np.argmin(np.abs(expiries - 23.0 / 365.0))])
    pillar = min(spline.pillars, key=lambda item: abs(item["T"] - target))
    k_lo = float(pillar["k"][0])
    k_hi = float(pillar["k"][-1])
    # Past the last quoted call strike, where the cubic wing is the point.
    k = np.linspace(-0.55, 0.30, 71)
    rows = []
    panels = {}
    for name, surface in (("linear", linear), ("spline", spline), ("svi", svi)):
        inside = dupire(surface, np.array([target]), k, mask_outside=False)
        panels[name] = inside
        for j, log_k in enumerate(k):
            rows.append(
                {
                    "surface": name,
                    "T": target,
                    "k": float(log_k),
                    "variance": float(inside.variance[0, j]),
                    "denominator": float(inside.denominator[0, j]),
                    "extrapolated": bool(inside.extrapolated[0, j]),
                    "bad_denominator": bool(inside.bad_denominator[0, j]),
                    "exploding": bool(inside.exploding[0, j]),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(FIG / "p2d_snapshot_localvol.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.4))
    peaks = {}
    for ax, name in zip(axes, ("linear", "spline", "svi"), strict=True):
        block = frame[frame["surface"] == name]
        quoted = block[~block["extrapolated"]]
        wing = block[block["extrapolated"]]
        ax.axvspan(k[0], k_lo, color="0.92")
        ax.axvspan(k_hi, k[-1], color="0.92", label="past this expiry's quoted strikes")
        ax.plot(
            quoted["k"],
            quoted["variance"],
            color="C0",
            lw=1.2,
            label="inside the surface's domain",
        )
        if len(wing):
            ax.plot(
                wing["k"],
                wing["variance"],
                color="C0",
                lw=1.0,
                ls="--",
                label="outside the surface's domain",
            )
        ax.axhline(0.0, color="k", lw=0.6)
        ax.set_yscale("symlog", linthresh=0.05)
        _symlog_ticks(ax, block["variance"].to_numpy(dtype=float))
        peak = _max_abs_frame(block)
        peaks[name] = peak
        ax.set_title(f"{name}: largest |local variance| {_sci(peak)}", fontsize=10)
        ax.set_xlabel("log-moneyness k = log(K / F)")
        ax.set_xlim(k[0], k[-1])
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("local variance (per year, symmetric log scale)")
    axes[2].legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.suptitle(
        "Dupire local variance needs a smooth, convex surface. On SPX at "
        f"{365 * target:.0f} days it reaches\n{_sci(peaks['linear'])} in size from linear "
        f"interpolation and {_sci(peaks['spline'])} from a cubic spline; raw SVI peaks at "
        f"{_sci(peaks['svi'])}\nbut turns negative past the last strike, and negative "
        "local variance has no local vol",
        fontsize=10.5,
    )
    fig.text(
        0.01,
        0.005,
        f"SPXW mids, CBOE delayed, 14 Oct 2026; expiry nearest 23 days (T = {target:.3f}). "
        "Local variance from the Dupire formula in total variance, quoted domain not masked.",
        fontsize=7.5,
        color="0.3",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.subplots_adjust(top=0.78)
    fig.savefig(FIG / "p2d_snapshot_localvol.png", dpi=140)
    plt.close(fig)

    # dw/dT is constant between expiries. One sample in each open interval
    # is that constant. The plot steps at the expiry.
    mids = 0.5 * (expiries[:-1] + expiries[1:])
    k_atm = np.array([0.0])
    cal_rows = []
    cal = {}
    for name, surface in (("spline", spline), ("svi", svi)):
        report = dupire(surface, mids, k_atm, mask_outside=True)
        cal[name] = report.variance[:, 0]
        for i, t in enumerate(mids):
            cal_rows.append(
                {
                    "surface": name,
                    "T_left": float(expiries[i]),
                    "T_right": float(expiries[i + 1]),
                    "T_mid": float(t),
                    "k": 0.0,
                    "variance": float(report.variance[i, 0]),
                }
            )
    days = 365.0 * expiries
    fig, axes = plt.subplots(2, 1, figsize=(7.6, 6.4), sharex=True)
    _step(axes[0], days, cal["svi"], "C0", "raw SVI, one slice per expiry")
    _step(axes[1], days, cal["spline"], "C1", "cubic spline of implied vol")
    for ax in axes:
        for expiry in days:
            ax.axvline(float(expiry), color="k", lw=0.3, alpha=0.3)
        ax.axhline(0.0, color="k", lw=0.6)
        ax.set_xscale("log")
        ax.set_xticks([0.5, 1, 2, 5, 10, 30, 100, 350])
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("ATM local variance (per year)")
    axes[0].plot([], [], color="k", lw=0.3, alpha=0.5, label="quoted expiry")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    axes[1].set_xlabel("time to expiry (days, log scale)")
    axes[1].set_xlim(days[0], days[-1])
    svi_jump = _largest_step(cal["svi"])
    spline_jump = _largest_step(cal["spline"])
    negative = np.flatnonzero(cal["spline"] < 0.0)
    axes[0].set_title(f"Raw SVI: largest jump at an expiry {svi_jump:.4f}", fontsize=10)
    neg_text = (
        f", negative from {days[negative[0]]:.0f} to {days[negative[-1] + 1]:.0f} days"
        if negative.size
        else ""
    )
    axes[1].set_title(f"Spline: largest jump {spline_jump:.3f}{neg_text}", fontsize=10)
    fig.suptitle(
        "At the money, local variance is constant between expiries and jumps at each one,\n"
        "because total variance is linear in T between slices; the spline's noisy strike\n"
        "derivatives make it jump further and go negative",
        fontsize=10.5,
    )
    fig.text(
        0.01,
        0.005,
        "SPXW mids, CBOE delayed, 14 Oct 2026; k = 0, one sample in each gap between "
        "quoted expiries.",
        fontsize=7.5,
        color="0.3",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.subplots_adjust(top=0.84)
    fig.savefig(FIG / "p2d_calendar_jumps.png", dpi=140)
    plt.close(fig)
    pd.DataFrame(cal_rows).to_csv(FIG / "p2d_calendar_jumps.csv", index=False)

    summary = {
        "T": target,
        "quoted_k": [k_lo, k_hi],
        "explode_cap": EXPLODE,
        "spline_n_negative_inside": int(
            (panels["spline"].bad_denominator & ~panels["spline"].extrapolated).sum()
        ),
        "spline_n_exploding": int(panels["spline"].n_exploding),
        "spline_max_abs_outside": _max_abs_mask(panels["spline"], outside=True),
        "spline_max_abs_inside": _max_abs(panels["spline"]),
        "svi_n_negative": int(panels["svi"].n_negative),
        "svi_n_exploding": int(panels["svi"].n_exploding),
        "svi_max_abs_inside": _max_abs(panels["svi"]),
        "svi_max_abs_outside": _max_abs_mask(panels["svi"], outside=True),
        "linear_n_negative": int(panels["linear"].n_negative),
        "linear_n_exploding": int(panels["linear"].n_exploding),
        "linear_max_abs": _max_abs_frame(frame[frame["surface"] == "linear"]),
        "svi_calendar_min": float(np.nanmin(cal["svi"])),
        "svi_calendar_max": float(np.nanmax(cal["svi"])),
        "svi_largest_step": svi_jump,
        "spline_calendar_min": float(np.nanmin(cal["spline"])),
        "spline_calendar_max": float(np.nanmax(cal["spline"])),
        "spline_largest_step": spline_jump,
    }
    return summary


def _max_abs(report) -> float:
    values = report.variance[~report.extrapolated]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.max(np.abs(values)))


def _max_abs_mask(report, *, outside: bool) -> float:
    mask = report.extrapolated if outside else ~report.extrapolated
    values = report.variance[mask]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.max(np.abs(values)))


def _max_abs_frame(block: pd.DataFrame) -> float:
    values = block["variance"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.max(np.abs(values)))


def _step(ax, expiries: np.ndarray, levels: np.ndarray, color: str, label: str) -> None:
    """Draw a level that is constant on each ``[expiries[i], expiries[i+1]]``."""
    edges = np.repeat(expiries, 2)[1:-1]
    heights = np.repeat(levels, 2)
    ax.plot(edges, heights, color=color, lw=1.1, label=label)


def _sci(x: float) -> str:
    """A magnitude for a title: plain below 1,000, otherwise mathtext powers of ten."""
    if not np.isfinite(x):
        return "nan"
    if abs(x) < 1000:
        return f"{x:.3g}"
    exponent = int(np.floor(np.log10(abs(x))))
    return f"${x / 10**exponent:.1f} \\times 10^{{{exponent}}}$"


def _symlog_ticks(ax, values: np.ndarray) -> None:
    """Powers of ten every other decade on a symlog axis, and one zero.

    The default locator labels every decade on both sides, and the labels
    either side of zero collide.
    """
    finite = values[np.isfinite(values)]
    top = max(float(finite.max()), 0.1)
    bottom = min(float(finite.min()), 0.0)
    up = int(np.ceil(np.log10(top)))
    ticks = [0.0]
    step = 1 if up <= 2 else 2
    ticks += [10.0**e for e in range(up, -2, -step) if 10.0**e >= 0.1]
    if bottom < -0.1:
        down = int(np.ceil(np.log10(-bottom)))
        step = 1 if down <= 2 else 2
        ticks += [-(10.0**e) for e in range(down, -2, -step) if 10.0**e >= 0.1]
    ax.set_yticks(sorted(ticks))
    ax.set_ylim(1.3 * bottom if bottom < 0 else -0.05, 1.3 * top)


def _largest_step(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return float("nan")
    gaps = np.diff(values)
    gaps = gaps[np.isfinite(gaps)]
    if gaps.size == 0:
        return float("nan")
    return float(np.max(np.abs(gaps)))


def heston_panel() -> dict:
    """Local variance of a Heston surface against the implied vol it came from."""
    heston = Heston(v0=0.04, kappa=1.5, theta=0.04, xi=0.3, rho=-0.5)
    market = Curves(F=_flat(100.0), df=_flat(1.0))
    surface = ModelSurface(heston, market, Domain(0.05, 1.5, -0.6, 0.6))
    k = np.linspace(-0.4, 0.4, 41)
    maturity = np.array([1.0])
    report = dupire(surface, maturity, k)
    cash = heston.price(100.0, 100.0 * np.exp(k), 1.0)
    iv = bs.implied_vol(cash, 100.0, 100.0 * np.exp(k), 1.0, 1.0)
    frame = pd.DataFrame(
        {
            "k": k,
            "local_variance": report.variance[0],
            "implied_vol": iv,
            "bad_denominator": report.bad_denominator[0],
        }
    )
    frame.to_csv(FIG / "p2d_heston_localvol.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    axes[0].plot(k, 100 * iv, color="C0")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("implied volatility (%)")
    axes[0].set_title("Heston implied vol, T = 1")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(k, report.variance[0], color="C1")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("local variance")
    axes[1].set_title("Dupire local variance of that surface")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "p2d_heston_localvol.png", dpi=140)
    plt.close(fig)

    # Preset: the implied surface, plus local variance on the same strikes.
    call = cash / 100.0
    write_surface_preset(
        ROOT / "data" / "presets" / "localvol.json",
        model="localvol",
        params={
            "v0": heston.v0,
            "kappa": heston.kappa,
            "theta": heston.theta,
            "xi": heston.xi,
            "rho": heston.rho,
        },
        T=maturity,
        k=k,
        call=call.reshape(1, -1),
        iv=iv.reshape(1, -1),
        forward=100.0,
        discount=1.0,
        note=(
            "Heston surface and its Dupire local variance at T=1. "
            "Local variance is not a second smile. It is (dw/dT)/g."
        ),
    )
    # The helper does not carry the variance grid. Append it, still under 200 KB.
    path = ROOT / "data" / "presets" / "localvol.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["local_variance"] = [float(v) if np.isfinite(v) else None for v in report.variance[0]]
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {
        "n_negative": int(report.n_negative),
        "n_exploding": int(report.n_exploding),
        "local_variance_atm": float(report.variance[0, len(k) // 2]),
        "implied_vol_atm": float(iv[len(k) // 2]),
        "preset_bytes": path.stat().st_size,
    }


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    chain = chains.load_snapshot(chains.latest_snapshot("SPX"))
    data = market_data_from_snapshot(chain)
    summary = snapshot_panels(data)
    heston = heston_panel()
    print(json.dumps({"snapshot": summary, "heston": heston}, indent=2))


if __name__ == "__main__":
    main()
