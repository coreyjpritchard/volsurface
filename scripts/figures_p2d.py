"""Figures for Dupire local variance on the SPY snapshot and on Heston.

Run from the repository root:

    python scripts/figures_p2d.py

Writes PNG and CSV under docs/figures/, and data/presets/localvol.json.
The snapshot is last trades. Local variance outside the quoted domain is
kept in the wing panel and marked extrapolated. The default dupire call
blanks that wing.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

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
    """Linear, spline and SVI local variance of the committed SPY chain.

    The expiry is the one nearest 23 calendar days. On that slice the spline
    of implied vol leaves the quotes near k = 0.08 and the local variance
    in the call wing runs to hundreds of thousands. The slice nearest 90
    days does not: its spline stays below 1 on the same strike window, so
    it hides the wing the figure is there to show. The vertical axis is
    symmetric-log so that wing and the at-the-money level share a panel.
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

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.8), sharey=True)
    for ax, name in zip(axes, ("linear", "spline", "svi"), strict=True):
        block = frame[frame["surface"] == name]
        quoted = block[~block["extrapolated"]]
        wing = block[block["extrapolated"]]
        ax.plot(quoted["k"], quoted["variance"], color="C0", lw=1.2, label="inside quotes")
        if len(wing):
            ax.plot(
                wing["k"], wing["variance"], color="C0", lw=1.0, ls="--", label="extrapolated"
            )
        ax.axhline(0.0, color="k", lw=0.6)
        ax.axvline(k_lo, color="0.5", lw=0.5)
        ax.axvline(k_hi, color="0.5", lw=0.5)
        ax.set_yscale("symlog", linthresh=0.05)
        peak = _max_abs_frame(block)
        ax.set_title(f"{name}\nmax |var| = {peak:.3g}")
        ax.set_xlabel("k")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("local variance")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(
        f"SPY at T = {target:.3f}. Grey lines are this expiry's quoted k. "
        "Dashed is each surface's extrapolated mask."
    )
    fig.tight_layout()
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
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 5.6), sharex=True)
    _step(axes[0], expiries, cal["svi"], "C0")
    _step(axes[1], expiries, cal["spline"], "C1")
    for ax in axes:
        for expiry in expiries:
            ax.axvline(float(expiry), color="k", lw=0.3, alpha=0.35)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("SVI local variance")
    axes[1].set_ylabel("spline local variance")
    axes[1].set_xlabel("maturity T")
    svi_jump = _largest_step(cal["svi"])
    spline_jump = _largest_step(cal["spline"])
    axes[0].set_title(
        "k = 0. Total variance is linear in T, so local variance is "
        f"constant between expiries. Largest SVI jump: {svi_jump:.4f}."
    )
    axes[1].set_title(f"Same slice, spline of implied vol. Largest jump: {spline_jump:.3f}.")
    fig.tight_layout()
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


def _step(ax, expiries: np.ndarray, levels: np.ndarray, color: str) -> None:
    """Draw a level that is constant on each ``[expiries[i], expiries[i+1]]``."""
    edges = np.repeat(expiries, 2)[1:-1]
    heights = np.repeat(levels, 2)
    ax.plot(edges, heights, color=color, lw=1.1)


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
    chain = chains.load_snapshot(chains.latest_snapshot("SPY"))
    data = market_data_from_snapshot(chain)
    summary = snapshot_panels(data)
    heston = heston_panel()
    print(json.dumps({"snapshot": summary, "heston": heston}, indent=2))


if __name__ == "__main__":
    main()
