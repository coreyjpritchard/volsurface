"""Figure for P1b: three interpolators on one saved SPY expiry, and the spline wing.

The chain is ``data/snapshots/SPY_2026-10-07.parquet``: last trades, empty
bid and ask. The loader marks SPY American and physical. The expiry in the
top two panels is the last one in the file, chosen because the natural cubic,
continued past the lowest quoted strike, crosses through zero implied
volatility. It is not chosen because SVI passes there. The count of slices
with ``g < 0`` on ``k`` in ``[-0.5, 0.5]`` is computed in this script. The
bottom panel is the fitted slice with the smallest such ``g``.

Run from the repository root:

    python docs/figures/p1b_surfaces.py

Writes docs/figures/p1b_surfaces.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from volsurface import chains
from volsurface.arbitrage import check_surface
from volsurface.core.market import market_data_from_snapshot
from volsurface.core.types import ArbTol, Curves, Domain
from volsurface.fit.svi import SVIFitter, fit_raw_svi
from volsurface.surface.linear import linear_surface
from volsurface.surface.spline import spline_surface
from volsurface.surface.svi import SVISurface

OUT = Path(__file__).with_name("p1b_surfaces.png")
# Pointwise tolerance in normalised-call units. A butterfly this large is
# the wing or a kink, not solver dust.
TOL = ArbTol(price=1e-4)


def _one_expiry(quotes, expiry):
    """Quotes for a single expiry, one row per log-moneyness, sorted."""
    group = quotes[quotes["expiry"] == expiry].drop_duplicates("k").sort_values("k")
    if len(group) < 5:
        raise ValueError(f"{expiry} does not have five strikes")
    return group


def main() -> None:
    chain = chains.load_snapshot(chains.latest_snapshot("SPY"))
    data = market_data_from_snapshot(chain)
    # Last expiry in the file: few strikes, and the cubic wing goes negative.
    expiry = data.quotes.groupby("expiry", sort=True)["T"].first().index[-1]
    quotes = _one_expiry(data.quotes, expiry)
    T = float(quotes["T"].iloc[0])
    k_obs = quotes["k"].to_numpy(dtype=float)
    iv_obs = quotes["iv"].to_numpy(dtype=float)
    market = Curves(data.F, data.df)

    linear = linear_surface(quotes, market)
    spline = spline_surface(quotes, market)
    raw, residual = fit_raw_svi(k_obs, iv_obs**2 * T)
    svi = SVISurface([(T, raw)], market, Domain(T, T, float(k_obs[0]), float(k_obs[-1])))

    # Same objective as the fitter. SPY has no spreads, so the weights are 1.
    fitted = SVIFitter().fit(data)
    k_band = np.linspace(-0.5, 0.5, 401)
    n_slices = len(fitted.surface.pillars)
    n_fail = 0
    worst = None
    for maturity, slice_ in fitted.surface.pillars:
        g_band = np.asarray(slice_.g(k_band), dtype=float)
        g_min = float(np.nanmin(g_band))
        if g_min < 0.0:
            n_fail += 1
        if worst is None or g_min < worst[0]:
            worst = (g_min, maturity, g_band)
    this_g = float(np.nanmin(np.asarray(raw.g(k_band), dtype=float)))
    this_passes = this_g >= 0.0
    n_pass = n_slices - n_fail

    # A little past each end, so the cubic is no longer an interpolant.
    pad = 0.12
    wide = np.linspace(k_obs[0] - pad, k_obs[-1] + pad, 361)
    inside = (wide >= k_obs[0]) & (wide <= k_obs[-1])

    iv_spline = spline.iv([T], wide)[0]
    iv_svi = svi.iv([T], wide)[0]
    iv_linear = linear.iv([T], wide)[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        density_spline = spline.density([T], wide)[0]
        density_svi = svi.density([T], wide)[0]

    report = check_surface(spline, np.array([T]), wide, tol=TOL)
    butterfly = report.masks["butterfly"][0]
    extrapolated = spline.extrapolated([T], wide)[0]

    knot_gap = np.sqrt(np.mean((svi.iv([T], k_obs)[0] - iv_obs) ** 2))
    wing_min = float(np.min(iv_spline[extrapolated]))
    inside_neg = int(np.sum((density_spline < 0.0) & inside & np.isfinite(density_spline)))
    inside_n = int(np.sum(inside & np.isfinite(density_spline)))

    fig, axes = plt.subplots(3, 1, figsize=(7.4, 10.2), sharex=True)

    ax = axes[0]
    ax.axvspan(wide[0], k_obs[0], color="0.85", label="past the quotes")
    ax.axvspan(k_obs[-1], wide[-1], color="0.85")
    ax.plot(wide, 100 * iv_linear, color="C1", label="linear call, then inverted")
    ax.plot(wide, 100 * iv_spline, color="C0", label="cubic spline of implied vol")
    ax.plot(wide, 100 * iv_svi, color="C2", label="raw SVI, least squares")
    ax.plot(k_obs, 100 * iv_obs, "o", color="k", ms=3.5, label="SPY last trade")
    # Butterfly nodes, drawn on the spline. Marks are clipped to the axes so a
    # cubic that has left the frame still shows where the check fired.
    ax.plot(
        wide[butterfly],
        np.clip(100 * iv_spline[butterfly], -40, 80),
        "x",
        color="C3",
        ms=4,
        label="butterfly on the spline",
    )
    ax.axhline(0.0, color="0.4", lw=0.6)
    ax.set_ylim(-40, 55)
    ax.set_ylabel("implied volatility (%)")
    ax.set_title(
        f"Last expiry in the file, {expiry.date()}, T = {T:.2f}y, "
        f"chosen because the cubic wing crosses zero. Last trades, American."
    )
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.axvspan(wide[0], k_obs[0], color="0.85")
    ax.axvspan(k_obs[-1], wide[-1], color="0.85")
    # The spline density spikes by tens between noisy lasts. The panel keeps
    # the scale of a probability density and lets those spikes leave the frame.
    lo, hi = -2.0, 8.0
    ax.plot(wide, np.clip(density_spline, lo, hi), color="C0", label="spline density, clipped")
    ax.plot(wide, density_svi, color="C2", label="SVI density")
    ax.fill_between(
        wide,
        0.0,
        np.clip(density_spline, lo, 0.0),
        where=density_spline < 0.0,
        color="C3",
        alpha=0.35,
        label="spline density below zero",
    )
    ax.axhline(0.0, color="0.4", lw=0.6)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("log-moneyness k = log(K / F)")
    ax.set_ylabel("density of log(F_T / F)")
    if this_passes:
        svi_line = (
            f"SVI: g<0 on {n_fail}/{n_slices} slices for k in [-0.5, 0.5]; "
            f"this is one of the {n_pass} that pass."
        )
    else:
        svi_line = (
            f"SVI: g<0 on {n_fail}/{n_slices} slices for k in [-0.5, 0.5]; "
            "this plotted expiry fails too."
        )
    ax.set_title(svi_line + f" In-sample knot RMS {100 * knot_gap:.2f} vol points.")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    g_min, worst_T, g_band = worst
    ax.axhline(0.0, color="0.4", lw=0.6)
    ax.plot(k_band, g_band, color="C3", label="worst fitted slice")
    ax.plot(k_band, raw.g(k_band), color="C2", label="expiry in the panels above")
    ax.set_xlabel("log-moneyness k = log(K / F)")
    ax.set_ylabel("Durrleman g")
    ax.set_title(
        f"Worst slice on this grid: T = {worst_T:.4f}y, min g = {g_min:.3f}. "
        "The fit does not penalise g."
    )
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    print(f"wrote {OUT}")
    print(
        f"expiry {expiry.date()} T={T:.4f} n={len(k_obs)} "
        f"svi_iv_rms={knot_gap:.6f} w_rms={np.sqrt(np.mean(residual**2)):.6e} "
        f"spline_wing_iv_min={wing_min:.3f} "
        f"spline_density_negative_inside={inside_neg}/{inside_n} "
        f"butterfly_nodes={int(butterfly.sum())}/{wide.size} "
        f"svi_density_min={float(np.nanmin(density_svi)):.4f} "
        f"g_fail={n_fail}/{n_slices} this_g_min={this_g:.4f} "
        f"worst_T={worst_T:.4f} worst_g={g_min:.4f}"
    )


if __name__ == "__main__":
    main()
