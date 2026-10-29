"""Discrete local volatility heatmap on the SPX SANOS surface.

Run from the repository root:

    python scripts/figures_p3d.py

Calls are read off the eta = 0.25, omega = 0 fit on one common strike
grid. Sigma between neighbouring expiries is the discrete Dupire ratio.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.colors
import matplotlib.patches
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from volsurface import chains
from volsurface.core.market import market_data_from_snapshot
from volsurface.fit.sanos import SanosFitter
from volsurface.surface.sanos_dlv import implied_sigma

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"


def main() -> None:
    chain = chains.load_snapshot(chains.require_snapshot("SPX"))
    data = market_data_from_snapshot(chain)
    result = SanosFitter(eta=0.25, omega=0.0, dx=0.05, time_interp="pchip").fit(data)
    surface = result.surface
    log_k = np.linspace(-0.25, 0.2, 37)
    strikes = np.exp(log_k)
    rows = []
    sigmas = []
    for j in range(1, surface.T.size):
        dT = float(surface.T[j] - surface.T[j - 1])
        prev = surface.call([surface.T[j - 1]], log_k)[0]
        now = surface.call([surface.T[j]], log_k)[0]
        sigma = implied_sigma(strikes, prev, now, dT)
        sigmas.append(sigma)
        for k_i, value in zip(log_k, sigma, strict=True):
            rows.append(
                {
                    "T_left": float(surface.T[j - 1]),
                    "T_right": float(surface.T[j]),
                    "k": float(k_i),
                    "sigma": float(value),
                }
            )
    frame = pd.DataFrame(rows)
    FIG.mkdir(parents=True, exist_ok=True)
    frame.to_csv(FIG / "p3d_dlv.csv", index=False)
    grid = np.vstack(sigmas)
    finite = grid[np.isfinite(grid)]
    second = grid[:, 2:] - 2.0 * grid[:, 1:-1] + grid[:, :-2]
    rms = float(np.sqrt(np.mean(second**2)))
    summary = {
        "n_steps": int(grid.shape[0]),
        "n_strikes": int(grid.shape[1]),
        "sigma_min": float(finite.min()),
        "sigma_median": float(np.median(finite)),
        "sigma_max": float(finite.max()),
        "second_diff_rms": rms,
        "fit_objective": result.detail["objective"],
        "fit_in_spread": result.in_spread,
    }
    pd.Series(summary).to_csv(FIG / "p3d_dlv_summary.csv", header=False)
    for key, value in summary.items():
        print(f"{key}: {value}")

    # Where the extremes sit: beyond a month the surface is tame.
    month = surface.T[1:] > 30.0 / 365.0
    late = grid[month][np.isfinite(grid[month])]
    i_max, j_max = np.unravel_index(np.nanargmax(grid), grid.shape)
    print(
        f"max_cell T_left={surface.T[i_max]:.6f} T_right={surface.T[i_max + 1]:.6f} "
        f"k={log_k[j_max]:.4f}; beyond 30 days: max={late.max():.4f} "
        f"median={np.median(late):.4f}; zero cells={np.mean(finite == 0.0):.4f}"
    )
    zero_cols = np.all(grid == 0.0, axis=0)
    print(f"all-zero strike columns: {log_k[zero_cols].round(4).tolist()}")

    # Cells are drawn between the two expiries that define them, on a log time
    # axis in days, so the short end where the extremes sit is readable. The
    # colour scale is logarithmic and covers the full range, so nothing saturates.
    keep = ~zero_cols
    k_kept = log_k[keep]
    half = 0.5 * (k_kept[1] - k_kept[0])
    k_edges = np.concatenate([k_kept - half, [k_kept[-1] + half]])
    t_edges = 365.0 * surface.T
    shown = grid[:, keep]
    positive = shown[np.isfinite(shown) & (shown > 0.0)]
    floor = float(np.quantile(positive, 0.001))
    cmap = matplotlib.colormaps["viridis"].copy()
    cmap.set_bad("0.75")
    masked = np.ma.masked_where(~np.isfinite(shown) | (shown <= 0.0), shown)
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    image = ax.pcolormesh(
        k_edges,
        t_edges,
        masked,
        shading="flat",
        cmap=cmap,
        norm=matplotlib.colors.LogNorm(vmin=floor, vmax=summary["sigma_max"]),
    )
    fig.colorbar(image, ax=ax, label="discrete local volatility (annualised, log scale)")
    ax.set_yscale("log")
    ax.set_yticks([0.5, 1, 2, 5, 10, 30, 100, 350])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("log-moneyness k = log(K / F)")
    ax.set_ylabel("time to expiry (days, log scale)")
    ax.legend(
        handles=[matplotlib.patches.Patch(color="0.75", label="zero: no local vol")],
        loc="upper left",
        fontsize=8,
        frameon=True,
    )
    fig.suptitle(
        f"SANOS discrete local vol on SPX is tame beyond a month (median {np.median(late):.2f}, "
        f"max {late.max():.2f}),\nbut reaches {summary['sigma_max']:.0f} in the wings "
        f"of the first few days; median over all cells {summary['sigma_median']:.2f}",
        fontsize=10.5,
    )
    fig.text(
        0.01,
        0.005,
        "SPXW mids, CBOE delayed, 14 Oct 2026. SANOS fit, eta = 0.25, omega = 0, PCHIP in time; "
        "one cell per pair of neighbouring expiries.",
        fontsize=7.5,
        color="0.3",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.subplots_adjust(top=0.88)
    fig.savefig(FIG / "p3d_dlv.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
