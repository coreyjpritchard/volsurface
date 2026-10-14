"""Discrete local volatility heatmap on the SPX SANOS surface.

Run from the repository root:

    python scripts/figures_p3d.py

Calls are read off the eta = 0.25, omega = 0 fit on one common strike
grid. Sigma between neighbouring expiries is the discrete Dupire ratio.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

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
    chain = chains.load_snapshot(chains.latest_snapshot("SPX"))
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

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    mid_t = 0.5 * (surface.T[:-1] + surface.T[1:])
    # The colour scale stops at the 99th percentile. The maximum is in the summary.
    ceiling = float(np.quantile(finite, 0.99))
    image = ax.pcolormesh(
        log_k,
        mid_t,
        np.clip(grid, 0.0, ceiling),
        shading="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=ceiling,
    )
    fig.colorbar(image, ax=ax, label="discrete local volatility")
    ax.set_xlabel("log-moneyness")
    ax.set_ylabel("maturity")
    ax.set_title(
        f"SANOS discrete local volatility on SPX. "
        f"Median {summary['sigma_median']:.3f}, max {summary['sigma_max']:.1f}."
    )
    fig.tight_layout()
    fig.savefig(FIG / "p3d_dlv.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
