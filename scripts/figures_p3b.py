"""Linear SANOS (eta = 0) on the committed SPX snapshot.

Run from the repository root:

    python scripts/figures_p3b.py

The left panel is the projected mass at four expiries. At eta = 0 the
density is atomic, so the mass is the density. The right panel is the
quote residual divided by the half-spread.
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

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
TARGETS = (7 / 365, 30 / 365, 90 / 365, 0.96)


def main() -> None:
    chain = chains.load_snapshot(chains.latest_snapshot("SPX"))
    data = market_data_from_snapshot(chain)
    result = SanosFitter(eta=0.0, omega=0.0, dx=0.05).fit(data)
    surface = result.surface
    FIG.mkdir(parents=True, exist_ok=True)
    result.slice_report.to_csv(FIG / "p3b_sanos_slices.csv", index=False)

    half = 0.5 * (result.detail["quote_ask"] - result.detail["quote_bid"])
    half = np.maximum(half, 1e-12)
    residual_spreads = result.residuals / half
    quotes = pd.DataFrame(
        {
            "T": result.detail["quote_T"],
            "x": result.detail["quote_x"],
            "k": np.log(result.detail["quote_x"]),
            "residual_spreads": residual_spreads,
        }
    )
    quotes.to_csv(FIG / "p3b_sanos_residuals.csv", index=False)

    chosen = []
    for target in TARGETS:
        j = int(np.argmin(np.abs(surface.T - target)))
        if j not in chosen:
            chosen.append(j)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    for j in chosen:
        label = f"T={surface.T[j]:.3f}"
        axes[0].plot(surface.strikes[j], surface.weights[j], lw=0.8, label=label)
    axes[0].set_xlabel("normalised strike")
    axes[0].set_ylabel("mass")
    axes[0].set_title("Projected mass, eta = 0")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(True, alpha=0.3)

    for j in chosen:
        mask = np.isclose(quotes["T"], surface.T[j])
        part = quotes[mask]
        axes[1].scatter(part["k"], part["residual_spreads"], s=6, label=f"T={surface.T[j]:.3f}")
    axes[1].axhline(0.0, color="0.4", lw=0.8)
    axes[1].axhline(1.0, color="0.6", lw=0.6, ls="--")
    axes[1].axhline(-1.0, color="0.6", lw=0.6, ls="--")
    axes[1].set_xlabel("log-moneyness")
    axes[1].set_ylabel("residual / half-spread")
    axes[1].set_title("Quote residual")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(
        f"Linear SANOS on SPX, objective {result.detail['objective']:.3e}, "
        f"in spread {result.in_spread:.3f}."
    )
    fig.tight_layout()
    fig.savefig(FIG / "p3b_sanos_eta0.png", dpi=140)
    plt.close(fig)

    summary = {
        "objective": result.detail["objective"],
        "in_spread": result.in_spread,
        "wall_time": result.wall_time,
        "projection_l1": result.detail["projection_l1"],
        "n_quotes": int(result.residuals.size),
        "n_expiries": int(surface.T.size),
        "k_min": result.detail["k_min"],
        "k_max": result.detail["k_max"],
        "dx": result.detail["dx"],
        "v_raised": result.detail["v_raised"],
        "solver_status": result.solver_status,
        "median_abs_residual_spreads": float(np.median(np.abs(residual_spreads))),
    }
    pd.Series(summary).to_csv(FIG / "p3b_sanos_summary.csv", header=False)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
