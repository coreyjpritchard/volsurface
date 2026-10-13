"""Figure for P1c: one geometric Brownian motion, inverted to implied volatility.

Eight thousand paths, one seed, three expiries, shared. The model is exact
Black dynamics with volatility 0.20, so the Black implied volatility is a
horizontal line. The Monte Carlo curves are one draw of that line.

Run from the repository root:

    python docs/figures/p1c_gbm.py

Writes docs/figures/p1c_gbm.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from volsurface import bs
from volsurface.mc.grid import price_grid
from volsurface.models.gbm import GBM

OUT = Path(__file__).with_name("p1c_gbm.png")


def main() -> None:
    F, sigma, df = 100.0, 0.2, 1.0
    # One week, one quarter, one year. The week is where a wing runs out of paths.
    maturities = np.array([7.0 / 365.0, 0.25, 1.0])
    labels = ["1 week", "3 months", "1 year"]
    k = np.linspace(-0.10, 0.10, 21)
    K = F * np.exp(k)
    model = GBM(sigma=sigma, scheme="exact", antithetic=False, control_variate=False)
    result = price_grid(model, F, K, maturities, df, "call", n_paths=8_000, seed=2)

    fig, axes = plt.subplots(2, 1, figsize=(7.4, 7.0), sharex=True)
    ax = axes[0]
    ax.axhline(100 * sigma, color="k", lw=1.0, label="Black, 20%")
    missing = 0
    for i, (t, label) in enumerate(zip(maturities, labels, strict=True)):
        iv = bs.implied_vol(result.value[i], F, K, t, df, "call")
        missing += int(np.isnan(iv).sum())
        ax.plot(k, 100 * iv, marker="o", ms=3.5, label=f"Monte Carlo, {label}")
    ax.set_ylabel("implied volatility (%)")
    ax.set_title("One simulation of geometric Brownian motion. Volatility is 20%.")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(14, 26)

    ax = axes[1]
    for i, (t, label) in enumerate(zip(maturities, labels, strict=True)):
        iv = bs.implied_vol(result.value[i], F, K, t, df, "call")
        ax.plot(k, 100 * (iv - sigma), marker="o", ms=3.5, label=label)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("log-moneyness k = log(K / F)")
    ax.set_ylabel("Monte Carlo minus Black (vol points)")
    ax.set_title("Shared paths: a long expiry shifts as a whole. A short wing can disappear.")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    print(f"wrote {OUT}")
    print(f"n_paths=8000 seed=2 missing_iv_nodes={missing} of {k.size * maturities.size}")


if __name__ == "__main__":
    main()
