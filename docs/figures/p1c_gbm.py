"""Figure for P1c: one geometric Brownian motion, inverted to implied volatility.

Eight thousand paths, one seed, three expiries, shared. Strikes left of the
money are priced as puts and strikes at or right of the money as calls, which
is the out-of-the-money contract. The model is exact Black dynamics with
volatility 0.20, so the Black implied volatility is a horizontal line.

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
    # Out of the money: a put when K < F, a call when K >= F.
    kind = np.where(k < 0.0, "put", "call")
    model = GBM(sigma=sigma, scheme="exact", antithetic=False, control_variate=False)
    result = price_grid(model, F, K, maturities, df, kind, n_paths=8_000, seed=2)

    def m(x: float, fmt: str) -> str:
        """A number in mathtext, so a negative sign is a true minus."""
        return f"${format(x, fmt)}$"

    ivs = [bs.implied_vol(result.value[i], F, K, t, df, kind) for i, t in enumerate(maturities)]
    gaps = [100 * (iv - sigma) for iv in ivs]
    missing = int(sum(np.isnan(iv).sum() for iv in ivs))
    # The longer two expiries, and the week inside its wings.
    core = np.concatenate([np.abs(g[np.isfinite(g)]) for g in gaps[1:]])
    week = gaps[0][np.isfinite(gaps[0])]

    fig, axes = plt.subplots(2, 1, figsize=(7.4, 7.4), sharex=True)
    ax = axes[0]
    ax.axhline(100 * sigma, color="k", lw=1.0, label="Black, 20% (the true value)")
    for i, label in enumerate(labels):
        ax.plot(k, 100 * ivs[i], marker="o", ms=3.5, color=f"C{i}", label=f"Monte Carlo, {label}")
    finite = np.concatenate([100 * iv[np.isfinite(iv)] for iv in ivs])
    pad = 0.4
    ax.set_ylim(min(finite.min(), 100 * sigma) - pad, max(finite.max(), 100 * sigma) + pad)
    ax.set_ylabel("implied volatility (%)")
    ax.set_title("Implied vol, one dot per strike: puts for k < 0, calls for k ≥ 0", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for i, label in enumerate(labels):
        ax.plot(k, gaps[i], marker="o", ms=3.5, color=f"C{i}", label=label)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("log-moneyness k = log(K / F)")
    ax.set_ylabel("Monte Carlo minus Black (vol points)")
    ax.set_title(
        f"Error in vol points. {missing} of {k.size * maturities.size} nodes have no implied "
        "vol: no path crossed the strike",
        fontsize=10,
    )
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.suptitle(
        f"8,000 GBM paths recover the flat 20% smile to within {core.max():.1f} vol points "
        f"at 3 months and 1 year;\nthe 1-week put wing is off by up to {m(week.min(), '.1f')} "
        "vol points, because few paths reach those strikes",
        fontsize=10.5,
    )
    fig.text(
        0.01,
        0.005,
        "Simulation: geometric Brownian motion, forward 100, volatility 20%, exact scheme, "
        "8,000 paths, seed 2, no variance reduction.",
        fontsize=7.5,
        color="0.3",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.subplots_adjust(top=0.9)
    fig.savefig(OUT, dpi=140)
    print(f"wrote {OUT}")
    print(f"n_paths=8000 seed=2 missing_iv_nodes={missing} of {k.size * maturities.size}")
    for i, (t, label) in enumerate(zip(maturities, labels, strict=True)):
        iv = bs.implied_vol(result.value[i], F, K, t, df, kind)
        gap = 100.0 * (iv - sigma)
        where = np.flatnonzero(~np.isfinite(iv))
        ks = ", ".join(f"{k[j]:+.3f}" for j in where) if where.size else "none"
        finite = gap[np.isfinite(gap)]
        mean_gap = float(np.mean(finite)) if finite.size else float("nan")
        print(f"{label}: missing k=[{ks}] mean_iv_gap_vol_pts={mean_gap:.3f}")


if __name__ == "__main__":
    main()
