"""Figure for P1a: the same price() method, two models, state made visible.

Run from the repository root:

    python docs/figures/p1a_interface.py

Writes docs/figures/p1a_protocol.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from volsurface.models import BlackScholes, Heston, implied_surface

OUT = Path(__file__).with_name("p1a_protocol.png")


def main() -> None:
    # One maturity, a strip of log-moneyness. F = 1 so the strike is exp(k).
    T = 0.5
    k = np.linspace(-0.4, 0.3, 41)
    K = np.exp(k)
    heston = Heston(v0=0.04, kappa=1.5, theta=0.04, xi=0.8, rho=-0.7)
    flat = BlackScholes(sigma=float(np.sqrt(heston.v0)))
    iv_heston = implied_surface(heston, 1.0, K, [T])[0]
    iv_flat = implied_surface(flat, 1.0, K, [T])[0]
    # Same Heston, but the variance arrives as state rather than as v0.
    shifted = Heston(v0=0.09, kappa=1.5, theta=0.04, xi=0.8, rho=-0.7)
    iv_state = implied_surface(shifted, 1.0, K, [T], state={"v": np.array(heston.v0)})[0]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(k, 100 * iv_heston, color="C0", label="Heston, v0 = 0.04")
    ax.plot(k, 100 * iv_state, "x", color="C0", markevery=4, label="same model, state v = 0.04")
    ax.plot(k, 100 * iv_flat, color="C1", label="Black-Scholes, sigma = sqrt(v0)")
    ax.set_xlabel("log-moneyness k = log(K / F)")
    ax.set_ylabel("implied volatility (%)")
    ax.set_title("One price method. The state is the variance, and None means the parameter.")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
