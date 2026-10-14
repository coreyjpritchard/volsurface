"""Market skew, the merged Heston fit, and rBergomi if the plugin is installed.

Run from the repository root:

    python scripts/figures_p3e.py

The Heston parameters are the merged P2c fit in LOG.md. The skew is
recomputed here. rBergomi is drawn only when ``create("rbergomi")`` resolves
to the roughvol package. A missing plugin is a line in the log, not a curve.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from volsurface import chains
from volsurface.core.market import market_data_from_snapshot
from volsurface.models.heston import Heston
from volsurface.plugins import create
from volsurface.smile import market_term_structure, model_term_structure, power_law

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"

# Merged P2c fit, LOG.md. Skew is recomputed, not copied.
HESTON = Heston(v0=0.007982, kappa=13.466, theta=0.03836, xi=2.703, rho=-0.6508)


def main() -> None:
    chain = chains.load_snapshot(chains.latest_snapshot("SPX"))
    data = market_data_from_snapshot(chain)
    market = market_term_structure(data.quotes)
    panel = model_term_structure(HESTON, market["T"].to_numpy(), n=9)
    frame = market.copy()
    frame["heston_atm"] = panel[:, 0]
    frame["heston_skew"] = panel[:, 1]
    frame["market_abs"] = np.abs(frame["skew"])
    frame["heston_abs"] = np.abs(frame["heston_skew"])
    amplitude, exponent = power_law(frame["T"].to_numpy(), frame["skew"].to_numpy())
    FIG.mkdir(parents=True, exist_ok=True)
    frame.to_csv(FIG / "p3e_skew.csv", index=False)

    plugin = importlib.util.find_spec("roughvol") is not None
    print(f"roughvol_installed: {plugin}")
    print(f"power_law: {amplitude:.5f} T^(-{exponent:.5f})")
    print(f"n_expiries: {len(frame)}")
    short = frame.iloc[0]
    long = frame.iloc[-1]
    print(f"short T={short['T']:.5f} market {short['skew']:.4f} heston {short['heston_skew']:.4f}")
    print(f"long T={long['T']:.5f} market {long['skew']:.4f} heston {long['heston_skew']:.4f}")

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(frame["T"], frame["market_abs"], "o", ms=3.5, label="SPX |skew|")
    ax.plot(frame["T"], frame["heston_abs"], "s", ms=3.5, label="Heston |skew|")
    grid = np.geomspace(float(frame["T"].min()), float(frame["T"].max()), 80)
    ax.plot(grid, amplitude * grid ** (-exponent), color="0.3", lw=1.0, label="power law")
    if plugin:
        model = create("rbergomi")
        if type(model).__module__.startswith("roughvol"):
            rough = model_term_structure(model, frame["T"].to_numpy(), n=9)
            ax.plot(frame["T"], np.abs(rough[:, 1]), "d", ms=3.5, label="rBergomi |skew|")
    ax.set_xscale("log")
    ax.set_xlabel("maturity")
    ax.set_ylabel("|ATM skew|")
    ax.set_title("SPX skew, the merged Heston fit, and the power law.")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "p3e_skew.png", dpi=140)
    plt.close(fig)
    print(f"heston_front_level: {short['heston_atm']:.4f}")
    print(f"market_front_level: {short['atm']:.4f}")


if __name__ == "__main__":
    main()
