"""Market skew, the merged Heston fit, and three rBergomi calibrations.

Run from the repository root:

    python scripts/figures_p3e.py

The Heston parameters are the merged P2c fit in LOG.md. The skew is
recomputed here. Each rBergomi curve is ``create("rbergomi")`` with ``xi0``
equal to the slope of the SANOS log contract. A missing plugin raises.
Nothing in this figure is a mock labelled rBergomi.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from volsurface import chains
from volsurface.core.market import market_data_from_snapshot
from volsurface.fit.rbergomi import calibrate, forward_variance
from volsurface.fit.sanos import SanosFitter
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

    print(f"power_law: {amplitude:.5f} T^(-{exponent:.5f})")
    print(f"n_expiries: {len(frame)}")
    short = frame.iloc[0]
    long = frame.iloc[-1]
    print(f"short T={short['T']:.5f} market {short['skew']:.4f} heston {short['heston_skew']:.4f}")
    print(f"long T={long['T']:.5f} market {long['skew']:.4f} heston {long['heston_skew']:.4f}")

    plugin = create("rbergomi")
    if not type(plugin).__module__.startswith("roughvol"):
        raise RuntimeError("create('rbergomi') did not resolve to roughvol")
    surface = SanosFitter(eta=0.25, omega=0.0).fit(data).surface
    xi0 = forward_variance(surface)
    tenors = frame["T"].to_numpy()
    print("xi0_front:", float(xi0(float(tenors[0]))))
    print("xi0_mid:", float(xi0(float(tenors[len(tenors) // 2]))))
    print("xi0_back:", float(xi0(float(tenors[-1]))))

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(frame["T"], frame["market_abs"], "o", ms=3.5, label="SPX |skew|")
    ax.plot(frame["T"], frame["heston_abs"], "s", ms=3.5, label="Heston |skew|")
    grid = np.geomspace(float(frame["T"].min()), float(frame["T"].max()), 80)
    ax.plot(grid, amplitude * grid ** (-exponent), color="0.3", lw=1.0, label="power law")
    markers = {"free": "d", "0.167": "^", "0.149": "v"}
    for label, hurst in (("free", None), ("0.167", 0.167), ("0.149", 0.149)):
        result = calibrate(data, H=hurst, seed=0, surface=surface)
        params = result.detail["parameters"]
        built = type(plugin)(
            H=params["H"],
            nu=params["nu"],
            rho=params["rho"],
            xi0=xi0,
            seed=0,
            n_paths=plugin.n_paths,
            n_steps=plugin.n_steps,
        )
        rough = model_term_structure(built, tenors, n=9)
        frame[f"rbergomi_{label}_skew"] = rough[:, 1]
        frame[f"rbergomi_{label}_atm"] = rough[:, 0]
        ax.plot(
            frame["T"],
            np.abs(rough[:, 1]),
            markers[label],
            ms=3.5,
            label=f"rBergomi H={label} |skew|",
        )
        print(
            f"setting={label} H={params['H']:.6f} nu={params['nu']:.6f} "
            f"rho={params['rho']:.6f} objective={result.detail['objective']:.6e} "
            f"wall_s={result.wall_time:.1f} status={result.solver_status} "
            f"xi0_front={result.detail['xi0_front']:.6f} "
            f"n_paths={result.detail['n_paths']} n_steps={result.detail['n_steps']}"
        )
        print(f"  skew short {rough[0, 1]:.4f} long {rough[-1, 1]:.4f} atm short {rough[0, 0]:.4f}")
    frame.to_csv(FIG / "p3e_skew.csv", index=False)
    ax.set_xscale("log")
    ax.set_xlabel("maturity")
    ax.set_ylabel("|ATM skew|")
    ax.set_title("SPX skew, Heston, and rBergomi with the SANOS forward curve.")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "p3e_skew.png", dpi=140)
    plt.close(fig)
    print(f"heston_front_level: {short['heston_atm']:.4f}")
    print(f"market_front_level: {short['atm']:.4f}")


if __name__ == "__main__":
    main()
