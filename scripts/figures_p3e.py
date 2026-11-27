"""Market skew, the merged Heston fit, and three rBergomi calibrations.

Run from the repository root:

    python scripts/figures_p3e.py

Heston is fitted to the loaded chain (the synthetic demo by default).
Each rBergomi curve is ``create("rbergomi")`` with ``xi0``
equal to the slope of the SANOS log contract. A missing plugin raises.
Nothing in this figure is a mock labelled rBergomi.

The Heston skew is measured with ``iv_tol=1e-8`` so the outer strikes at the
shortest expiries are not dropped (see ``docs/notes/heston_fit_old_vs_new.md``).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullLocator

from volsurface.fit.heston import HestonFitter
from volsurface.fit.rbergomi import calibrate, forward_variance
from volsurface.fit.sanos import SanosFitter
from volsurface.models.heston import Heston
from volsurface.plugins import create
from volsurface.smile import market_term_structure, model_term_structure, power_law

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"

def main() -> None:
    from volsurface.demo_market import load_figure_market, market_data
    chain, _data, source, FIG, _is_synthetic = load_figure_market()
    FIG.mkdir(parents=True, exist_ok=True)
    data = market_data(chain)
    market = market_term_structure(data.quotes)
    fitted_heston = HestonFitter(n_starts=4, loss="vega", seed=0).fit(data)
    row = fitted_heston.slice_report.iloc[0]
    heston_model = Heston(
        v0=float(row["v0"]),
        kappa=float(row["kappa"]),
        theta=float(row["theta"]),
        xi=float(row["xi"]),
        rho=float(row["rho"]),
    )
    print(
        f"heston v0={heston_model.v0:.6f} kappa={heston_model.kappa:.4f} "
        f"theta={heston_model.theta:.6f} xi={heston_model.xi:.4f} "
        f"rho={heston_model.rho:.4f} feller={heston_model.feller_ratio():.3f}"
    )
    panel = model_term_structure(heston_model, market["T"].to_numpy(), n=9, iv_tol=1e-8)
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

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    days = 365.0 * frame["T"]
    ax.plot(days, frame["market_abs"], "o", ms=5, color="k", label="chain, one dot per expiry")
    ax.plot(
        days,
        frame["heston_abs"],
        "s",
        ms=3.5,
        mfc="none",
        color="C1",
        label=(
            f"Heston, full-price fit (xi = {heston_model.xi:.2f}, "
            f"Feller {heston_model.feller_ratio():.2f})"
        ),
    )
    grid = np.geomspace(float(frame["T"].min()), float(frame["T"].max()), 80)
    ax.plot(
        365.0 * grid,
        amplitude * grid ** (-exponent),
        color="0.3",
        lw=1.0,
        ls=":",
        label=f"power law ${amplitude:.3f}\\,T^{{-{exponent:.3f}}}$, H = {0.5 - exponent:.2f}",
    )
    markers = {"free": "d", "0.10": "^", "0.15": "v"}
    fitted_h = {}
    over = frame[frame["heston_abs"] > 1.05 * frame["market_abs"]]
    if len(over):
        over_lo, over_hi = 365.0 * over["T"].min(), 365.0 * over["T"].max()
        print(
            f"heston skew above market by >5%: {over_lo:.2f} to {over_hi:.2f} days, "
            f"{len(over)} expiries"
        )
    else:
        print("heston skew is within 5% of the chain at every expiry")
    for label, hurst in (("free", None), ("0.10", 0.10), ("0.15", 0.15)):
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
        fitted_h[label] = params["H"]
        name = f"H free (fitted {params['H']:.3f})" if label == "free" else f"H = {label}"
        ax.plot(
            days,
            np.abs(rough[:, 1]),
            markers[label],
            ms=3.5,
            mfc="none",
            color={"free": "C0", "0.10": "C2", "0.15": "C4"}[label],
            label=f"rough Bergomi, {name}",
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
    ax.set_yscale("log")
    xt = [0.5, 1, 2, 7, 30, 91, 365]
    ax.set_xticks(xt, [f"{x:g}" for x in xt])
    ax.xaxis.set_minor_locator(NullLocator())
    yt = [0.2, 0.3, 0.5, 1, 2, 3]
    ax.set_yticks(yt, [f"{y:g}" for y in yt])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("maturity (calendar days, log scale)")
    ax.set_ylabel("ATM skew |d(iv)/dk| at k = 0 (log scale)")
    ax.set_title(
        f"Rough Bergomi (H from {min(fitted_h.values()):.2f} to {max(fitted_h.values()):.2f}) "
        "tracks the chain skew across the listed expiries;\n"
        f"Heston (xi = {heston_model.xi:.2f}) compared on the same expiries",
        fontsize=10,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.text(
        0.5,
        0.01,
        f"{source}. Rough Bergomi: forward variance from the "
        "SANOS log contract, Monte Carlo, seed 0.",
        ha="center",
        fontsize=7,
        color="0.3",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "p3e_skew.png", dpi=140)
    plt.close(fig)
    print(f"heston_front_level: {short['heston_atm']:.4f}")
    print(f"market_front_level: {short['atm']:.4f}")


if __name__ == "__main__":
    main()
