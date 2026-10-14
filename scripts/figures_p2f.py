"""Figures for Hagan's SABR smile and the wing where the density goes negative.

Run from the repository root:

    python scripts/figures_p2f.py

Writes PNG and CSV under docs/figures/, and data/presets/sabr.json.
The Monte Carlo panel uses fixed seeds. The right-hand maturity is outside
the small-nu^2 T regime. The gap there is the picture, not a tolerance.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from volsurface import bs
from volsurface.core.types import Curves, Domain
from volsurface.models.sabr import SABR, euler_price
from volsurface.surface.model import ModelSurface
from volsurface.surface_export import write_surface_preset

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"


def _flat(level):
    return lambda t: np.full(np.shape(t), level, dtype=float)


def smile_panel() -> dict:
    """Two smiles: rho at fixed nu, and nu at fixed rho. beta = 1, T = 1."""
    k = np.linspace(-0.5, 0.5, 81)
    strike = 100.0 * np.exp(k)
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8), sharey=True)
    for rho in (-0.6, -0.3, 0.0, 0.3, 0.6):
        model = SABR(alpha=0.2, beta=1.0, rho=rho, nu=0.4)
        iv = model.implied_vol(100.0, strike, 1.0)
        axes[0].plot(k, 100.0 * iv, lw=1.1, label=f"rho = {rho:+.1f}")
        for j, log_k in enumerate(k):
            rows.append(
                {"panel": "rho", "rho": rho, "nu": 0.4, "k": float(log_k), "iv": float(iv[j])}
            )
    for nu in (0.0, 0.2, 0.4, 0.8):
        model = SABR(alpha=0.2, beta=1.0, rho=-0.3, nu=nu)
        iv = model.implied_vol(100.0, strike, 1.0)
        axes[1].plot(k, 100.0 * iv, lw=1.1, label=f"nu = {nu:.1f}")
        for j, log_k in enumerate(k):
            rows.append(
                {"panel": "nu", "rho": -0.3, "nu": nu, "k": float(log_k), "iv": float(iv[j])}
            )
    axes[0].set_title("nu = 0.4. The slope follows rho.")
    axes[1].set_title("rho = -0.3. The wings follow nu.")
    for ax in axes:
        ax.set_xlabel("k")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel("implied volatility (%)")
    fig.suptitle("SABR, beta = 1, alpha = 0.2, T = 1, F = 100.")
    fig.tight_layout()
    fig.savefig(FIG / "p2f_smiles.png", dpi=140)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(FIG / "p2f_smiles.csv", index=False)
    return {"n_rows": len(rows)}


def density_panel() -> dict:
    """Hagan density at T = 10, nu = 1.5. The low strike wing is negative."""
    model = SABR(alpha=0.3, beta=1.0, rho=-0.3, nu=1.5)
    k = np.linspace(-1.5, 0.5, 81)
    market = Curves(F=_flat(100.0), df=_flat(1.0))
    surface = ModelSurface(model, market, Domain(0.5, 12.0, -2.0, 1.0))
    density = surface.density(np.array([10.0]), k)[0]
    iv = model.implied_vol(100.0, 100.0 * np.exp(k), 10.0)
    frame = pd.DataFrame({"k": k, "density": density, "iv": iv})
    frame.to_csv(FIG / "p2f_negative_density.csv", index=False)
    negative = density < 0.0
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    axes[0].plot(k, 100.0 * iv, color="C0")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("implied volatility (%)")
    axes[0].set_title("Hagan implied vol, T = 10")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(k, density, color="C1")
    axes[1].axhline(0.0, color="k", lw=0.6)
    if np.any(negative):
        axes[1].fill_between(
            k, density, 0.0, where=negative, color="C3", alpha=0.35, interpolate=True,
        )
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("density of log-moneyness")
    axes[1].set_title("Same prices. Shaded where the density is negative.")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("alpha = 0.3, beta = 1, rho = -0.3, nu = 1.5. nu^2 T = 22.5.")
    fig.tight_layout()
    fig.savefig(FIG / "p2f_negative_density.png", dpi=140)
    plt.close(fig)

    call = model.price(100.0, 100.0 * np.exp(k), 10.0) / 100.0
    write_surface_preset(
        ROOT / "data" / "presets" / "sabr.json",
        model="sabr",
        params={"alpha": 0.3, "beta": 1.0, "rho": -0.3, "nu": 1.5},
        T=np.array([10.0]),
        k=k,
        call=call.reshape(1, -1),
        iv=iv.reshape(1, -1),
        forward=100.0,
        discount=1.0,
        note=(
            "Hagan SABR at T=10, nu=1.5. The low-strike density of these "
            "prices is negative. The implied vol stays positive."
        ),
    )
    path = ROOT / "data" / "presets" / "sabr.json"
    finite = density[np.isfinite(density)]
    return {
        "min_density": float(np.min(finite)),
        "k_at_min": float(k[np.argmin(density)]),
        "n_negative": int(np.sum(negative)),
        "iv_min": float(np.min(iv)),
        "iv_max": float(np.max(iv)),
        "preset_bytes": path.stat().st_size,
    }


def monte_carlo_panel() -> dict:
    """Hagan against log-Euler in a small-nu^2 T regime and outside it.

    Strikes share paths. The error bar is one standard error of the implied
    vol. The right-hand panel is not a pass/fail comparison.
    """
    k = np.linspace(-0.2, 0.2, 9)
    strike = 100.0 * np.exp(k)
    specs = (
        ("small", SABR(alpha=0.2, beta=1.0, rho=-0.2, nu=0.15), 1.0, 8_000, 32, 11),
        ("large", SABR(alpha=0.25, beta=1.0, rho=0.0, nu=1.0), 2.0, 20_000, 64, 3),
    )
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8), sharey=False)
    summaries = {}
    for ax, (name, model, maturity, n_paths, n_steps, seed) in zip(axes, specs, strict=True):
        hagan = model.implied_vol(100.0, strike, maturity)
        result = euler_price(
            model, 100.0, strike, maturity, n_paths=n_paths, n_steps=n_steps, seed=seed,
        )
        iv = bs.implied_vol(result.value, 100.0, strike, maturity, 1.0)
        vega = bs.vega(100.0, strike, maturity, hagan, 1.0)
        se_iv = result.stderr / vega
        ax.plot(k, 100.0 * hagan, color="C0", lw=1.2, label="Hagan")
        ax.errorbar(
            k, 100.0 * iv, yerr=100.0 * se_iv, fmt="o", color="C1", ms=3.5, lw=0.8,
            label="log-Euler",
        )
        ax.set_xlabel("k")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=7)
        nu2t = model.nu**2 * maturity
        gap_bp = (iv - hagan) * 1e4
        ax.set_title(f"{name}: nu^2 T = {nu2t:.3f}, T = {maturity:g}")
        for j, log_k in enumerate(k):
            rows.append(
                {
                    "regime": name,
                    "k": float(log_k),
                    "iv_hagan": float(hagan[j]),
                    "iv_mc": float(iv[j]),
                    "se_iv": float(se_iv[j]),
                    "gap_bp": float(gap_bp[j]),
                    "nu2_T": float(nu2t),
                    "n_paths": n_paths,
                    "n_steps": n_steps,
                    "seed": seed,
                }
            )
        summaries[name] = {
            "nu2_T": float(nu2t),
            "max_abs_gap_bp": float(np.max(np.abs(gap_bp))),
            "gap_bp": [float(v) for v in gap_bp],
            "se_iv_bp": [float(v * 1e4) for v in se_iv],
        }
    axes[0].set_ylabel("implied volatility (%)")
    fig.suptitle("Strikes share paths. Bars are one standard error of the implied vol.")
    fig.tight_layout()
    fig.savefig(FIG / "p2f_hagan_vs_mc.png", dpi=140)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(FIG / "p2f_hagan_vs_mc.csv", index=False)
    return summaries


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "presets").mkdir(parents=True, exist_ok=True)
    summary = {
        "smiles": smile_panel(),
        "density": density_panel(),
        "monte_carlo": monte_carlo_panel(),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
