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
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from volsurface import bs, chains
from volsurface.core.market import market_data_from_snapshot
from volsurface.core.types import Curves, Domain
from volsurface.fit.sabr import SABRFitter
from volsurface.models.sabr import SABR, euler_price, integrated_black_mixture
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
            k,
            density,
            0.0,
            where=negative,
            color="C3",
            alpha=0.35,
            interpolate=True,
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
            model,
            100.0,
            strike,
            maturity,
            n_paths=n_paths,
            n_steps=n_steps,
            seed=seed,
        )
        iv = bs.implied_vol(result.value, 100.0, strike, maturity, 1.0)
        vega = bs.vega(100.0, strike, maturity, hagan, 1.0)
        se_iv = result.stderr / vega
        ax.plot(k, 100.0 * hagan, color="C0", lw=1.2, label="Hagan")
        ax.errorbar(
            k,
            100.0 * iv,
            yerr=100.0 * se_iv,
            fmt="o",
            color="C1",
            ms=3.5,
            lw=0.8,
            label="log-Euler",
        )
        iv_exact = np.full(k.shape, np.nan)
        se_exact = np.full(k.shape, np.nan)
        if model.beta == 1.0 and model.rho == 0.0:
            mixture = integrated_black_mixture(
                model,
                100.0,
                strike,
                maturity,
                n_paths=100_000,
                n_steps=1_000,
                seed=seed + 17,
            )
            iv_exact = bs.implied_vol(mixture.value, 100.0, strike, maturity, 1.0)
            exact_vega = bs.vega(100.0, strike, maturity, iv_exact, 1.0)
            se_exact = mixture.stderr / exact_vega
            ax.plot(k, 100.0 * iv_exact, color="C2", lw=1.2, label="exact mixture")
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
                    "iv_exact": float(iv_exact[j]),
                    "se_exact": float(se_exact[j]),
                    "hagan_minus_exact_bp": float((hagan[j] - iv_exact[j]) * 1e4),
                    "mc_minus_exact_bp": float((iv[j] - iv_exact[j]) * 1e4),
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
            "hagan_minus_exact_bp": [float(v) for v in (hagan - iv_exact) * 1e4],
            "mc_minus_exact_bp": [float(v) for v in (iv - iv_exact) * 1e4],
            "se_exact_bp": [float(v * 1e4) for v in se_exact],
        }
    axes[0].set_ylabel("implied volatility (%)")
    fig.suptitle(
        "Right-hand panel: the mixture is exact (beta = 1, rho = 0). Bars are one standard error."
    )
    fig.tight_layout()
    fig.savefig(FIG / "p2f_hagan_vs_mc.png", dpi=140)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(FIG / "p2f_hagan_vs_mc.csv", index=False)
    return summaries


def _thin(quotes: pd.DataFrame) -> pd.DataFrame:
    """Up to 11 strikes per expiry inside |k| <= 0.4, the same cut as the Heston fit."""
    kept = []
    for _, group in quotes.groupby("expiry", sort=True):
        group = group[np.abs(group["k"]) <= 0.4].dropna(subset=["iv", "price"])
        if len(group) < 5:
            continue
        if len(group) > 11:
            order = np.argsort(group["k"].to_numpy())
            pick = np.linspace(0, len(order) - 1, 11).round().astype(int)
            group = group.iloc[order[pick]]
        kept.append(group)
    return pd.concat(kept, ignore_index=True)


def spx_fit() -> dict:
    """One SABR parameter set per SPXW expiry. beta is fixed at 1."""
    chain = chains.load_snapshot(chains.latest_snapshot("SPX"))
    full = market_data_from_snapshot(chain)
    data = full.__class__(
        asof=full.asof,
        quotes=_thin(full.quotes),
        F=full.F,
        df=full.df,
        F_err=full.F_err,
        day_count=full.day_count,
    )
    fitted = SABRFitter(beta=1.0, n_starts=4, loss="vega", seed=0).fit(data)
    report = fitted.slice_report
    report.to_csv(FIG / "p2f_spx_sabr.csv", index=False)

    days = 365.0 * report["T"].to_numpy()
    nu_cap = 4.0
    rho_floor = -0.999
    at_nu = np.isclose(report["nu"].to_numpy(), nu_cap, atol=1e-6)
    at_rho = np.isclose(report["rho"].to_numpy(), rho_floor, atol=1e-6)
    print(
        f"nu at cap on {int(at_nu.sum())} expiries, last at {days[at_nu].max():.1f} days; "
        f"rho at floor on {int(at_rho.sum())} expiries, last at {days[at_rho].max():.1f} days",
        flush=True,
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.4), sharex=True)
    panels = (
        (axes[0, 0], 100.0 * report["alpha"], "alpha: vol level (%, annualised)", None, None),
        (axes[0, 1], report["rho"], "rho: spot-vol correlation", rho_floor, at_rho),
        (axes[1, 0], report["nu"], "nu: vol of vol (annualised)", nu_cap, at_nu),
        (axes[1, 1], report["rmse_price"], "price RMSE (index points)", None, None),
    )
    for ax, values, label, bound, hit in panels:
        values = np.asarray(values, dtype=float)
        ax.plot(days, values, "-o", ms=3, lw=1.0, color="C0", label="fitted value")
        if bound is not None:
            ax.axhline(bound, color="C3", ls="--", lw=0.8, label=f"bound ${bound:g}$")
            ax.plot(
                days[hit],
                values[hit],
                "o",
                ms=6,
                mfc="none",
                color="C3",
                label=f"at the bound ({int(hit.sum())} expiries)",
            )
            ax.legend(frameon=False, fontsize=7.5, loc="best")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    for ax in axes[1]:
        ax.set_xscale("log")
        ax.set_xticks([0.5, 1, 2, 5, 10, 30, 100, 350])
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("time to expiry (days, log scale)")
    median = float(report["rmse_price"].median())
    fig.suptitle(
        f"SABR fitted one SPX expiry at a time has a median price RMSE of {median:.2f} index "
        f"points, but the short end is bound-limited:\n"
        f"nu sits at its cap of {nu_cap:g} up to {days[at_nu].max():.0f} days and rho at "
        f"${rho_floor:g}$ on the first {int(at_rho.sum())} expiries, so read those values "
        "as limits, not estimates",
        fontsize=10.5,
    )
    fig.text(
        0.01,
        0.005,
        f"SPXW mids, CBOE delayed, 14 Oct 2026. beta = 1, Hagan lognormal formula, vega-weighted "
        f"price loss, 4 starts; up to 11 strikes per expiry with |k| <= 0.4 "
        f"({len(data.quotes)} quotes, {len(report)} expiries).",
        fontsize=7.5,
        color="0.3",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.subplots_adjust(top=0.88)
    fig.savefig(FIG / "p2f_spx_sabr.png", dpi=140)
    plt.close(fig)

    summary = {
        "n_quotes": int(len(data.quotes)),
        "n_expiries": int(len(report)),
        "solver_status": fitted.solver_status,
        "rmse_median": float(report["rmse_price"].median()),
        "rmse_max": float(report["rmse_price"].max()),
        "T_at_rmse_max": float(report.loc[report["rmse_price"].idxmax(), "T"]),
        "alpha_at_T1": float(report.iloc[(report["T"] - 1.0).abs().argmin()]["alpha"]),
        "rho_at_T1": float(report.iloc[(report["T"] - 1.0).abs().argmin()]["rho"]),
        "nu_at_T1": float(report.iloc[(report["T"] - 1.0).abs().argmin()]["nu"]),
        "T_nearest_1": float(report.iloc[(report["T"] - 1.0).abs().argmin()]["T"]),
    }
    (FIG / "p2f_spx_sabr.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary["wall_time"] = fitted.wall_time
    return summary


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "presets").mkdir(parents=True, exist_ok=True)
    summary = {
        "smiles": smile_panel(),
        "density": density_panel(),
        "monte_carlo": monte_carlo_panel(),
        "spx": spx_fit(),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
