"""SANOS smoothing, the eta sweep, and the shoot-out on SPX.

Run from the repository root:

    python scripts/figures_p3c.py

The market bar is zero static-arbitrage violations at tolerance 1e-9 with
omega = 0, and an in-spread fraction strictly above 0.558. The solver is
the HiGHS interior point method. omega = 1 is a second fit, labelled not
guaranteed.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from volsurface import bs, chains
from volsurface.arbitrage import check_surface
from volsurface.core.market import market_data_from_snapshot
from volsurface.core.types import ArbTol, Curves
from volsurface.fit.sanos import SanosFitter, quote_in_spread
from volsurface.fit.svi import SVIFitter
from volsurface.surface.linear import linear_surface
from volsurface.surface.spline import spline_surface

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
TARGETS = (7 / 365, 30 / 365, 90 / 365, 0.9)
K_GRID = np.linspace(-0.4, 0.3, 71)
TOL = ArbTol(price=1e-9)


def _times(surface) -> np.ndarray:
    if hasattr(surface, "T"):
        raw = surface.T
    else:
        raw = [pillar["T"] for pillar in surface.pillars]
    pillars = np.asarray(raw, dtype=float)
    pillars = np.unique(pillars)
    if pillars.size < 2:
        return pillars
    mids = 0.5 * (pillars[:-1] + pillars[1:])
    return np.unique(np.concatenate([pillars, mids]))


def _smoothness(surface, times, log_k) -> float:
    """Root mean square of the second difference of local variance along k."""
    picked = times[:: max(len(times) // 6, 1)][:6]
    nodes = log_k[::3]
    try:
        local = np.asarray(surface.local_var(picked, nodes), dtype=float)
    except Exception:
        return float("nan")
    finite = np.isfinite(local)
    if finite.sum() < 5:
        return float("nan")
    second = local[:, 2:] - 2.0 * local[:, 1:-1] + local[:, :-2]
    mask = np.isfinite(second)
    if not mask.any():
        return float("nan")
    return float(np.sqrt(np.mean(second[mask] ** 2)))


def _row(name: str, surface, fit_time: float, in_spread: float, times: np.ndarray) -> dict:
    report = check_surface(surface, times, K_GRID, tol=TOL)
    return {
        "model": name,
        "in_spread": in_spread,
        "n_violations": int(len(report.violations)),
        "min_density": report.min_density,
        "local_var_rms": _smoothness(surface, times, K_GRID),
        "fit_time_s": fit_time,
        "ok": report.ok,
    }


def _nearest(surface, target: float) -> int:
    times = np.asarray(surface.T, dtype=float)
    return int(np.argmin(np.abs(times - target)))


def main() -> None:
    chain = chains.load_snapshot(chains.require_snapshot("SPX"))
    data = market_data_from_snapshot(chain)
    FIG.mkdir(parents=True, exist_ok=True)
    market = Curves(data.F, data.df)

    sweep_rows = []
    surfaces = {}
    for eta in (0.0, 0.0625, 0.25):
        print(f"fitting eta={eta}", flush=True)
        t0 = time.perf_counter()
        result = SanosFitter(
            eta=eta, omega=0.0, dx=0.05, time_interp="pchip", method="highs-ipm"
        ).fit(data)
        elapsed = time.perf_counter() - t0
        surfaces[eta] = result
        times = _times(result.surface)
        report = check_surface(result.surface, times, K_GRID, tol=TOL)
        sweep_rows.append(
            {
                "eta": eta,
                "omega": 0.0,
                "omega_label": result.detail["omega_label"],
                "objective": result.detail["objective"],
                "in_spread": result.in_spread,
                "projection_l1": result.detail["projection_l1"],
                "wall_time_s": elapsed,
                "fit_time_s": result.wall_time,
                "n_violations": int(len(report.violations)),
                "min_density": report.min_density,
                "n_checked": report.n_checked,
                "ok": report.ok,
            }
        )
        print(sweep_rows[-1], flush=True)
        result.slice_report.to_csv(FIG / f"p3c_slices_eta{eta}.csv", index=False)

    print("fitting omega=1", flush=True)
    t0 = time.perf_counter()
    try:
        variant = SanosFitter(
            eta=0.25, omega=1.0, dx=0.05, time_interp="pchip", method="highs-ipm"
        ).fit(data)
        elapsed = time.perf_counter() - t0
        times = _times(variant.surface)
        report = check_surface(variant.surface, times, K_GRID, tol=TOL)
        sweep_rows.append(
            {
                "eta": 0.25,
                "omega": 1.0,
                "omega_label": variant.detail["omega_label"],
                "objective": variant.detail["objective"],
                "in_spread": variant.in_spread,
                "projection_l1": variant.detail["projection_l1"],
                "wall_time_s": elapsed,
                "fit_time_s": variant.wall_time,
                "n_violations": int(len(report.violations)),
                "min_density": report.min_density,
                "n_checked": report.n_checked,
                "ok": report.ok,
            }
        )
    except Exception as exc:
        sweep_rows.append(
            {
                "eta": 0.25,
                "omega": 1.0,
                "omega_label": "not guaranteed",
                "objective": np.nan,
                "in_spread": np.nan,
                "projection_l1": np.nan,
                "wall_time_s": time.perf_counter() - t0,
                "fit_time_s": np.nan,
                "n_violations": np.nan,
                "min_density": np.nan,
                "n_checked": np.nan,
                "ok": False,
                "error": str(exc),
            }
        )
    print(sweep_rows[-1], flush=True)
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(FIG / "p3c_eta_sweep.csv", index=False)

    production = surfaces[0.25]
    surface = production.surface
    chosen = []
    for target in TARGETS:
        j = _nearest(surface, target)
        if j not in chosen:
            chosen.append(j)

    # PCHIP jump of dC/dT at an interior expiry, three step sizes.
    interior = [j for j in chosen if 0 < j < len(surface.T) - 1]
    pivot = interior[0] if interior else chosen[0]
    maturity = float(surface.T[pivot])
    jumps = []
    for step in (1e-4, 1e-5, 1e-6):
        for log_k in (0.0, -0.15, 0.15):
            above = float(np.asarray(surface.call([maturity + step], [log_k])).reshape(-1)[0])
            mid = float(np.asarray(surface.call([maturity], [log_k])).reshape(-1)[0])
            below = float(np.asarray(surface.call([maturity - step], [log_k])).reshape(-1)[0])
            jumps.append(
                {
                    "T": maturity,
                    "k": log_k,
                    "h": step,
                    "jump": (above - mid) / step - (mid - below) / step,
                }
            )
    pd.DataFrame(jumps).to_csv(FIG / "p3c_pchip_jumps.csv", index=False)
    print(pd.DataFrame(jumps).to_string(index=False), flush=True)

    fig, axes = plt.subplots(2, 4, figsize=(11.2, 5.6), sharex=True)
    quotes = data.quotes
    for col, j in enumerate(chosen):
        maturity = float(surface.T[j])
        log_k = np.linspace(-0.35, 0.25, 81)
        model_iv = surface.iv([maturity], log_k)[0]
        axes[0, col].plot(log_k, model_iv, color="C0", lw=1.2, label="SANOS eta=0.25")
        group = quotes[np.isclose(quotes["T"], maturity, rtol=0.0, atol=1e-8)]
        if group.empty:
            # The quote year fraction can differ in the last digits from the frame.
            expiry = production.slice_report["T"]
            nearest = float(expiry.iloc[(expiry - maturity).abs().argmin()])
            group = quotes[np.isclose(quotes["T"], nearest, rtol=0.0, atol=1e-6)]
        if not group.empty:
            bid_iv = []
            ask_iv = []
            ks = []
            for _, row in group.iterrows():
                crossed = not (
                    np.isfinite(row["bid"]) and np.isfinite(row["ask"]) and row["ask"] > row["bid"]
                )
                if crossed:
                    continue
                kind = row["kind"]
                b = bs.implied_vol(row["bid"], row["F"], row["K"], row["T"], row["df"], kind)
                a = bs.implied_vol(row["ask"], row["F"], row["K"], row["T"], row["df"], kind)
                b = float(np.asarray(b).reshape(-1)[0])
                a = float(np.asarray(a).reshape(-1)[0])
                if np.isfinite(b) and np.isfinite(a):
                    ks.append(float(row["k"]))
                    bid_iv.append(b)
                    ask_iv.append(a)
            axes[0, col].scatter(ks, bid_iv, s=4, c="0.45", label="bid")
            axes[0, col].scatter(ks, ask_iv, s=4, c="0.75", label="ask")
        axes[0, col].set_title(f"T = {maturity:.3f}")
        axes[0, col].grid(True, alpha=0.3)
        phi = surface.density([maturity], log_k)[0]
        axes[1, col].plot(log_k, phi, color="C0", lw=1.2)
        axes[1, col].grid(True, alpha=0.3)
        axes[1, col].set_xlabel("log-moneyness")
    axes[0, 0].set_ylabel("implied vol")
    axes[1, 0].set_ylabel("density of log spot")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle(
        f"SANOS eta = 0.25, omega = 0. in spread {production.in_spread:.3f}, "
        f"objective {production.detail['objective']:.3e}."
    )
    fig.tight_layout()
    fig.savefig(FIG / "p3c_sanos_smiles.png", dpi=140)
    plt.close(fig)

    # One axis. The in-spread fraction and violation count go in the point labels,
    # because they barely move and a second axis would only show a flat line.
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    guaranteed = sweep[sweep["omega"] == 0.0]
    ax.plot(
        guaranteed["eta"],
        guaranteed["objective"],
        "-o",
        color="C0",
        label="omega = 0 (calendar arbitrage ruled out by construction)",
    )
    for _, row in guaranteed.iterrows():
        ax.annotate(
            f"in spread {row['in_spread']:.3f}\n{int(row['n_violations'])} violations",
            (row["eta"], row["objective"]),
            textcoords="offset points",
            xytext=(5, 6),
            ha="left",
            fontsize=7.5,
        )
    loose = sweep[(sweep["omega"] == 1.0) & np.isfinite(sweep["objective"])]
    if not loose.empty:
        row = loose.iloc[0]
        ax.plot(
            [row["eta"]],
            [row["objective"]],
            "s",
            mfc="none",
            color="C3",
            label="omega = 1 (not guaranteed)",
        )
        ax.annotate(
            f"in spread {row['in_spread']:.3f}\n{int(row['n_violations'])} violations",
            (row["eta"], row["objective"]),
            textcoords="offset points",
            xytext=(-8, -4),
            ha="right",
            fontsize=7.5,
        )
    ax.set_yscale("log")
    shown = sweep["objective"][np.isfinite(sweep["objective"])]
    ax.set_ylim(float(shown.min()) / 1.5, float(shown.max()) * 2.0)
    ax.set_xlim(-0.02, 0.31)
    ax.set_xlabel("smoothing parameter eta (dimensionless)")
    ax.set_ylabel("LP objective (rescaled, dimensionless, log scale)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    top = guaranteed.iloc[-1]
    labels = [f"{value:g}" for value in guaranteed["eta"]]
    etas = ", ".join(labels[:-1]) + " and " + labels[-1]
    objectives = guaranteed["objective"].to_numpy()
    rising = bool(np.all(np.diff(objectives) > 0.0))
    clean = bool(np.all(guaranteed["n_violations"] == 0) and np.all(guaranteed["in_spread"] == 1.0))
    peak_exp = int(np.floor(np.log10(objectives.max())))
    peak = f"${objectives.max() / 10**peak_exp:.1f} \\times 10^{{{peak_exp}}}$"
    trend = "rises with eta" if rising else "does not rise steadily with eta"
    status = (
        "keeps every SPX quote inside its spread\nwith no static-arbitrage violations"
        if clean
        else "does not keep every quote inside its spread with zero violations"
    )
    extra = ""
    if not loose.empty:
        extra = f" (omega = 1 fails {int(loose.iloc[0]['n_violations'])} checks)"
    fig.suptitle(
        f"With omega = 0, each of eta = {etas} {status}; the LP objective stays\n"
        f"at or below {peak} and {trend}{extra}",
        fontsize=10.5,
    )
    fig.text(
        0.01,
        0.005,
        "SPXW mids, CBOE delayed, 14 Oct 2026. HiGHS interior point, PCHIP in time; "
        f"violations checked on {int(top['n_checked'])} points at tolerance 1e-9.",
        fontsize=7.5,
        color="0.3",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.subplots_adjust(top=0.84)
    fig.savefig(FIG / "p3c_eta_sweep.png", dpi=140)
    plt.close(fig)

    print("shoot-out", flush=True)
    times = _times(surface)
    rows = []
    t0 = time.perf_counter()
    linear = linear_surface(data.quotes, market)
    rows.append(
        _row(
            "linear",
            linear,
            time.perf_counter() - t0,
            quote_in_spread(data.quotes, linear),
            times,
        )
    )
    print(rows[-1], flush=True)
    t0 = time.perf_counter()
    spline = spline_surface(data.quotes, market)
    rows.append(
        _row(
            "spline",
            spline,
            time.perf_counter() - t0,
            quote_in_spread(data.quotes, spline),
            times,
        )
    )
    print(rows[-1], flush=True)
    t0 = time.perf_counter()
    svi = SVIFitter().fit(data)
    rows.append(
        _row(
            "SVI",
            svi.surface,
            time.perf_counter() - t0,
            quote_in_spread(data.quotes, svi.surface),
            times,
        )
    )
    print(rows[-1], "svi_own", svi.in_spread, flush=True)
    rows.append(
        _row(
            "SANOS eta=0.25",
            surface,
            float(production.wall_time),
            float(production.in_spread),
            times,
        )
    )
    print(rows[-1], flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(FIG / "p3c_shootout.csv", index=False)
    print(table.to_string(index=False), flush=True)

    bar = sweep[(sweep["eta"] == 0.25) & (sweep["omega"] == 0.0)].iloc[0]
    if not (bar["ok"] and bar["n_violations"] == 0 and bar["in_spread"] > 0.558):
        raise SystemExit(f"market bar failed: {bar.to_dict()}")
    print("market bar passed", flush=True)


if __name__ == "__main__":
    main()
