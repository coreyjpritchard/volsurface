"""Figures for the Heston cosine pricer and the SPX calibration.

Run from the repository root:

    python scripts/figures_p2c.py

Writes PNG and CSV under docs/figures/, and the widget preset
data/presets/heston.json. Seeds are fixed. The chain is a local SPXW snapshot
(CBOE delayed mids, European and cash-settled), which is not committed. The
committed smile CSVs hold the fitted model and per-expiry fit errors only; the
per-quote market implied vols go to data/cache/figures/, which git ignores.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

from volsurface import bs, chains
from volsurface.core.market import market_data_from_snapshot
from volsurface.fit.heston import HestonFitter
from volsurface.models.heston import COS_L, COS_N, Heston
from volsurface.smile import atm_fit, skew_window
from volsurface.surface_export import write_surface_preset

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
LOCAL = ROOT / "data" / "cache" / "figures"
FO = dict(v0=0.0175, kappa=1.5768, theta=0.0398, xi=0.5751, rho=-0.5711)
FO_CALL = 5.785155450
SOURCE = "SPXW mids, CBOE delayed, 14 Oct 2026"


def smile_errors(smiles: pd.DataFrame) -> pd.DataFrame:
    """Per-expiry model minus market implied vol, in vol points (1 point = 0.01 of vol)."""
    rows = []
    for maturity, group in smiles.groupby("T", sort=True):
        error = 100.0 * (group["model_iv"] - group["market_iv"]).to_numpy(dtype=float)
        rows.append(
            {
                "T": float(maturity),
                "n_quotes": int(error.size),
                "mean_vol_pts": float(np.mean(error)),
                "rmse_vol_pts": float(np.sqrt(np.mean(error**2))),
                "max_abs_vol_pts": float(np.max(np.abs(error))),
            }
        )
    return pd.DataFrame(rows)


def write_smiles(smiles: pd.DataFrame) -> None:
    """Commit the model smiles and per-expiry errors; keep per-quote market vols local."""
    LOCAL.mkdir(parents=True, exist_ok=True)
    smiles.to_csv(LOCAL / "p2c_spx_smiles.csv", index=False)
    smiles[["T", "k", "model_iv"]].to_csv(FIG / "p2c_spx_smiles.csv", index=False)
    smile_errors(smiles).to_csv(FIG / "p2c_spx_smile_errors.csv", index=False)


def cos_error_against_n() -> pd.DataFrame:
    """At-the-money T = 1 error against the Fang and Oosterlee reference."""
    model = Heston(**FO)
    lewis = float(model.price(100.0, np.array([100.0]), 1.0)[0])
    rows = []
    for n_terms in (16, 32, 64, 96, 128, 192, 256, 384, 512):
        cos = float(model.cos_price(100.0, np.array([100.0]), 1.0, n_terms=n_terms)[0])
        rows.append(
            {
                "N": n_terms,
                "L": COS_L,
                "cos": cos,
                "lewis": lewis,
                "reference": FO_CALL,
                "cos_minus_reference": cos - FO_CALL,
                "lewis_minus_reference": lewis - FO_CALL,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(FIG / "p2c_cos_error.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.loglog(
        frame["N"],
        np.abs(frame["cos_minus_reference"]),
        marker="o",
        label="|COS minus reference|",
    )
    ax.axhline(abs(lewis - FO_CALL), color="k", lw=0.8, label="|Lewis minus reference|")
    ax.axvline(COS_N, color="0.5", lw=0.8, ls="--", label=f"default N = {COS_N}")
    ax.set_xticks(frame["N"], [str(n) for n in frame["N"]])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks([1e-8, 1e-6, 1e-4, 1e-2, 1])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("number of cosine terms N (log scale)")
    ax.set_ylabel("absolute error in the call, cash units (F = 100)")
    at_default = float(np.abs(frame.loc[frame["N"] == COS_N, "cos_minus_reference"].iloc[0]))
    ax.set_title(
        f"COS error falls fast in N: {at_default:.0e} at the default N = {COS_N}\n"
        "Fang and Oosterlee (2008) Heston test case, T = 1, K = F = 100, L = 10",
        fontsize=9,
    )
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "p2c_cos_error.png", dpi=140)
    plt.close(fig)
    return frame


def lewis_minus_cos() -> pd.DataFrame:
    """Cash-price gap on a maturity by log-moneyness grid. FO parameters."""
    model = Heston(**FO)
    maturities = np.array([7.0 / 365.0, 1.0 / 12.0, 0.25, 0.5, 1.0, 2.0])
    log_k = np.linspace(-1.0, 1.0, 21)
    gap = np.empty((maturities.size, log_k.size))
    for i, maturity in enumerate(maturities):
        strike = 100.0 * np.exp(log_k)
        lewis = model.price(100.0, strike, float(maturity))
        cos = model.cos_price(100.0, strike, float(maturity))
        gap[i] = lewis - cos
    rows = []
    for i, maturity in enumerate(maturities):
        for j, k in enumerate(log_k):
            rows.append({"T": float(maturity), "k": float(k), "lewis_minus_cos": float(gap[i, j])})
    frame = pd.DataFrame(rows)
    frame.to_csv(FIG / "p2c_lewis_minus_cos.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    names = ["7 days", "1 month", "3 months", "6 months", "1 year", "2 years"]
    for i, name in enumerate(names):
        ax.plot(log_k, 1e8 * gap[i], marker="o", ms=3, lw=1, label=name)
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xlabel("log-moneyness k = log(K/F)")
    ax.set_ylabel("Lewis minus COS, cash price (units of 1e-8, F = 100)")
    ax.set_title(
        f"Lewis and COS agree to {np.abs(gap).max():.0e} in cash (F = 100) on the whole grid\n"
        "Fang and Oosterlee (2008) Heston parameters, COS with N = 256, L = 10",
        fontsize=9,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(
        title="maturity",
        frameon=False,
        fontsize=8,
        ncol=3,
        loc="center",
        bbox_to_anchor=(0.5, 0.58),
    )
    fig.tight_layout()
    fig.savefig(FIG / "p2c_lewis_minus_cos.png", dpi=140)
    plt.close(fig)
    return frame


def _thin(quotes: pd.DataFrame) -> pd.DataFrame:
    """Up to 11 strikes per expiry inside |k| <= 0.4, so the fit is the smile."""
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


def spx_calibration() -> tuple[pd.DataFrame, pd.DataFrame, Heston]:
    chain = chains.load_snapshot(chains.require_snapshot("SPX"))
    full = market_data_from_snapshot(chain)
    data = full.__class__(
        asof=full.asof,
        quotes=_thin(full.quotes),
        F=full.F,
        df=full.df,
        F_err=full.F_err,
        day_count=full.day_count,
    )
    fitted = HestonFitter(n_starts=4, loss="vega", seed=0).fit(data)
    row = fitted.slice_report.iloc[0]
    model = Heston(
        v0=float(row["v0"]),
        kappa=float(row["kappa"]),
        theta=float(row["theta"]),
        xi=float(row["xi"]),
        rho=float(row["rho"]),
    )
    summary = {
        "v0": model.v0,
        "kappa": model.kappa,
        "theta": model.theta,
        "xi": model.xi,
        "rho": model.rho,
        "feller": model.feller_ratio(),
        "solver_status": fitted.solver_status,
        "rmse_price": float(row["rmse_price"]),
        "max_abs_price": float(row["max_abs_price"]),
        "n_quotes": int(len(data.quotes)),
        "n_expiries": int(data.quotes["expiry"].nunique()),
    }
    (FIG / "p2c_spx_fit.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    # Smiles at four expiries nearest 1 day, 1 week, 1 month, 1 year.
    targets = np.array([1.0 / 365.0, 7.0 / 365.0, 30.0 / 365.0, 1.0])
    expiries = data.quotes.groupby("expiry", sort=True)["T"].first().sort_values()
    chosen = []
    for target in targets:
        label = expiries.index[int(np.argmin(np.abs(expiries.to_numpy() - target)))]
        if label not in chosen:
            chosen.append(label)

    smile_rows = []
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    handles = []
    for label in chosen:
        group = data.quotes[data.quotes["expiry"] == label].sort_values("k")
        maturity = float(group["T"].iloc[0])
        forward = float(group["F"].iloc[0])
        discount = float(group["df"].iloc[0])
        strike = group["K"].to_numpy(dtype=float)
        kind = group["kind"].to_numpy()
        model_price = model.price(forward, strike, maturity, discount, kind)
        model_iv = bs.implied_vol(model_price, forward, strike, maturity, discount, kind)
        (market_line,) = ax.plot(group["k"], 100 * group["iv"], marker="o", ms=3.5, lw=0)
        ax.plot(group["k"], 100 * model_iv, lw=1.2, color=market_line.get_color())
        n_days = 365.0 * maturity
        name = f"{n_days:.0f} days" if n_days >= 1.5 else f"{24 * n_days:.0f} hours"
        handles.append(
            Line2D([], [], color=market_line.get_color(), marker="o", ms=3.5, label=name)
        )
        for k, market_iv, fitted_iv in zip(group["k"], group["iv"], model_iv, strict=True):
            smile_rows.append(
                {
                    "T": maturity,
                    "k": float(k),
                    "market_iv": float(market_iv),
                    "model_iv": float(fitted_iv),
                }
            )
    handles += [
        Line2D([], [], color="0.3", marker="o", ms=3.5, lw=0, label="dots: SPXW mid quotes"),
        Line2D([], [], color="0.3", lw=1.2, label="lines: Heston, full-price fit"),
    ]
    ax.set_xlabel("log-moneyness k = log(K/F)")
    ax.set_ylabel("implied volatility (%)")
    ax.set_title(
        f"One Heston fit to all 40 expiries (prices, vega weighted) needs xi = {model.xi:.2f}\n"
        f"and a Feller ratio of {model.feller_ratio():.2f}; shown at four expiries",
        fontsize=9,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(handles=handles, title="expiry", frameon=False, fontsize=8)
    fig.text(
        0.5,
        0.01,
        f"{SOURCE}. Up to 11 strikes per expiry with |k| <= 0.4.",
        ha="center",
        fontsize=7,
        color="0.3",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "p2c_spx_smiles.png", dpi=140)
    plt.close(fig)
    smiles = pd.DataFrame(smile_rows)
    write_smiles(smiles)

    skew_rows = []
    # Market skew uses every quote, not the 11-strike subset the fitter saw.
    for _, group in full.quotes.groupby("expiry", sort=True):
        maturity = float(group["T"].iloc[0])
        window = skew_window(maturity)
        market_level, market_skew = atm_fit(group["k"], group["iv"], window)
        forward = float(group["F"].iloc[0])
        discount = float(group["df"].iloc[0])
        # A common strike grid inside the window, so the model skew is not the quote locations.
        grid = np.linspace(-window, window, 11)
        strike = forward * np.exp(grid)
        kind = np.where(grid < 0.0, "put", "call")
        price = model.price(forward, strike, maturity, discount, kind)
        iv = bs.implied_vol(price, forward, strike, maturity, discount, kind)
        model_level, model_skew = atm_fit(grid, iv, window)
        skew_rows.append(
            {
                "T": maturity,
                "market_level": market_level,
                "market_skew": market_skew,
                "model_level": model_level,
                "model_skew": model_skew,
            }
        )
    skew = pd.DataFrame(skew_rows).sort_values("T")
    skew.to_csv(FIG / "p2c_spx_skew.csv", index=False)

    finite = skew[np.isfinite(skew["market_skew"]) & np.isfinite(skew["model_skew"])]
    short = finite.iloc[0]
    over = finite[np.abs(finite["model_skew"]) > np.abs(finite["market_skew"]) * 1.05]
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    days = 365.0 * finite["T"]
    ax.loglog(
        days, np.abs(finite["market_skew"]), "o", ms=4.5, color="k", label="SPX, one dot per expiry"
    )
    ax.loglog(
        days,
        np.abs(finite["model_skew"]),
        "s-",
        ms=3.5,
        mfc="none",
        color="C1",
        lw=1,
        label=f"Heston, full-price fit: xi = {model.xi:.2f}, kappa = {model.kappa:.1f}, "
        f"Feller ratio {model.feller_ratio():.2f}",
    )
    xt = [0.5, 1, 2, 7, 30, 91, 365]
    ax.set_xticks(xt, [f"{x:g}" for x in xt])
    ax.xaxis.set_minor_locator(NullLocator())
    yt = [0.2, 0.3, 0.5, 1, 2, 3]
    ax.set_yticks(yt, [f"{y:g}" for y in yt])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("maturity (calendar days, log scale)")
    ax.set_ylabel("ATM skew |d(iv)/dk| at k = 0 (log scale)")
    ax.set_title(
        f"Fitting prices, Heston matches the 10-hour skew ({abs(short['model_skew']):.2f} "
        f"against {abs(short['market_skew']):.2f})\nonly with a Feller-violating "
        f"xi = {model.xi:.2f}, and then overshoots from "
        f"{365 * over['T'].min():.0f} to {365 * over['T'].max():.0f} days",
        fontsize=9,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.text(
        0.5,
        0.01,
        f"{SOURCE}. Skew: quadratic fit for |k| <= max(0.15 sqrt(T), 0.01).\n"
        "The headline figure fits ATM level and skew directly, with xi capped at 5.",
        ha="center",
        fontsize=7,
        color="0.3",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG / "p2c_spx_skew.png", dpi=140)
    plt.close(fig)
    return smiles, skew, model


def preset() -> None:
    """Fang and Oosterlee parameters, a small grid, schema volsurface.surface/v2."""
    maturities = np.array([7.0 / 365.0, 0.25, 1.0, 2.0])
    log_k = np.linspace(-0.5, 0.5, 21)
    fo = Heston(**FO)
    call = np.empty((maturities.size, log_k.size))
    iv = np.empty_like(call)
    for i, maturity in enumerate(maturities):
        strike = 100.0 * np.exp(log_k)
        cash = fo.cos_price(100.0, strike, float(maturity))
        call[i] = cash / 100.0
        iv[i] = bs.implied_vol(cash, 100.0, strike, float(maturity), 1.0, "call")
    write_surface_preset(
        ROOT / "data" / "presets" / "heston.json",
        model="heston",
        params=FO,
        T=maturities,
        k=log_k,
        call=call,
        iv=iv,
        forward=100.0,
        discount=1.0,
        note=(
            "Fang and Oosterlee (2008) parameters. Normalised call from the "
            "cosine put plus put-call parity. N is chosen from the interval "
            "and raised until successive prices agree to 1e-7. L=10. F=100, df=1."
        ),
    )


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    error = cos_error_against_n()
    gap = lewis_minus_cos()
    _, skew, _ = spx_calibration()
    preset()
    at_256 = error.loc[error["N"] == 256].iloc[0]
    print(f"cos N=256 minus reference {at_256['cos_minus_reference']:.3e}")
    print(f"lewis minus reference {at_256['lewis_minus_reference']:.3e}")
    print(f"heatmap max |Lewis-COS| {gap['lewis_minus_cos'].abs().max():.3e}")
    finite = skew[np.isfinite(skew["market_skew"])]
    short = finite.iloc[0]
    long = finite.iloc[-1]
    print(
        f"skew T={short['T']:.4f} market {short['market_skew']:.3f} model {short['model_skew']:.3f}"
    )
    print(f"skew T={long['T']:.4f} market {long['market_skew']:.3f} model {long['model_skew']:.3f}")
    print(json.loads((FIG / "p2c_spx_fit.json").read_text()))
    over = finite[np.abs(finite["model_skew"]) > 1.05 * np.abs(finite["market_skew"])]
    print(
        f"heston skew above market by >5%: {365 * over['T'].min():.2f} to "
        f"{365 * over['T'].max():.2f} days, {len(over)} expiries"
    )


if __name__ == "__main__":
    main()
