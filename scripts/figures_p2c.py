"""Figures for the Heston cosine pricer and the SPX calibration.

Run from the repository root:

    python scripts/figures_p2c.py

Writes PNG and CSV under docs/figures/, and the widget preset
data/presets/heston.json. Seeds are fixed. The chain is the committed
SPXW file: CBOE delayed mids, European and cash-settled.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from volsurface import bs, chains
from volsurface.core.market import market_data_from_snapshot
from volsurface.fit.heston import HestonFitter
from volsurface.models.heston import COS_L, COS_N, Heston
from volsurface.smile import atm_fit, skew_window
from volsurface.surface_export import write_surface_preset

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
FO = dict(v0=0.0175, kappa=1.5768, theta=0.0398, xi=0.5751, rho=-0.5711)
FO_CALL = 5.785155450


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
        frame["N"], np.abs(frame["cos_minus_reference"]), marker="o", label="COS minus reference",
    )
    ax.axhline(abs(lewis - FO_CALL), color="k", lw=0.8, label="|Lewis minus reference|")
    ax.axvline(COS_N, color="0.5", lw=0.8, ls="--", label=f"N = {COS_N}")
    ax.set_xlabel("number of cosine terms N")
    ax.set_ylabel("absolute error in the cash call")
    ax.set_title("Fang and Oosterlee, T = 1, K = F = 100, L = 10")
    ax.grid(True, which="both", alpha=0.3)
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
    image = ax.imshow(
        gap,
        origin="lower",
        aspect="auto",
        extent=[log_k[0], log_k[-1], 0, maturities.size - 1],
        cmap="coolwarm",
    )
    ax.set_yticks(range(maturities.size))
    ax.set_yticklabels([f"{t:.3f}" for t in maturities])
    ax.set_xlabel("log-moneyness k")
    ax.set_ylabel("maturity T (year fraction)")
    ax.set_title("Lewis minus COS, cash price, N = 256, L = 10")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="cash price")
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
    fitted = HestonFitter(n_starts=4, loss="vega", seed=0).fit(data)
    row = fitted.slice_report.iloc[0]
    model = Heston(
        v0=float(row["v0"]), kappa=float(row["kappa"]), theta=float(row["theta"]),
        xi=float(row["xi"]), rho=float(row["rho"]),
    )
    summary = {
        "v0": model.v0,
        "kappa": model.kappa,
        "theta": model.theta,
        "xi": model.xi,
        "rho": model.rho,
        "feller": model.feller_ratio(),
        "wall_time": fitted.wall_time,
        "solver_status": fitted.solver_status,
        "rmse_price": float(row["rmse_price"]),
        "max_abs_price": float(row["max_abs_price"]),
        "n_quotes": int(len(data.quotes)),
        "n_expiries": int(data.quotes["expiry"].nunique()),
    }
    (FIG / "p2c_spx_fit.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    # Smiles at four expiries nearest 1 day, 1 week, 1 month, 1 year.
    targets = np.array([1.0 / 365.0, 7.0 / 365.0, 30.0 / 365.0, 1.0])
    expiries = (
        data.quotes.groupby("expiry", sort=True)["T"].first().sort_values()
    )
    chosen = []
    for target in targets:
        label = expiries.index[int(np.argmin(np.abs(expiries.to_numpy() - target)))]
        if label not in chosen:
            chosen.append(label)

    smile_rows = []
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for label in chosen:
        group = data.quotes[data.quotes["expiry"] == label].sort_values("k")
        maturity = float(group["T"].iloc[0])
        forward = float(group["F"].iloc[0])
        discount = float(group["df"].iloc[0])
        strike = group["K"].to_numpy(dtype=float)
        kind = group["kind"].to_numpy()
        model_price = model.price(forward, strike, maturity, discount, kind)
        model_iv = bs.implied_vol(model_price, forward, strike, maturity, discount, kind)
        market_line, = ax.plot(
            group["k"], 100 * group["iv"], marker="o", ms=3.5, lw=0,
            label=f"market T={maturity:.3f}",
        )
        ax.plot(
            group["k"], 100 * model_iv, lw=1.2, color=market_line.get_color(),
            label=f"Heston T={maturity:.3f}",
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
    ax.set_xlabel("log-moneyness k")
    ax.set_ylabel("implied volatility (%)")
    ax.set_title("Heston fit against SPX mids, 14 Oct 2026")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "p2c_spx_smiles.png", dpi=140)
    plt.close(fig)
    smiles = pd.DataFrame(smile_rows)
    smiles.to_csv(FIG / "p2c_spx_smiles.csv", index=False)

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
    long = finite.iloc[-1]
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.plot(finite["T"], finite["market_skew"], marker="o", ms=3.5, label="SPX mids")
    ax.plot(finite["T"], finite["model_skew"], marker="s", ms=3.5, label="fitted Heston")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xlabel("maturity T (year fraction)")
    ax.set_ylabel("ATM skew d iv / d k")
    ax.set_title(
        f"T={short['T']:.4f}: market {short['market_skew']:.2f}, "
        f"model {short['model_skew']:.2f}. "
        f"T={long['T']:.3f}: market {long['market_skew']:.2f}, "
        f"model {long['model_skew']:.2f}."
    )
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
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
            "cosine put plus put-call parity, N=256, L=10. F=100, df=1."
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
    print(
        f"skew T={long['T']:.4f} market {long['market_skew']:.3f} model {long['model_skew']:.3f}"
    )
    print(json.loads((FIG / "p2c_spx_fit.json").read_text()))


if __name__ == "__main__":
    main()
