"""Fit Black-Scholes and Heston to a SANOS implied-vol grid, in normalised units.

A SANOS-calibrated market means a parametric fit to the SANOS surface from one
historical snapshot of index options, in a normalised market with spot 100.
The surface itself is not simulated. This module never writes quotes.

Two artefacts are produced:

* A training document, with the fitted parameters, for use in a hedger. It is
  written under a local exports folder and is not a site pack.
* A site document, with the normalised contract, premiums and labels only. It
  omits the Heston parameters, the implied-vol grid, dates and expiry labels.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from volsurface import bs
from volsurface.core.types import MarketData
from volsurface.fit.heston import HestonFitter
from volsurface.models.heston import Heston

SITE_SCHEMA = "volsurface.snapshot_markets/v1"
TRAIN_SCHEMA = "volsurface.sanos_calibration/v1"
GRID_SCHEMA = "volsurface.sanos_grid/v1"
SITE_LABEL = (
    "calibrated to the fitted surface from one historical snapshot of "
    "index options, in a normalised market with spot 100"
)
NOTE = (
    "A SANOS-calibrated market is a parametric fit to the SANOS surface from "
    "one historical snapshot of index options, in a normalised market with "
    "spot 100. The target is the SANOS surface, not raw quotes."
)

# Normalised contract used by the hedger.
SPOT = 100.0
STRIKE = 100.0
MATURITY = 30.0 / 365.0
RATE = 0.0
N_STEPS = 30
FIT_SEED = 0
FIT_STARTS = 4
SKEW_DK = 0.01

# SANOS settings that match the public surface demo.
SANOS_ETA = 0.25
SANOS_OMEGA = 0.0
SANOS_DX = 0.05
SANOS_METHOD = "highs-ipm"
SANOS_TIME_INTERP = "pchip"
SANOS_K = np.linspace(-0.4, 0.3, 71)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def git_commit(cwd: Path | None = None) -> str:
    """Return the current commit hash, or empty if git is unavailable."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=cwd or _repo_root(),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return ""


def load_grid(path: Path | str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a normalised implied-vol grid: arrays ``T``, ``k``, ``iv``.

    The file is JSON with ``T``, ``k`` and ``iv``. Missing implied vols are
    JSON null and become NaN. Dates and expiry labels are refused.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("dates", "expiry", "expiries", "asof"):
        if key in payload:
            raise ValueError(f"grid file must not contain {key!r}")
    T = np.asarray(payload["T"], dtype=float).reshape(-1)
    k = np.asarray(payload["k"], dtype=float).reshape(-1)
    iv = np.asarray(payload["iv"], dtype=float)
    if iv.shape != (T.size, k.size):
        raise ValueError(f"iv shape {iv.shape} does not match T {T.size} and k {k.size}")
    if T.size < 2 or k.size < 3:
        raise ValueError("grid needs at least two maturities and three strikes")
    order = np.argsort(T)
    T, iv = T[order], iv[order]
    order = np.argsort(k)
    k, iv = k[order], iv[:, order]
    return T, k, iv


def write_grid(
    path: Path | str,
    T: np.ndarray,
    k: np.ndarray,
    iv: np.ndarray,
    *,
    note: str = NOTE,
) -> None:
    """Write a normalised implied-vol grid. No dates, no quotes, no labels."""
    T = np.asarray(T, dtype=float).reshape(-1)
    k = np.asarray(k, dtype=float).reshape(-1)
    iv = np.asarray(iv, dtype=float)
    if iv.shape != (T.size, k.size):
        raise ValueError(f"iv shape {iv.shape} does not match T {T.size} and k {k.size}")
    payload = {
        "schema": GRID_SCHEMA,
        "T": T.tolist(),
        "k": k.tolist(),
        "iv": [[None if not np.isfinite(v) else float(v) for v in row] for row in iv],
        "note": note,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def interpolate_iv(T: np.ndarray, k: np.ndarray, iv: np.ndarray, T_q: float, k_q: float) -> float:
    """Implied vol at ``(T_q, k_q)`` by linear interpolation of total variance."""
    if T_q <= 0.0:
        raise ValueError("maturity must be positive")
    w = np.square(iv) * T[:, None]
    w_k = np.full(k.size, np.nan)
    for j in range(k.size):
        col = w[:, j]
        ok = np.isfinite(col)
        if ok.sum() < 2:
            continue
        w_k[j] = float(np.interp(T_q, T[ok], col[ok]))
    ok = np.isfinite(w_k)
    if ok.sum() < 2:
        raise ValueError("not enough finite total variance columns to interpolate")
    w_q = float(np.interp(k_q, k[ok], w_k[ok]))
    if w_q < 0.0:
        raise ValueError("interpolated total variance is negative")
    return float(np.sqrt(w_q / T_q))


def atm_level_and_skew(
    T: np.ndarray,
    k: np.ndarray,
    iv: np.ndarray,
    maturity: float,
    dk: float = SKEW_DK,
) -> tuple[float, float]:
    """At-the-money implied vol and central-difference skew at ``maturity``."""
    level = interpolate_iv(T, k, iv, maturity, 0.0)
    up = interpolate_iv(T, k, iv, maturity, dk)
    down = interpolate_iv(T, k, iv, maturity, -dk)
    return level, (up - down) / (2.0 * dk)


def normalised_book(T: np.ndarray, k: np.ndarray, iv: np.ndarray) -> MarketData:
    """Build a unit-forward book from the grid: spot 100, discount 1, no spreads."""
    rows = []
    tt, kk = np.meshgrid(T, k, indexing="ij")
    for t, log_k, sigma in zip(tt.ravel(), kk.ravel(), iv.ravel(), strict=True):
        if not np.isfinite(sigma) or sigma <= 0.0 or t <= 0.0:
            continue
        strike = SPOT * np.exp(log_k)
        price = float(bs.price(SPOT, strike, t, sigma, df=1.0, kind="call"))
        rows.append(
            {
                "T": float(t),
                "K": float(strike),
                "F": SPOT,
                "df": 1.0,
                "k": float(log_k),
                "iv": float(sigma),
                "price": price,
                "kind": "call",
                "exercise": "european",
            }
        )
    if not rows:
        raise ValueError("grid has no finite implied vols")
    frame = pd.DataFrame(rows)

    def _curve(value: float):
        return lambda t, _value=value: np.full(np.shape(t), _value, dtype=float)

    return MarketData(
        asof=datetime(1970, 1, 1, tzinfo=UTC),
        quotes=frame,
        F=_curve(SPOT),
        df=_curve(1.0),
        F_err=None,
        day_count="ACT/365",
    )



def heston_iv_grid(model: Heston, T: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Implied vols of a Heston model on the ``(T, k)`` grid, one maturity at a time."""
    T = np.asarray(T, dtype=float).reshape(-1)
    k = np.asarray(k, dtype=float).reshape(-1)
    iv = np.full((T.size, k.size), np.nan)
    strikes = SPOT * np.exp(k)
    for i, maturity in enumerate(T):
        prices = np.asarray(
            model.price(SPOT, strikes, float(maturity), df=1.0, kind="call"),
            dtype=float,
        )
        iv[i] = bs.implied_vol(prices, SPOT, strikes, float(maturity), df=1.0, kind="call")
    return iv


def _heston_price(model: Heston, maturity: float) -> float:
    return float(np.asarray(model.price(SPOT, STRIKE, maturity, df=1.0, kind="call")).reshape(()))


def calibrate_from_grid(
    T: np.ndarray,
    k: np.ndarray,
    iv: np.ndarray,
    *,
    seed: int = FIT_SEED,
    n_starts: int = FIT_STARTS,
    maturity: float = MATURITY,
    commit: str | None = None,
) -> dict[str, Any]:
    """Fit Black-Scholes and Heston to the grid and return a training document."""
    T, k, iv = np.asarray(T, dtype=float), np.asarray(k, dtype=float), np.asarray(iv, dtype=float)
    sigma, skew = atm_level_and_skew(T, k, iv, maturity)
    surface_premium = float(bs.price(SPOT, STRIKE, maturity, sigma, df=1.0, kind="call"))
    data = normalised_book(T, k, iv)
    fitted = HestonFitter(n_starts=n_starts, loss="vega", seed=seed).fit(data)
    row = fitted.slice_report.iloc[0]
    model = Heston(
        v0=float(row["v0"]),
        kappa=float(row["kappa"]),
        theta=float(row["theta"]),
        xi=float(row["xi"]),
        rho=float(row["rho"]),
    )
    heston_premium = _heston_price(model, maturity)
    heston_iv = heston_iv_grid(model, T, k)
    ok = np.isfinite(iv) & np.isfinite(heston_iv)
    iv_gap = heston_iv[ok] - iv[ok]
    rmse_iv = float(np.sqrt(np.mean(np.square(iv_gap)))) if ok.any() else float("nan")
    model_iv = np.where(np.isfinite(heston_iv), heston_iv, np.nan)
    model_atm, model_skew = atm_level_and_skew(T, k, model_iv, maturity)
    return {
        "schema": TRAIN_SCHEMA,
        "commit": commit if commit is not None else git_commit(),
        "note": NOTE,
        "seed": int(seed),
        "n_starts": int(n_starts),
        "contract": {
            "s0": SPOT,
            "strike": STRIKE,
            "maturity": float(maturity),
            "r": RATE,
            "n_steps": N_STEPS,
        },
        "black_scholes": {
            "sigma": sigma,
            "premium": surface_premium,
        },
        "surface": {
            "atm_vol": sigma,
            "atm_skew": skew,
            "premium": surface_premium,
        },
        "heston": {
            "v0": float(row["v0"]),
            "kappa": float(row["kappa"]),
            "theta": float(row["theta"]),
            "xi": float(row["xi"]),
            "rho": float(row["rho"]),
            "feller": float(row["feller"]),
            "premium": heston_premium,
            "solver_status": fitted.solver_status,
            "wall_time": fitted.wall_time,
            "rmse_price": float(row["rmse_price"]),
            "max_abs_price": float(row["max_abs_price"]),
            "rmse_iv": rmse_iv,
            "n_points": int(ok.sum()),
            "atm_vol": model_atm,
            "atm_skew": model_skew,
        },
        "premium_gap": heston_premium - surface_premium,
    }


def site_document(train: dict[str, Any]) -> dict[str, Any]:
    """Strip a training document down to a site pack: no parameters, no grid."""
    contract = dict(train["contract"])
    surface_premium = float(train["surface"]["premium"])
    heston_premium = float(train["heston"]["premium"])
    return {
        "schema": SITE_SCHEMA,
        "note": NOTE,
        "label": SITE_LABEL,
        "contract": contract,
        "markets": {
            "black_scholes": {
                "label": SITE_LABEL,
                "premium": float(train["black_scholes"]["premium"]),
            },
            "heston": {
                "label": (
                    "The network is trained in a Heston simulator fitted to that "
                    "surface. The surface itself is not being simulated."
                ),
                "premium": heston_premium,
            },
        },
        "surface_premium": surface_premium,
        "premium_gap": float(train["premium_gap"]),
    }


def write_json(path: Path | str, document: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def grid_from_snapshot(path: Path | str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rebuild a normalised SANOS implied-vol grid from a local snapshot file.

    The snapshot path is an input. Quotes are not written. Pillars and the
    midpoints between them are evaluated on ``SANOS_K``.
    """
    from volsurface.chains import load_snapshot
    from volsurface.core.market import market_data_from_snapshot
    from volsurface.fit.sanos import SanosFitter

    data = market_data_from_snapshot(load_snapshot(path), exercise="european")
    surface = SanosFitter(
        eta=SANOS_ETA,
        omega=SANOS_OMEGA,
        dx=SANOS_DX,
        method=SANOS_METHOD,
        time_interp=SANOS_TIME_INTERP,
    ).fit(data).surface
    pillars = np.unique(np.asarray(surface.T, dtype=float))
    times = np.unique(np.concatenate([pillars, 0.5 * (pillars[:-1] + pillars[1:])]))
    k = np.asarray(SANOS_K, dtype=float)
    iv = np.asarray(surface.iv(times, k), dtype=float)
    return times, k, iv
