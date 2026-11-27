"""Fit Black-Scholes and Heston to a SANOS implied-vol grid, in normalised units.

A SANOS-calibrated market means a parametric fit to the SANOS surface from one
historical snapshot of index options, in a normalised market with spot 100.
The surface itself is not simulated. This module never writes quotes.

Two artefacts are produced from a user-supplied grid or snapshot:

* A training document, with the fitted parameters, for use in a hedger. It is
  written under a local exports folder and is not committed.
* An optional local site document, with the normalised contract, premiums and
  labels only. It is also written under the local folder. A committed example
  pack for the same schema is built from a labelled synthetic surface, not
  from a historical snapshot.

Heston is fitted twice: equally across the surface, and with residual weights
peaked at the hedged maturity. The default kappa box of 20 is a numerical
bound, not a no-arbitrage constraint. The contract-weighted fit uses a wider
kappa box so a short-dated calibration is not forced onto that bound. Heston
cannot match the short-dated skew of a typical equity-index surface; both
fits remain parametric Heston models, not a simulation of the surface.
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from volsurface import bs
from volsurface.core.params import from_unconstrained, to_unconstrained
from volsurface.core.types import MarketData, ParamSpec
from volsurface.fit._price_fit import (
    black_vega,
    gaussian_starts,
    model_prices,
    quote_frame,
    unconstrained_bounds,
)
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
SYNTHETIC_NOTE = (
    "Example pack for the snapshot-markets schema. Premiums are from a "
    "labelled synthetic Heston surface in a normalised market with spot 100, "
    "not from a historical snapshot."
)
HESTON_LIMIT_NOTE = (
    "Heston is a Markovian stochastic-volatility diffusion and cannot match "
    "the short-dated skew of a typical equity-index surface. The full-surface "
    "fit therefore spreads the error across maturities, including the hedged "
    "contract. The contract-weighted fit emphasises the hedged maturity; it "
    "is still a parametric Heston model, not a simulation of the surface. "
    "The default kappa upper bound of 20 is a numerical box, not a "
    "no-arbitrage constraint; the contract-weighted fit uses an upper bound "
    "of 50 so a short-dated calibration is not forced onto that box."
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

# The default Heston box, and a wider kappa box for the contract-weighted fit.
# 1/kappa at 50 is about one week, still a mean-reverting diffusion.
KAPPA_UPPER_DEFAULT = 20.0
KAPPA_UPPER_WIDE = 50.0
XI_UPPER_DEFAULT = 5.0
# Gaussian width for contract weights, in years. At 15 days from the
# hedged maturity the weight is exp(-1/2).
CONTRACT_WEIGHT_WIDTH = 15.0 / 365.0
BOUND_FRAC = 0.01

# Labelled synthetic surface used only for the committed schema example.
SYNTHETIC_HESTON = Heston(v0=0.03, kappa=1.2, theta=0.05, xi=0.4, rho=-0.6)

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


def contract_weights(
    T: np.ndarray,
    maturity: float,
    width: float = CONTRACT_WEIGHT_WIDTH,
) -> np.ndarray:
    """Gaussian weights peaked at the hedged maturity."""
    if width <= 0.0:
        raise ValueError("weight width must be positive")
    T = np.asarray(T, dtype=float)
    return np.exp(-0.5 * np.square((T - maturity) / width))


def heston_specs(*, kappa_upper: float, xi_upper: float) -> list[ParamSpec]:
    """Heston parameter specs with optional wider numerical boxes."""
    specs = []
    for spec in Heston.param_spec():
        if spec.name == "kappa":
            spec = replace(spec, upper=float(kappa_upper))
        elif spec.name == "xi":
            spec = replace(spec, upper=float(xi_upper))
        specs.append(spec)
    return specs


def bound_hits(
    params: dict[str, float],
    specs: list[ParamSpec],
    frac: float = BOUND_FRAC,
) -> list[str]:
    """Names of parameters that sit on their numerical box, to relative tolerance ``frac``."""
    hits = []
    for spec in specs:
        value = float(params[spec.name])
        scale = max(abs(spec.upper), abs(spec.lower), 1e-12)
        if abs(value - spec.upper) <= frac * scale or abs(value - spec.lower) <= frac * scale:
            hits.append(spec.name)
    return hits


def _model_from_x(x: np.ndarray, specs: list[ParamSpec]) -> Heston:
    return Heston(**from_unconstrained(np.asarray(x, dtype=float), specs))


def fit_heston(
    data: MarketData,
    T: np.ndarray,
    k: np.ndarray,
    iv: np.ndarray,
    *,
    weighting: str,
    n_starts: int = FIT_STARTS,
    seed: int = FIT_SEED,
    kappa_upper: float = KAPPA_UPPER_DEFAULT,
    xi_upper: float = XI_UPPER_DEFAULT,
    maturity: float = MATURITY,
    width: float = CONTRACT_WEIGHT_WIDTH,
) -> dict[str, Any]:
    """Vega-weighted least squares for one Heston parameter set.

    ``weighting="full"`` uses equal weights. ``weighting="contract"``
    multiplies the vega residual by the square root of a Gaussian weight
    peaked at ``maturity``, so the objective is a weighted sum of squares.
    """
    if weighting not in ("full", "contract"):
        raise ValueError(f"weighting must be 'full' or 'contract', got {weighting!r}")
    started = time.perf_counter()
    frame, _n_american = quote_frame(data, fitter="HestonFitter", allow_american=False)
    market = frame["price"].to_numpy(dtype=float)
    vega = black_vega(frame)
    if weighting == "contract":
        weights = contract_weights(frame["T"].to_numpy(dtype=float), maturity, width)
    else:
        weights = np.ones(len(frame), dtype=float)
    scale = np.sqrt(weights)
    specs = heston_specs(kappa_upper=kappa_upper, xi_upper=xi_upper)
    bounds = unconstrained_bounds(specs)
    origin = {spec.name: spec.default for spec in specs}
    starts = gaussian_starts(to_unconstrained(origin, specs), specs, n_starts, seed)
    best = None
    best_cost = np.inf
    for start in starts:
        result = least_squares(
            lambda x, _frame=frame, _market=market, _vega=vega, _scale=scale, _specs=specs: (
                _residual(_frame, _market, _vega, _model_from_x(x, _specs)) * _scale
            ),
            start,
            bounds=bounds,
            method="trf",
        )
        if result.cost < best_cost:
            best = result
            best_cost = float(result.cost)
    assert best is not None
    model = _model_from_x(best.x, specs)
    params = {
        "v0": float(model.v0),
        "kappa": float(model.kappa),
        "theta": float(model.theta),
        "xi": float(model.xi),
        "rho": float(model.rho),
    }
    residuals = model_prices(model, frame) - market
    metrics = _iv_metrics(model, T, k, iv, maturity, width)
    weight_sum = float(np.sum(weights))
    weight_sq = float(np.sum(np.square(weights)))
    return {
        **params,
        "feller": float(model.feller_ratio()),
        "premium": _heston_price(model, maturity),
        "solver_status": f"{best.message} cost={best.cost:.3e} starts={n_starts}",
        "wall_time": time.perf_counter() - started,
        "rmse_price": float(np.sqrt(np.mean(np.square(residuals)))),
        "max_abs_price": float(np.max(np.abs(residuals))),
        "weighting": weighting,
        "weight_width": None if weighting == "full" else float(width),
        "kappa_upper": float(kappa_upper),
        "xi_upper": float(xi_upper),
        "bound_hits": bound_hits(params, specs),
        "n_quotes": int(len(frame)),
        "weight_sum": weight_sum,
        "n_effective": float(weight_sum * weight_sum / weight_sq) if weight_sq > 0.0 else 0.0,
        **metrics,
    }


def _residual(frame, market: np.ndarray, vega: np.ndarray, model: Heston) -> np.ndarray:
    return (model_prices(model, frame) - market) / vega


def _iv_metrics(
    model: Heston,
    T: np.ndarray,
    k: np.ndarray,
    iv: np.ndarray,
    maturity: float,
    width: float,
) -> dict[str, Any]:
    model_iv = heston_iv_grid(model, T, k)
    ok = np.isfinite(iv) & np.isfinite(model_iv)
    gap = model_iv - iv
    rmse_iv = float(np.sqrt(np.mean(np.square(gap[ok])))) if ok.any() else float("nan")
    w_t = contract_weights(T, maturity, width)
    near = w_t >= math.exp(-0.5)
    ok_near = ok & near[:, None]
    if ok_near.any():
        rmse_iv_contract = float(np.sqrt(np.mean(np.square(gap[ok_near]))))
    else:
        rmse_iv_contract = float("nan")
    w = np.broadcast_to(w_t[:, None], iv.shape)
    if ok.any() and float(np.sum(w[ok])) > 0.0:
        rmse_iv_weighted = float(np.sqrt(np.sum(w[ok] * np.square(gap[ok])) / np.sum(w[ok])))
    else:
        rmse_iv_weighted = float("nan")
    filled = np.where(np.isfinite(model_iv), model_iv, np.nan)
    model_atm, model_skew = atm_level_and_skew(T, k, filled, maturity)
    return {
        "rmse_iv": rmse_iv,
        "rmse_iv_contract": rmse_iv_contract,
        "rmse_iv_weighted": rmse_iv_weighted,
        "n_points": int(ok.sum()),
        "n_points_contract": int(ok_near.sum()),
        "atm_vol": model_atm,
        "atm_skew": model_skew,
    }


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
    """Fit Black-Scholes and two Heston models to the grid.

    The default ``heston`` entry is the contract-weighted fit, which is the
    one a 30-day hedger should use. ``heston_fits["full"]`` keeps the
    equal-weighted surface fit with the default numerical box as a
    documented alternative.
    """
    T, k, iv = np.asarray(T, dtype=float), np.asarray(k, dtype=float), np.asarray(iv, dtype=float)
    sigma, skew = atm_level_and_skew(T, k, iv, maturity)
    surface_premium = float(bs.price(SPOT, STRIKE, maturity, sigma, df=1.0, kind="call"))
    data = normalised_book(T, k, iv)
    full = fit_heston(
        data,
        T,
        k,
        iv,
        weighting="full",
        n_starts=n_starts,
        seed=seed,
        kappa_upper=KAPPA_UPPER_DEFAULT,
        xi_upper=XI_UPPER_DEFAULT,
        maturity=maturity,
    )
    contract = fit_heston(
        data,
        T,
        k,
        iv,
        weighting="contract",
        n_starts=n_starts,
        seed=seed,
        kappa_upper=KAPPA_UPPER_WIDE,
        xi_upper=XI_UPPER_DEFAULT,
        maturity=maturity,
    )
    return {
        "schema": TRAIN_SCHEMA,
        "commit": commit if commit is not None else git_commit(),
        "note": NOTE,
        "heston_limit_note": HESTON_LIMIT_NOTE,
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
        "heston_fits": {
            "full": full,
            "contract": contract,
        },
        "heston": contract,
        "premium_gap": contract["premium"] - surface_premium,
    }


def site_document(train: dict[str, Any]) -> dict[str, Any]:
    """Strip a training document down to a site pack: no parameters, no grid.

    This pack may contain premiums from a user-supplied snapshot. Write it
    only under a local, gitignored folder. Do not commit it.
    """
    contract = dict(train["contract"])
    surface_premium = float(train["surface"]["premium"])
    heston_premium = float(train["heston"]["premium"])
    return {
        "schema": SITE_SCHEMA,
        "note": NOTE,
        "label": SITE_LABEL,
        "synthetic": False,
        "contract": contract,
        "markets": {
            "black_scholes": {
                "label": SITE_LABEL,
                "premium": float(train["black_scholes"]["premium"]),
            },
            "heston": {
                "label": (
                    "The network is trained in a Heston simulator fitted to that "
                    "surface. The surface itself is not being simulated. The fit "
                    "is weighted towards the hedged maturity."
                ),
                "premium": heston_premium,
            },
        },
        "surface_premium": surface_premium,
        "premium_gap": float(train["premium_gap"]),
    }


def synthetic_example_document(
    model: Heston = SYNTHETIC_HESTON,
    *,
    maturity: float = MATURITY,
) -> dict[str, Any]:
    """Site-pack schema example, priced on a labelled synthetic Heston surface."""
    sigma, _skew = _synthetic_atm(model, maturity)
    surface_premium = float(bs.price(SPOT, STRIKE, maturity, sigma, df=1.0, kind="call"))
    heston_premium = _heston_price(model, maturity)
    return {
        "schema": SITE_SCHEMA,
        "note": SYNTHETIC_NOTE,
        "label": SYNTHETIC_NOTE,
        "synthetic": True,
        "contract": {
            "s0": SPOT,
            "strike": STRIKE,
            "maturity": float(maturity),
            "r": RATE,
            "n_steps": N_STEPS,
        },
        "markets": {
            "black_scholes": {
                "label": SYNTHETIC_NOTE,
                "premium": surface_premium,
            },
            "heston": {
                "label": (
                    "The network would be trained in this synthetic Heston "
                    "simulator. Values are not from a historical snapshot."
                ),
                "premium": heston_premium,
            },
        },
        "surface_premium": surface_premium,
        "premium_gap": heston_premium - surface_premium,
    }


def _synthetic_atm(model: Heston, maturity: float) -> tuple[float, float]:
    dk = SKEW_DK
    strikes = SPOT * np.exp(np.array([-dk, 0.0, dk]))
    prices = np.asarray(model.price(SPOT, strikes, maturity, df=1.0, kind="call"), dtype=float)
    ivs = bs.implied_vol(prices, SPOT, strikes, maturity, df=1.0, kind="call")
    return float(ivs[1]), float(ivs[2] - ivs[0]) / (2.0 * dk)


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
    surface = (
        SanosFitter(
            eta=SANOS_ETA,
            omega=SANOS_OMEGA,
            dx=SANOS_DX,
            method=SANOS_METHOD,
            time_interp=SANOS_TIME_INTERP,
        )
        .fit(data)
        .surface
    )
    pillars = np.unique(np.asarray(surface.T, dtype=float))
    times = np.unique(np.concatenate([pillars, 0.5 * (pillars[:-1] + pillars[1:])]))
    k = np.asarray(SANOS_K, dtype=float)
    iv = np.asarray(surface.iv(times, k), dtype=float)
    return times, k, iv
