"""Calibrate each dashboard model to the 14 October 2026 SPXW chain and save the result.

Run from the repository root, after ``scripts/build_dashboard_presets.py`` (it reads the
SVI and SANOS fits saved there). About five minutes, most of it the Heston multi-start:

    python scripts/build_model_fits.py

Writes ``data/presets/spx_model_fits.json``: the fitted parameters, how each was fitted,
and the implied-vol RMSE against the quotes, so the vol lab's "Fitted" mode can show a
calibrated model without calibrating (and its web build without any download).

The quotes are the 8,973 out-of-the-money SPXW mids of the local ``data/snapshots/SPX_2026-10-14``
(:func:`volsurface.environments.otm_quotes`): 40 expiries, 10 hours to 351 days. One RMSE
is computed the same way for every model, :func:`volsurface.environments.fit_quality`,
even though each model was fitted to a different loss:

- Black-Scholes: the one vol that minimises the implied-vol RMSE over all the quotes,
  which is their mean.
- Heston: the fit of ``docs/figure_00.py`` (same box, starts, loss and tolerance): the
  at-the-money level and skew of every expiry, ``xi`` in [0.05, 5]. ``xi`` ends on its
  upper bound; there is no interior optimum in ``xi`` on this loss (LOG.md, 2026-10-14).
- SABR: :class:`volsurface.fit.sabr.SABRFitter` with ``beta = 1`` held fixed, one
  ``(alpha, rho, nu)`` per expiry, vega-weighted price residuals.
- SVI and SANOS: the saved fits, unchanged.
- Rough Bergomi: :func:`volsurface.fit.rbergomi.calibrate` with ``H`` free, on the
  forward variance curve of the saved SANOS surface, at-the-money level and skew, seed 0
  (needs the rough-volatility dev extra). Its RMSE is a Monte Carlo estimate.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc

from volsurface import chains, models, smile
from volsurface import environments as env_lib
from volsurface.core.market import market_data_from_snapshot
from volsurface.fit.sabr import SABRFitter
from volsurface.models.sabr import free_spec

ROOT = Path(__file__).resolve().parents[1]
OUT = env_lib.MODEL_FITS_PATH
SABR_BETA = 1.0

# Heston: the box, starts and tolerance of docs/figure_00.py (keep the two in step).
LO = np.array([1e-4, 0.1, 0.001, 0.05, -0.99])
HI = np.array([0.15, 50.0, 0.3, 5.0, 0.99])
NAMES = ("v0", "kappa", "theta", "xi", "rho")
HAND_STARTS = (
    np.array([0.011, 3.0, 0.04, 0.4, -0.9]),
    np.array([0.04, 1.5, 0.04, 0.5, -0.7]),
    np.array([0.02, 10.0, 0.08, 1.5, -0.99]),
)
IV_TOL = 1e-8


def heston_fit(snapshot) -> dict:
    market = chains.market_implied_vols(snapshot)
    ts = smile.market_term_structure(market)
    maturities = ts["T"].to_numpy()
    target = np.concatenate([ts["atm"].to_numpy(), ts["skew"].to_numpy()])

    def residuals(x):
        pred = smile.model_term_structure(models.Heston(*x), maturities, iv_tol=IV_TOL)
        return np.concatenate([pred[:, 0], pred[:, 1]]) - target

    def safe(x):
        r = residuals(x)
        return np.where(np.isfinite(r), r, 10.0)

    sobol = qmc.Sobol(d=5, scramble=True, seed=0).random_base2(5)
    lo, hi = LO + 1e-3 * (HI - LO), HI - 1e-3 * (HI - LO)
    starts = [*HAND_STARTS, *qmc.scale(sobol, lo, hi)]
    starts = [x0 for x0 in starts if np.all(np.isfinite(residuals(x0)))]
    results = [least_squares(safe, x0, bounds=(LO, HI), method="trf") for x0 in starts]
    order = sorted(range(len(results)), key=lambda i: (round(float(results[i].cost), 10), i))
    best = results[order[0]]
    tol = 1e-4 * (HI - LO)
    at_bound = [
        n
        for n, v, a, b, t in zip(NAMES, best.x, LO, HI, tol, strict=True)
        if abs(v - a) <= t or abs(v - b) <= t
    ]
    params = dict(zip(NAMES, map(float, best.x), strict=True))
    return {
        "params": params,
        "at_bound": at_bound,
        "bounds": {n: [float(a), float(b)] for n, a, b in zip(NAMES, LO, HI, strict=True)},
        "cost": float(best.cost),
        "n_starts": len(results),
        "feller": 2 * params["kappa"] * params["theta"] / params["xi"] ** 2,
        "method": (
            "At-the-money level and skew of all 40 expiries, unweighted least squares "
            "(docs/figure_00.py), 35 starts. xi ends on its upper bound of 5: on this loss "
            "Heston has no interior optimum in xi."
        ),
    }


def sabr_fit(data) -> dict:
    report = SABRFitter(beta=SABR_BETA, n_starts=4, seed=0).fit(data).slice_report
    bounds = {spec.name: (spec.lower, spec.upper) for spec in free_spec()}
    # Expiries whose fit ends within 0.1% of the box edge, per parameter.
    at_bound = {}
    for name, (lo, hi) in bounds.items():
        values = report[name].to_numpy()
        edge = (np.abs(values - lo) <= 1e-3 * (hi - lo)) | (np.abs(values - hi) <= 1e-3 * (hi - lo))
        at_bound[name] = report["T"].to_numpy()[edge].tolist()
    return {
        "bounds": {name: [float(lo), float(hi)] for name, (lo, hi) in bounds.items()},
        "at_bound_T": at_bound,
        "beta": SABR_BETA,
        "T": report["T"].tolist(),
        "alpha": report["alpha"].tolist(),
        "rho": report["rho"].tolist(),
        "nu": report["nu"].tolist(),
        "method": (
            "Hagan (2002) SABR fitted expiry by expiry, beta fixed at 1, (alpha, rho, nu) by "
            "vega-weighted least squares on the mid prices (volsurface.fit.sabr), 4 starts. "
            "On the shortest expiries nu ends on its upper bound of 4 and rho near -1: "
            "one SABR smile cannot bend as sharply as the SPX smile a few days out."
        ),
    }


def rbergomi_fit(data, spx) -> dict | None:
    from volsurface.fit.rbergomi import calibrate

    try:
        result = calibrate(data, H=None, seed=0, surface=spx["spx_sanos"].surface)
    except KeyError:
        return None
    p = result.detail["parameters"]
    return {
        "params": {"H": p["H"], "eta": p["nu"], "rho": p["rho"]},
        "objective": result.detail["objective"],
        "n_paths": result.detail["n_paths"],
        "n_steps": result.detail["n_steps"],
        "seed": 0,
        "method": (
            "At-the-money level and skew of all 40 expiries (volsurface.fit.rbergomi), H "
            "free, forward variance read off the saved SANOS surface, Monte Carlo with "
            "4,000 paths, 64 steps, seed 0, no error bars."
        ),
    }


def quality(env, market) -> dict:
    q = env_lib.fit_quality(env, market)
    return {
        "rmse_vol_points": round(q.rmse, 4),
        "n_used": q.n_used,
        "n_quotes": q.n_quotes,
        "rmse_by_expiry": [
            None if not np.isfinite(v) else round(float(v), 4) for v in q.by_expiry_rmse
        ],
    }


def main() -> None:
    started = time.perf_counter()
    path = chains.require_snapshot("SPX")
    snapshot = chains.load_snapshot(path)
    data = market_data_from_snapshot(snapshot)
    spx = env_lib.load_spx_presets()
    market = env_lib.otm_quotes(data)
    by_T = data.quotes.groupby("T")[["F", "df"]].first()
    out = {
        "data": {
            "underlying": "SPX (S&P 500 index), SPXW options only (PM-settled)",
            "snapshot": str(path.relative_to(ROOT)),
            "quote_time_utc": str(snapshot["quote_time"].iloc[0]),
            "index_print": float(snapshot["spot"].iloc[0]),
            "source": "CBOE public delayed quotes, mid prices",
            "n_quotes": int(len(market["T"])),
            "n_expiries": int(np.unique(market["T"]).size),
            "T_min": float(np.min(market["T"])),
            "T_max": float(np.max(market["T"])),
            "expiry_T": np.unique(market["T"]).tolist(),
            "forward": by_T["F"].tolist(),
            "discount_factor": by_T["df"].tolist(),
            "forward_note": (
                "Forward and discount factor per expiry from put-call parity on the eight "
                "strikes nearest the money; no separate rate or dividend input. Moneyness "
                "is k = log(K/F) against each expiry's own forward."
            ),
        },
        "models": {},
    }
    m = out["models"]

    sigma = float(np.mean(market["iv"]))
    m["flat"] = {
        "params": {"sigma": sigma},
        "method": "The single vol that minimises the implied-vol RMSE over all the quotes "
        "(their mean).",
    }
    m["flat"].update(quality(env_lib.flat_environment(sigma), market))
    print(f"flat {m['flat']['params']} rmse {m['flat']['rmse_vol_points']}", flush=True)

    m["sabr"] = sabr_fit(data)
    sabr_env = env_lib.fitted_sabr_environment(out)
    m["sabr"].update(quality(sabr_env, market))
    print(f"sabr rmse {m['sabr']['rmse_vol_points']}", flush=True)

    for key, name in (("spx_svi", "svi"), ("spx_sanos", "sanos")):
        m[name] = {"method": "The saved fit in data/presets/spx_fits.npz, unchanged."}
        m[name].update(quality(spx[key], market))
        print(f"{name} rmse {m[name]['rmse_vol_points']}", flush=True)

    rb = rbergomi_fit(data, spx)
    if rb is not None:
        rb_env = env_lib.rbergomi_environment(**rb["params"], xi0=env_lib.fitted_rbergomi_xi0(spx))
        rb.update(quality(rb_env, market))
        m["rbergomi"] = rb
        print(f"rbergomi {rb['params']} rmse {rb['rmse_vol_points']}", flush=True)

    m["heston"] = heston_fit(snapshot)
    m["heston"].update(quality(env_lib.heston_environment(**m["heston"]["params"]), market))
    print(f"heston {m['heston']['params']} rmse {m['heston']['rmse_vol_points']}", flush=True)

    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"{OUT.relative_to(ROOT)} in {time.perf_counter() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
