"""COS against Lewis on random Heston draws over the parameter bounds.

Run from the repository root:

    python scripts/cos_stress.py

Seed 0 draws 600 parameter sets uniformly in the search coordinates of
``Heston.param_spec()``: ``v0``, ``kappa``, ``theta`` and ``xi`` log-uniform
between their bounds, ``rho`` uniform in ``[-0.95, 0.95]``. Each set gets one
maturity drawn from one day, one week, three months, one year and three
years. Puts with ``F = 100`` at ``k = -0.5, -0.086, 0, 0.3`` are priced by
Lewis (``price``) and COS (``cos_price``). The script prints how many sets
have a cash gap above 1e-4, the worst cash gap and worst relative gap with
their put price, maturity and kurtosis (relative gaps on puts worth at
least 0.01), and whether ``converged`` was false
on every set above 1e-4. Every row is written to
``docs/notes/cos_stress.csv``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from volsurface.models.heston import Heston

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "notes" / "cos_stress.csv"
N_DRAWS = 600
MATURITIES = (1.0 / 365.0, 7.0 / 365.0, 0.25, 1.0, 3.0)
LOG_K = np.array([-0.5, -0.086, 0.0, 0.3])
TOL = 1e-4
REL_FLOOR = 0.01


def _kurtosis(model: Heston, maturity: float) -> float:
    _c1, c2, c4 = model.cumulants(maturity)
    if not (np.isfinite(c4) and c2 > 0.0 and c4 > 0.0):
        return float("nan")
    return float(c4 / c2**2)


def main() -> None:
    rng = np.random.default_rng(0)
    specs = {spec.name: spec for spec in Heston.param_spec()}
    draws = {}
    for name in ("v0", "kappa", "theta", "xi"):
        lo, hi = specs[name].lower, specs[name].upper
        draws[name] = np.exp(rng.uniform(np.log(lo), np.log(hi), N_DRAWS))
    draws["rho"] = rng.uniform(-0.95, 0.95, N_DRAWS)
    maturity = rng.choice(MATURITIES, N_DRAWS)
    strike = 100.0 * np.exp(LOG_K)
    rows = []
    for i in range(N_DRAWS):
        params = {name: float(values[i]) for name, values in draws.items()}
        model = Heston(**params)
        T = float(maturity[i])
        lewis = np.asarray(model.price(100.0, strike, T, kind="put"), dtype=float)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            cos = model.cos_price(100.0, strike, T, kind="put")
        gap = np.abs(lewis - np.asarray(cos, dtype=float))
        j = int(np.argmax(gap))
        # Relative gaps only on puts worth at least one cent; below that the
        # ratio is noise on a price that rounds to zero.
        rel = np.where(lewis >= REL_FLOOR, gap / np.maximum(lewis, REL_FLOOR), 0.0)
        jr = int(np.argmax(rel))
        rows.append(
            {
                **params,
                "T": T,
                "kurtosis": _kurtosis(model, T),
                "max_gap": float(gap[j]),
                "put_at_max_gap": float(lewis[j]),
                "k_at_max_gap": float(LOG_K[j]),
                "max_rel_gap": float(rel[jr]),
                "put_at_max_rel": float(lewis[jr]),
                "k_at_max_rel": float(LOG_K[jr]),
                "converged": bool(cos.converged),
                "warned": any("Lewis" in str(w.message) for w in caught),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT, index=False)
    bad = frame[frame["max_gap"] > TOL]
    worst = frame.loc[frame["max_gap"].idxmax()]
    worst_rel = frame.loc[frame["max_rel_gap"].idxmax()]
    print(f"draws: {N_DRAWS}, maturities per draw: 1, strikes: {LOG_K.size}")
    print(f"sets with cash gap above {TOL:g}: {len(bad)}")
    print(f"  their maturities: {sorted(set(bad['T'].round(4)))}")
    print(f"  their kurtosis: {bad['kurtosis'].round(0).tolist()}")
    print(f"  converged false on all of them: {bool((~bad['converged']).all())}")
    print(
        f"worst cash gap: {worst['max_gap']:.3e} on a put of {worst['put_at_max_gap']:.4f}"
        f" (k={worst['k_at_max_gap']}, T={worst['T']:.4f}, kurtosis {worst['kurtosis']:.0f},"
        f" relative {worst['max_gap'] / worst['put_at_max_gap']:.2e},"
        f" converged {worst['converged']})"
    )
    print(
        f"worst relative gap: {worst_rel['max_rel_gap']:.3e} on a put of"
        f" {worst_rel['put_at_max_rel']:.3e} (k={worst_rel['k_at_max_rel']},"
        f" T={worst_rel['T']:.4f}, kurtosis {worst_rel['kurtosis']:.0f},"
        f" converged {worst_rel['converged']})"
    )
    quiet = bad[bad["converged"]]
    print(f"sets above {TOL:g} that did not warn: {len(quiet)}")
    print(f"not converged (warned): {int((~frame['converged']).sum())}")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
