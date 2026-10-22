"""Check the 14 October session 00 Heston fit against the current one.

Run from the repository root:

    python scripts/heston_fit_old_vs_new.py

See docs/notes/heston_fit_old_vs_new.md. The 14 October parameters are the
ones printed in LOG.md, to their printed precision. Seeded; about a minute.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from volsurface import chains, models, smile

LO = np.array([0.01, 0.1, 0.01, 0.05, -0.99])
HI = np.array([0.15, 20.0, 0.15, 2.0, 0.99])
OLD = np.array([0.012593, 8.805, 0.044344, 1.740, -0.5334])
HALF_DIGIT = np.array([5e-7, 5e-4, 5e-7, 5e-4, 5e-5])
OLD_COST = 0.7454
N_DRAWS = 600

ts = smile.market_term_structure(
    chains.market_implied_vols(chains.load_snapshot(chains.latest_snapshot("SPX")))
)
T = ts["T"].to_numpy()
ATM = ts["atm"].to_numpy()
SKEW = ts["skew"].to_numpy()
SHOW = [int(np.argmin(np.abs(T - d / 365))) for d in (0.0, 1.42, 7.42, 351.0)]


def residuals(x: np.ndarray, iv_tol: float | None) -> np.ndarray:
    pred = smile.model_term_structure(models.Heston(*x), T, iv_tol=iv_tol)
    return np.concatenate([pred[:, 0] - ATM, pred[:, 1] - SKEW])


def cost(x: np.ndarray, iv_tol: float | None) -> float:
    r = residuals(x, iv_tol)
    return 0.5 * float(r @ r)


def skews(x: np.ndarray, iv_tol: float | None) -> str:
    pred = smile.model_term_structure(models.Heston(*x), T[SHOW], iv_tol=iv_tol)
    return " ".join(f"{-s:.3f}" for s in pred[:, 1])


def main() -> None:
    print(f"market skew at T={np.round(T[SHOW], 6)}: {' '.join(f'{-s:.3f}' for s in SKEW[SHOW])}")
    rng = np.random.default_rng(0)
    draws = OLD + rng.uniform(-1.0, 1.0, (N_DRAWS, 5)) * HALF_DIGIT
    costs = np.array([cost(x, None) for x in draws])
    print(
        f"old loss, {N_DRAWS} draws inside the printed precision: "
        f"cost {costs.min():.4f} to {costs.max():.4f}"
    )
    for lo, hi in ((0.74, 0.75), (0.83, 0.84)):
        print(f"  share in [{lo}, {hi}): {np.mean((costs >= lo) & (costs < hi)):.3f}")
    match = draws[int(np.argmin(np.abs(costs - OLD_COST)))]
    print(f"closest draw to {OLD_COST}: params {np.round(match, 7)}")
    print(f"  old loss: cost {cost(match, None):.4f}, skews {skews(match, None)}")
    print(f"  smooth loss (iv_tol=1e-8): cost {cost(match, 1e-8):.4f}, skews {skews(match, 1e-8)}")
    jac = least_squares(residuals, match, args=(1e-8,), bounds=(LO, HI), max_nfev=1).jac
    print(
        f"  smooth-loss gradient (v0, kappa, theta, xi, rho): "
        f"{np.round(jac.T @ residuals(match, 1e-8), 4)}"
    )
    polish = least_squares(residuals, match, args=(1e-8,), bounds=(LO, HI), method="trf")
    print(
        f"  one least_squares run from it on the smooth loss: cost {polish.cost:.4f}, "
        f"params {np.round(polish.x, 4)}, status {polish.status}"
    )


if __name__ == "__main__":
    main()
