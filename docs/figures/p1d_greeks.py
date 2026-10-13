"""Figure for P1d: what a greek costs, and where a finite-difference bump sits.

The reference is the Black-Scholes delta. Pathwise, autodiff and the
likelihood ratio estimate it by Monte Carlo on exact geometric Brownian
motion. The finite-difference panel bumps the closed form.

Run from the repository root:

    python docs/figures/p1d_greeks.py

Writes docs/figures/p1d_greeks.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from volsurface.core.types import Contract, Forward, Vanilla
from volsurface.greeks import Analytic, Autodiff, FiniteDifference, LikelihoodRatio, Pathwise
from volsurface.models.bs import BlackScholes
from volsurface.models.gbm import GBM

OUT = Path(__file__).with_name("p1d_greeks.png")


def _rmse(engine, model, contracts, truth, n_paths, seeds) -> tuple[float, int]:
    """Root mean square error over strikes and seeds. One ``n_evals`` from the first seed."""
    errors = []
    n_evals = 0
    for seed in seeds:
        got = engine.compute(model, contracts, [Forward()], n_paths=n_paths, seed=seed)
        errors.append(got.value[:, 0] - truth)
        n_evals = got.n_evals
    stacked = np.concatenate(errors)
    return float(np.sqrt(np.mean(stacked**2))), n_evals


def main() -> None:
    F, df, sigma, T = 100.0, 1.0, 0.2, 1.0
    k = np.linspace(-0.2, 0.2, 9)
    contracts = [Contract(Vanilla(float(F * np.exp(log_k))), T) for log_k in k]
    truth = Analytic(F=F, df=df).compute(
        BlackScholes(sigma=sigma), contracts, [Forward()],
    ).value[:, 0]
    model = GBM(sigma=sigma, scheme="exact", n_steps=1)
    counts = [1_000, 4_000, 16_000, 64_000]
    seeds = range(6)
    engines = [
        ("pathwise", Pathwise(F, df), "C0"),
        ("autodiff", Autodiff(F, df), "C1"),
        ("likelihood ratio", LikelihoodRatio(F, df), "C3"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.2))
    ax = axes[0]
    pathwise_start = None
    for label, engine, colour in engines:
        xs, ys = [], []
        for n in counts:
            error, n_evals = _rmse(engine, model, contracts, truth, n, seeds)
            xs.append(n_evals)
            ys.append(error)
        ax.loglog(xs, ys, marker="o", color=colour, label=label)
        if pathwise_start is None:
            pathwise_start = (xs[0], ys[0])
    # Slope -1/2: four times the paths, half the error. Anchored on pathwise.
    guide_x = np.array([pathwise_start[0], counts[-1]])
    guide_y = pathwise_start[1] * np.sqrt(pathwise_start[0] / guide_x)
    ax.loglog(guide_x, guide_y, color="0.4", lw=0.8, label="slope -1/2")
    ax.set_xlabel("evaluations (paths)")
    ax.set_ylabel("RMSE of the forward delta")
    ax.set_title("Same greek. The likelihood ratio spends more paths.")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    bumps = np.geomspace(1e-1, 1e-8, 15)
    atm = [contracts[len(contracts) // 2]]
    atm_truth = float(truth[len(contracts) // 2])
    error = np.empty(bumps.size)
    for i, h in enumerate(bumps):
        got = FiniteDifference(F=F, df=df, bump=float(h)).compute(
            BlackScholes(sigma=sigma), atm, [Forward()],
        )
        error[i] = abs(float(got.value[0, 0]) - atm_truth)
    ax.loglog(bumps, error, color="C2", marker="o", ms=3.5)
    valley = int(np.argmin(error))
    ax.plot(bumps[valley], error[valley], "o", color="k", ms=6, label="smallest error")
    ax.set_xlabel("bump in the forward")
    ax.set_ylabel("absolute error of the delta")
    ax.set_title("Finite difference. Too large truncates, too small rounds.")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    print(f"wrote {OUT}")
    print(f"valley bump={bumps[valley]:.3e} error={error[valley]:.3e}")


if __name__ == "__main__":
    main()
